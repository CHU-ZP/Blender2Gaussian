"""Visibility of surface samples from free interior space, by exact GPU ray casts (NVIDIA Warp).

A sample is visible if an unobstructed segment joins it, or one of its
footprint probes, to at least one viewpoint. Probes matter at occluder edges,
seams and silhouettes, where a Gaussian's centre is hidden but its tangent
footprint still covers visible pixels. Viewpoints are grid points that keep a
clearance from geometry and are mostly enclosed by it. Neither face
orientation nor crossing parity is used to reject points inside solids: both
proved unreliable on non-watertight assets with inconsistent normals, and a
viewpoint inside a closed cabinet only makes the test more conservative.
Surfaces are treated as double-sided because Gaussians have no back-face
culling. Layers closer than `epsilon` count as the same surface; any farther
layer can occlude.

Geometry is recentred before the float32 upload so that world coordinates of
about 100 m keep micrometre precision.
"""
import math

import numpy as np
from scipy.spatial import cKDTree
import warp as wp


@wp.func
def _segment_distance2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    edge = b - a
    length2 = wp.dot(edge, edge)
    t = wp.float64(0.0)
    if length2 > wp.float64(0.0):
        t = wp.clamp(wp.dot(p - a, edge) / length2, wp.float64(0.0), wp.float64(1.0))
    delta = p - a - t * edge
    return wp.dot(delta, delta)


@wp.func
def _triangle_distance2(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    # Edges handle degenerate and arbitrarily skinny triangles robustly.
    distance2 = wp.min(_segment_distance2(p, a, b),
                       wp.min(_segment_distance2(p, b, c), _segment_distance2(p, c, a)))
    normal = wp.cross(b - a, c - a)
    length2 = wp.dot(normal, normal)
    if length2 > wp.float64(0.0):
        signed = wp.dot(p - a, normal)
        projected = p - normal * (signed / length2)
        if wp.dot(wp.cross(b - a, projected - a), normal) >= wp.float64(0.0) and \
           wp.dot(wp.cross(c - b, projected - b), normal) >= wp.float64(0.0) and \
           wp.dot(wp.cross(a - c, projected - c), normal) >= wp.float64(0.0):
            distance2 = wp.min(distance2, signed * signed / length2)
    return distance2


@wp.kernel
def _near_surface_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3d),
                         triangles: wp.array(dtype=wp.vec3d), tolerance: wp.float64, guard: float,
                         near: wp.array(dtype=wp.int8)):
    i = wp.tid()
    p = points[i]
    centre = wp.vec3(wp.float32(p[0]), wp.float32(p[1]), wp.float32(p[2]))
    radius = wp.vec3(wp.float32(tolerance) + guard)
    query = wp.mesh_query_aabb(mesh, centre - radius, centre + radius)
    face = int(0)
    threshold = tolerance * tolerance
    while wp.mesh_query_aabb_next(query, face):
        if _triangle_distance2(p, triangles[3 * face], triangles[3 * face + 1],
                               triangles[3 * face + 2]) <= threshold:
            near[i] = wp.int8(1)
            return


def mesh_near_surface(tester, triangles, points, tolerance, chunk=262144):
    """Complete-Mesh narrow-band test, including skinny faces skipped by Warp point queries.

    The float32 BVH only supplies conservative AABB candidates. Their actual
    distances use the original float64 vertices and float64 query positions;
    segment fallbacks cover nearly degenerate faces. The inflated box accounts
    for float32 BVH/query rounding without relaxing the distance threshold.
    """
    local_tri = np.asarray(triangles, np.float64).reshape(-1, 3) - tester.origin
    local_points = np.asarray(points, np.float64) - tester.origin
    max_abs = max(float(np.abs(local_tri).max()), float(np.abs(local_points).max(initial=0)))
    guard = max(8 * float(np.spacing(np.float32(max_abs))), 1e-8)
    vertices = wp.array(local_tri, dtype=wp.vec3d, device=tester.device)
    result = []
    for start in range(0, len(points), chunk):
        p = local_points[start:start + chunk]
        near = wp.zeros(len(p), dtype=wp.int8, device=tester.device)
        wp.launch(_near_surface_kernel, dim=len(p), device=tester.device,
                  inputs=[tester.mesh.id, wp.array(p, dtype=wp.vec3d, device=tester.device),
                          vertices, float(tolerance), guard, near])
        result.append(near.numpy().astype(bool))
    return np.concatenate(result) if result else np.zeros(0, bool)


@wp.kernel
def _distance_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), max_distance: float,
                     distance: wp.array(dtype=wp.float32)):
    i = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[i], max_distance)
    distance[i] = max_distance
    if query.result:
        nearest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distance[i] = wp.length(points[i] - nearest)


def mesh_distance(tester, points, max_distance=1.):
    """Unsigned distance to the full cached Mesh, including open/double-sided faces."""
    points = np.asarray(points, np.float64)
    result = wp.zeros(len(points), dtype=wp.float32, device=tester.device)
    if len(points):
        wp.launch(_distance_kernel, dim=len(points), device=tester.device,
                  inputs=[tester.mesh.id, wp.array((points - tester.origin).astype(np.float32),
                                                  dtype=wp.vec3, device=tester.device), max_distance, result])
    return result.numpy()


@wp.kernel
def _probe_kernel(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                  max_t: float, distance: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    hit = wp.mesh_query_ray(mesh, origins[i], directions[j], max_t)
    if hit.result:
        distance[i, j] = hit.t
    else:
        distance[i, j] = -1.0


@wp.kernel
def _visible_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), neighbours: wp.array2d(dtype=wp.int32),
                    viewpoints: wp.array(dtype=wp.vec3), epsilon: float, visible: wp.array(dtype=wp.int8)):
    i = wp.tid()
    point = points[i]
    for k in range(neighbours.shape[1]):
        delta = viewpoints[neighbours[i, k]] - point
        length = wp.length(delta)
        direction = delta / length
        if not wp.mesh_query_ray_anyhit(mesh, point + epsilon * direction, direction, length - epsilon):
            visible[i] = wp.int8(1)
            return


@wp.kernel
def _normal_gap_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                       epsilon: float, max_t: float, gap: wp.array(dtype=wp.float32)):
    i = wp.tid()
    best = max_t
    for side in range(2):
        direction = normals[i] * (1.0 - 2.0 * float(side))
        hit = wp.mesh_query_ray(mesh, points[i] + epsilon * direction, direction, max_t)
        if hit.result:
            best = wp.min(best, hit.t + epsilon)
    gap[i] = best


@wp.kernel
def _ray_kernel(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                max_t: float, distance: wp.array(dtype=wp.float32)):
    i = wp.tid()
    hit = wp.mesh_query_ray(mesh, origins[i], directions[i], max_t)
    if hit.result:
        distance[i] = hit.t
    else:
        distance[i] = -1.0


@wp.kernel
def _face_kernel(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                 max_t: float, distance: wp.array(dtype=wp.float32), face: wp.array(dtype=wp.int32)):
    i = wp.tid()
    hit = wp.mesh_query_ray(mesh, origins[i], directions[i], max_t)
    if hit.result:
        distance[i] = hit.t
        face[i] = hit.face
    else:
        distance[i] = -1.0
        face[i] = -1


def fibonacci_directions(count):
    index = np.arange(count) + .5
    z = 1 - 2 * index / count
    angle = math.pi * (1 + 5 ** .5) * index
    radius = np.sqrt(1 - z * z)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


class VisibilityTester:
    """Select free-space viewpoints once, then test sample points in batches."""

    def __init__(self, triangles, bounds, spacing=.25, neighbours=64, distant=64, epsilon=1e-4,
                 directions=64, min_clearance=.1, min_enclosure=.75, margin=0., floor=False, device=None):
        wp.config.quiet = True
        wp.init()
        self.device = device or ('cuda:0' if wp.is_cuda_available() else 'cpu')
        low, high = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        self.origin = (low + high) / 2
        flat = (np.asarray(triangles, np.float64).reshape(-1, 3) - self.origin).astype(np.float32)
        self.mesh = wp.Mesh(points=wp.array(flat, dtype=wp.vec3, device=self.device),
                            indices=wp.array(np.arange(len(flat), dtype=np.int32), dtype=wp.int32, device=self.device))
        del flat
        self.neighbours, self.distant, self.epsilon = neighbours, distant, epsilon
        # A margin extends the viewpoint grid beyond the geometry, for open
        # scenes (objects without enclosing walls) that are viewed from outside.
        # With floor=True the margin is not applied below the scene: the lowest
        # geometry is the ground, which is never seen from underneath.
        below = [margin, margin, 0. if floor else margin]
        axes = [np.arange(a - m + spacing / 2, b + margin, spacing) for a, b, m in zip(low, high, below)]
        candidates = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        probe = self._probe(candidates, fibonacci_directions(directions), float(np.linalg.norm(high - low)))
        keep = (probe[:, 0] >= min_enclosure) & (probe[:, 1] >= min_clearance)
        self.viewpoints = candidates[keep]
        self.viewpoint_clearance = probe[keep, 1]
        # A tester without viewpoints still serves ray queries (_probe, normal_gap);
        # visible() then requires viewpoints.
        self.tree = cKDTree(self.viewpoints) if len(self.viewpoints) else None
        self.candidates, self.probe = candidates, probe
        self.report = dict(
            device=self.device, grid_spacing=spacing, margin=margin, floor=floor, candidates=len(candidates), viewpoints=len(self.viewpoints),
            directions=directions, min_clearance=min_clearance, min_enclosure=min_enclosure,
            neighbours=neighbours, distant=distant, epsilon=epsilon,
            rejected_enclosure=int((probe[:, 0] < min_enclosure).sum()),
            rejected_clearance=int((probe[:, 1] < min_clearance).sum()))

    @classmethod
    def for_scene(cls, triangles, bounds, mode='auto', spacing=.25, open_margin=2., min_indoor_fraction=.01, **options):
        """Indoor scenes keep enclosed viewpoints; open scenes (objects without
        walls) use unenclosed viewpoints up to open_margin outside the bounds,
        never below the lowest geometry. 'auto' picks open when fewer than
        min_indoor_fraction of the candidate grid points are enclosed."""
        if mode not in ('auto', 'indoor', 'open'):
            raise ValueError(f'Unknown visibility mode: {mode}')
        if mode != 'open':
            tester = cls(triangles, bounds, spacing=spacing, floor=True, **options)
            if mode == 'indoor' or len(tester.viewpoints) >= min_indoor_fraction * len(tester.candidates):
                tester.mode = tester.report['mode'] = 'indoor'
                return tester
            indoor_viewpoints = len(tester.viewpoints)
            del tester
        tester = cls(triangles, bounds, spacing=spacing, min_enclosure=0., margin=open_margin, floor=True, **options)
        tester.mode = tester.report['mode'] = 'open'
        if mode == 'auto':
            tester.report['indoor_viewpoints_rejected'] = indoor_viewpoints
        return tester

    def clearance(self, points, directions=64):
        """Distance from each point to the nearest surface (sampled directions)."""
        return self._probe(np.asarray(points, np.float64), fibonacci_directions(directions), 1e3)[:, 1]

    def first_hit(self, origins, directions):
        """Distance along each (origin, unit direction) ray to the first surface (inf if none)."""
        origins = np.asarray(origins, np.float64)
        out = np.empty(len(origins))
        for i in range(0, len(origins), 8192):
            o, d = origins[i:i + 8192], np.asarray(directions, np.float64)[i:i + 8192]
            t = wp.zeros(len(o), dtype=wp.float32, device=self.device)
            wp.launch(_ray_kernel, dim=len(o), device=self.device,
                      inputs=[self.mesh.id, wp.array((o - self.origin).astype(np.float32), dtype=wp.vec3, device=self.device),
                              wp.array(d.astype(np.float32), dtype=wp.vec3, device=self.device), 1e3, t])
            out[i:i + len(o)] = t.numpy()
        return np.where(out < 0, np.inf, out)

    def first_hit_face(self, origins, directions):
        """Distance (inf if none) and index in `triangles` (-1 if none) of the first surface
        along unit directions, from one origin or one per ray; a whole image at once."""
        directions = np.asarray(directions, np.float32)
        t = wp.zeros(len(directions), dtype=wp.float32, device=self.device)
        face = wp.zeros(len(directions), dtype=wp.int32, device=self.device)
        origins = np.broadcast_to((np.asarray(origins, np.float64) - self.origin).astype(np.float32), directions.shape)
        wp.launch(_face_kernel, dim=len(directions), device=self.device,
                  inputs=[self.mesh.id, wp.array(origins.copy(), dtype=wp.vec3, device=self.device),
                          wp.array(directions, dtype=wp.vec3, device=self.device), 1e3, t, face])
        t = t.numpy()
        return np.where(t < 0, np.inf, t), face.numpy()

    def _probe(self, origins, directions, max_distance, chunk=8192):
        """Per origin: hit fraction and nearest hit distance."""
        directions = wp.array(directions.astype(np.float32), dtype=wp.vec3, device=self.device)
        result = []
        for start in range(0, len(origins), chunk):
            o = (origins[start:start + chunk] - self.origin).astype(np.float32)
            distance = wp.zeros((len(o), directions.shape[0]), dtype=wp.float32, device=self.device)
            wp.launch(_probe_kernel, dim=distance.shape, device=self.device,
                      inputs=[self.mesh.id, wp.array(o, dtype=wp.vec3, device=self.device), directions,
                              max_distance, distance])
            d = distance.numpy()
            hits = d >= 0
            count = hits.sum(1)
            result.append(np.column_stack((count / d.shape[1], np.where(hits, d, np.inf).min(1))))
        return np.concatenate(result) if result else np.zeros((0, 2))

    def visible(self, points, axes=None, reach=2.):
        """Visibility of centres; with axes [N,2,3] (tangent sigma vectors) also of the
        four footprint probes centre +- reach*axis, tested only where the centre is hidden."""
        visible = self._visible(points)
        if axes is not None and not visible.all():
            hidden = np.flatnonzero(~visible)
            offsets = reach * np.asarray(axes, np.float64)[hidden]
            probes = np.asarray(points, np.float64)[hidden, None] + np.concatenate((offsets, -offsets), axis=1)
            visible[hidden] = self._visible(probes.reshape(-1, 3)).reshape(-1, 4).any(1)
        return visible

    def normal_gap(self, points, normals, max_distance=1.):
        """Distance to the nearest other surface along +-normal (max_distance if none)."""
        points = np.asarray(points, np.float64)
        gap = wp.zeros(len(points), dtype=wp.float32, device=self.device)
        if len(points):
            wp.launch(_normal_gap_kernel, dim=len(points), device=self.device,
                      inputs=[self.mesh.id, wp.array((points - self.origin).astype(np.float32), dtype=wp.vec3, device=self.device),
                              wp.array(np.asarray(normals, np.float32), dtype=wp.vec3, device=self.device),
                              self.epsilon, max_distance, gap])
        return gap.numpy()

    def _visible(self, points, chunk=1 << 21):
        points = np.asarray(points, np.float64)
        if not len(points):
            return np.zeros(0, bool)
        if self.tree is None:
            raise ValueError('No free-space viewpoints passed the enclosure/clearance tests')
        viewpoints = wp.array((self.viewpoints - self.origin).astype(np.float32), dtype=wp.vec3, device=self.device)
        k = min(self.neighbours, len(self.viewpoints))
        # Nearest viewpoints first, then a fixed spread over the whole set so
        # that surfaces seen only from across a room are not missed.
        spread = np.linspace(0, len(self.viewpoints) - 1, min(self.distant, len(self.viewpoints))).astype(np.int32)
        result = []
        for start in range(0, len(points), chunk):
            p = points[start:start + chunk]
            _, neighbours = self.tree.query(p, k=k)
            neighbours = np.asarray(neighbours, np.int32).reshape(len(p), k)
            neighbours = np.concatenate((neighbours, np.broadcast_to(spread, (len(p), len(spread)))), axis=1)
            visible = wp.zeros(len(p), dtype=wp.int8, device=self.device)
            wp.launch(_visible_kernel, dim=len(p), device=self.device,
                      inputs=[self.mesh.id, wp.array((p - self.origin).astype(np.float32), dtype=wp.vec3, device=self.device),
                              wp.array(neighbours, dtype=wp.int32, device=self.device), viewpoints,
                              self.epsilon, visible])
            result.append(visible.numpy().astype(bool))
        return np.concatenate(result)

"""Exact closest points on original float64 triangles, accelerated by a Warp BVH.

The native float32 nearest query supplies an upper bound. An original owning
face supplies another bound (including skinny faces skipped by the native
query). A conservatively inflated AABB then visits every possible closer
triangle and evaluates its original float64 geometry. Targets are detached;
the squared distance to the current closest point differentiates in torch.
"""
import numpy as np
import torch
import warp as wp


@wp.func
def _closest_segment(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    e = b - a
    t = wp.float64(0.0)
    if wp.dot(e, e) > wp.float64(0.0):
        t = wp.clamp(wp.dot(p - a, e) / wp.dot(e, e), wp.float64(0.0), wp.float64(1.0))
    return a + t * e


@wp.func
def _closest_triangle(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    best = _closest_segment(p, a, b)
    d2 = wp.dot(p - best, p - best)
    other = _closest_segment(p, b, c)
    if wp.dot(p - other, p - other) < d2:
        best = other
        d2 = wp.dot(p - other, p - other)
    other = _closest_segment(p, c, a)
    if wp.dot(p - other, p - other) < d2:
        best = other
        d2 = wp.dot(p - other, p - other)
    n = wp.cross(b - a, c - a)
    if wp.dot(n, n) > wp.float64(0.0):
        other = p - n * (wp.dot(p - a, n) / wp.dot(n, n))
        if wp.dot(wp.cross(b - a, other - a), n) >= wp.float64(0.0) and \
           wp.dot(wp.cross(c - b, other - b), n) >= wp.float64(0.0) and \
           wp.dot(wp.cross(a - c, other - c), n) >= wp.float64(0.0):
            if wp.dot(p - other, p - other) < d2:
                best = other
    return best


@wp.kernel
def _owning_kernel(vertices: wp.array(dtype=wp.vec3d), points: wp.array(dtype=wp.vec3),
                   owning: wp.array(dtype=wp.int32), targets: wp.array(dtype=wp.vec3),
                   normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    pf = points[i]
    p = wp.vec3d(wp.float64(pf[0]), wp.float64(pf[1]), wp.float64(pf[2]))
    face = owning[i]
    a, b, c = vertices[3*face], vertices[3*face+1], vertices[3*face+2]
    best = _closest_triangle(p, a, b, c)
    n = wp.cross(b-a, c-a)
    if wp.length(n) > wp.float64(0.0):
        n = n/wp.length(n)
    targets[i] = wp.vec3(wp.float32(best[0]), wp.float32(best[1]), wp.float32(best[2]))
    normals[i] = wp.vec3(wp.float32(n[0]), wp.float32(n[1]), wp.float32(n[2]))


@wp.kernel
def _closest_kernel(mesh: wp.uint64, vertices: wp.array(dtype=wp.vec3d),
                    points: wp.array(dtype=wp.vec3), owning: wp.array(dtype=wp.int32),
                    guard: float, targets: wp.array(dtype=wp.vec3),
                    normals: wp.array(dtype=wp.vec3), faces: wp.array(dtype=wp.int32)):
    i = wp.tid()
    pf = points[i]
    p = wp.vec3d(wp.float64(pf[0]), wp.float64(pf[1]), wp.float64(pf[2]))
    face = owning[i]
    best = _closest_triangle(p, vertices[3 * face], vertices[3 * face + 1], vertices[3 * face + 2])
    d2 = wp.dot(p - best, p - best)
    native = wp.mesh_query_point_no_sign(mesh, pf, wp.float32(wp.sqrt(d2)) + guard)
    if native.result:
        other = _closest_triangle(p, vertices[3 * native.face], vertices[3 * native.face + 1],
                                  vertices[3 * native.face + 2])
        if wp.dot(p - other, p - other) < d2:
            face = native.face
            best = other
            d2 = wp.dot(p - other, p - other)
    radius = wp.vec3(wp.float32(wp.sqrt(d2)) + guard)
    query = wp.mesh_query_aabb(mesh, pf - radius, pf + radius)
    candidate = int(0)
    while wp.mesh_query_aabb_next(query, candidate):
        other = _closest_triangle(p, vertices[3 * candidate], vertices[3 * candidate + 1],
                                  vertices[3 * candidate + 2])
        other_d2 = wp.dot(p - other, p - other)
        if other_d2 < d2:
            best = other
            face = candidate
            d2 = other_d2
    n = wp.cross(vertices[3 * face + 1] - vertices[3 * face], vertices[3 * face + 2] - vertices[3 * face])
    if wp.length(n) > wp.float64(0.0):
        n = n / wp.length(n)
    targets[i] = wp.vec3(wp.float32(best[0]), wp.float32(best[1]), wp.float32(best[2]))
    normals[i] = wp.vec3(wp.float32(n[0]), wp.float32(n[1]), wp.float32(n[2]))
    faces[i] = face


class MeshTargets:
    def __init__(self, triangles, origin, device='cuda:0'):
        wp.config.quiet = True
        wp.init()
        self.device = device
        flat = np.asarray(triangles, np.float64).reshape(-1, 3) - np.asarray(origin)
        self.guard = max(8 * float(np.spacing(np.float32(np.abs(flat).max()))), 1e-8)
        self.vertices = wp.array(flat, dtype=wp.vec3d, device=device)
        self.mesh = wp.Mesh(points=wp.array(flat.astype(np.float32), dtype=wp.vec3, device=device),
                            indices=wp.array(np.arange(len(flat), dtype=np.int32), dtype=wp.int32, device=device))

    @torch.no_grad()
    def closest_owning(self, positions, owning_faces):
        """Closest point on the original source triangle, with no face switching."""
        positions = positions.detach().contiguous()
        faces = owning_faces.to(torch.int32).contiguous()
        targets, normals = torch.empty_like(positions), torch.empty_like(positions)
        if len(positions):
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(positions.device))) \
                    if positions.is_cuda else wp.ScopedDevice('cpu'):
                wp.launch(_owning_kernel, len(positions), device=self.device,
                          inputs=[self.vertices, wp.from_torch(positions, dtype=wp.vec3),
                                  wp.from_torch(faces, dtype=wp.int32), wp.from_torch(targets, dtype=wp.vec3),
                                  wp.from_torch(normals, dtype=wp.vec3)])
        return targets, normals, faces

    @torch.no_grad()
    def closest(self, positions, owning_faces):
        positions = positions.detach().contiguous()
        owning_faces = owning_faces.to(torch.int32).contiguous()
        targets, normals = torch.empty_like(positions), torch.empty_like(positions)
        faces = torch.empty(len(positions), dtype=torch.int32, device=positions.device)
        if len(positions):
            # Use torch's current stream so target writes precede subsequent
            # torch operations, even when the caller uses a non-default stream.
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(positions.device))) \
                    if positions.is_cuda else wp.ScopedDevice('cpu'):
                wp.launch(_closest_kernel, len(positions), device=self.device,
                          inputs=[self.mesh.id, self.vertices, wp.from_torch(positions, dtype=wp.vec3),
                                  wp.from_torch(owning_faces, dtype=wp.int32), self.guard,
                                  wp.from_torch(targets, dtype=wp.vec3), wp.from_torch(normals, dtype=wp.vec3),
                                  wp.from_torch(faces, dtype=wp.int32)])
        return targets, normals, faces

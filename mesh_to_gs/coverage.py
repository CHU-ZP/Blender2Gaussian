"""Uncapped surface sampling and continuous, finite-footprint certificates.

No sample count controls the stopping condition. All nondegenerate source
faces are independent targets. Verification subdivision does not create a
Gaussian unless an actual probe has insufficient support. Certificates refer
to the actual serialized geometry, not a softened training distance metric.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from rtree.index import Index, Property
from scipy.spatial.transform import Rotation

from .adaptive import bilinear
from .color import linear_to_srgb, srgb_to_linear
from .surface import charts
from .support_constraints import project


@dataclass(frozen=True)
class CoverageConfig:
    max_spacing: float = .024
    threshold: float = .95
    construction_threshold: float = .98
    opacity: float = .99
    support_radius: float = 3.
    edge_tolerance: float = .001
    colour_error: float = .02
    plane_error: float = .0001
    normal_fraction: float = .02
    outside_support_threshold: float = .05
    checkpoint_cells: int = 10000

    def __post_init__(self):
        if not 0 < self.threshold < self.construction_threshold < self.opacity < 1:
            raise ValueError('Require 0 < coverage < construction < opacity < 1')
        for key in ('max_spacing', 'support_radius', 'edge_tolerance', 'colour_error',
                    'plane_error', 'normal_fraction', 'checkpoint_cells'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 0 < self.outside_support_threshold < 1:
            raise ValueError('outside_support_threshold must be between 0 and 1')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        if a.size:
            h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def spatial_index():
    properties = Property()
    properties.dimension = 3
    return Index(properties=properties)


def box(vertices):
    return tuple(np.r_[np.min(vertices, axis=0), np.max(vertices, axis=0)])


def footprint_box(point, scales, rotation, radius):
    extent = radius * np.sqrt((rotation ** 2) @ (scales ** 2))
    # Outward padding protects candidate completeness at touching AABBs.
    pad = 32 * np.finfo(float).eps * np.maximum(np.abs(point) + extent, 1.)
    return tuple(np.r_[point - extent - pad, point + extent + pad])


def energy(points, centers, scales, rotations):
    delta = np.asarray(points)[None] - np.asarray(centers)[:, None]
    local = np.einsum('npi,nij->npj', delta, rotations) / scales[:, None]
    return np.sum(local ** 2, axis=-1)


def support(q, opacity, radius=3.):
    u = np.where(q <= radius ** 2, opacity[:, None] * np.exp(-q / 2), 0.)
    return -np.expm1(np.log1p(-u).sum(axis=0))


def local_coordinates_with_error(vertices, centers, scales, rotations, vertex_error=None):
    eps = np.finfo(float).eps
    delta = vertices[None] - centers[:, None]
    error = 8 * eps * (np.abs(vertices)[None] + np.abs(centers)[:, None])
    if vertex_error is not None:
        error += np.asarray(vertex_error)[None]
    local = np.einsum('npi,nij->npj', delta, rotations) / scales[:, None]
    local_error = np.einsum('npi,nij->npj', error + 16 * eps * np.abs(delta),
                            np.abs(rotations)) / scales[:, None] + 16 * eps * np.abs(local)
    return local, local_error


def cell_vertex_error(bary, triangle):
    return 8 * np.finfo(float).eps * (np.abs(bary) @ np.abs(triangle))


def triangle_lower_bound(vertices, centers, scales, rotations, opacity, radius=3., vertex_error=None):
    """A conservative lower bound at EVERY point of the entire triangle."""
    if not len(centers):
        return 0.
    local, error = local_coordinates_with_error(vertices, centers, scales, rotations, vertex_error)
    q = np.sum((np.abs(local) + error) ** 2, axis=-1)
    # Inflate quadratic maxima, round the final bound down.
    qmax = q.max(axis=1) * (1 + 128 * np.finfo(float).eps) + 128 * np.finfo(float).eps
    value = support(qmax[:, None], np.asarray(opacity), radius)[0]
    return max(0., float(np.nextafter(value, -np.inf)))


def cell_lower_bounds(vertices, candidates, centers, scales, rotations, opacity, radius=3., vertex_error=None):
    """Vectorize independent cells with every candidate, without nearest-k padding.

    The ragged candidate lists are flattened, evaluated with the same rounding
    guard as triangle_lower_bound, and reduced separately for each source cell.
    Batching changes verification cost; it does not truncate any footprint.
    """
    counts = np.array([len(ids) for ids in candidates], np.int64)
    bounds = np.zeros(len(counts))
    if not counts.sum():
        return bounds
    ids = np.concatenate(candidates)
    cell = np.repeat(np.arange(len(counts)), counts)
    eps = np.finfo(float).eps
    v, p, r, sc = np.asarray(vertices)[cell], centers[ids], rotations[ids], scales[ids]
    delta = v - p[:, None]
    error = 8 * eps * (np.abs(v) + np.abs(p[:, None]))
    if vertex_error is not None:
        error += np.asarray(vertex_error)[cell]
    local = np.einsum('nvi,nij->nvj', delta, r) / sc[:, None]
    local_error = np.einsum('nvi,nij->nvj', error + 16 * eps * np.abs(delta), np.abs(r)) / sc[:, None]
    local_error += 16 * eps * np.abs(local)
    qmax = np.sum((np.abs(local) + local_error) ** 2, axis=-1).max(axis=1)
    qmax = qmax * (1 + 128 * eps) + 128 * eps
    u = np.where(qmax <= radius ** 2, opacity[ids] * np.exp(-qmax / 2), 0.)
    offsets = np.r_[0, np.cumsum(counts[:-1])]
    nonempty = counts > 0
    log_transmittance = np.add.reduceat(np.log1p(-u), offsets[nonempty])
    bounds[nonempty] = np.maximum(0., np.nextafter(-np.expm1(log_transmittance), -np.inf))
    return bounds


def outside_upper_bound(vertices, centers, scales, rotations, opacity, radius=3., vertex_error=None):
    """Conservative upper bound; callers must supply ALL AABB contributors.

    In each Gaussian's whitened frame, the triangle lies inside the AABB of
    its vertices. Distance from zero to that box is a lower energy bound.
    """
    if not len(centers):
        return 0.
    local, error = local_coordinates_with_error(vertices, centers, scales, rotations, vertex_error)
    low, high = (local - error).min(1), (local + error).max(1)
    distance = np.maximum(np.maximum(low, -high), 0.)
    qmin = np.maximum(0., np.sum(distance ** 2, axis=1) * (1 - 128 * np.finfo(float).eps)
                      - 128 * np.finfo(float).eps)
    value = support(qmin[:, None], np.asarray(opacity), radius)[0]
    return min(1., float(np.nextafter(value, np.inf)))


def minimum_triangle_energy_lower(vertices, center, scales, rotation):
    """Certified separation using the convex quadratic's supporting planes.

    Approximate closest points propose dual vectors z. For ANY such vector,
    min_y ||y||² >= 2 min_vertex z·vertex - ||z||². Thus least-squares and
    barycentric rounding cannot turn an overestimated distance into a false
    intersection rejection.
    """
    local, error = local_coordinates_with_error(vertices, center[None], scales[None], rotation[None])
    v, verr = local[0], error[0]
    proposals = list(v)
    for j in range(3):
        edge = v[(j + 1) % 3] - v[j]
        denominator = edge @ edge
        if denominator > 0:
            t = np.clip(-(v[j] @ edge) / denominator, 0, 1)
            proposals.append(v[j] + t * edge)
    edges = np.column_stack((v[1] - v[0], v[2] - v[0]))
    weight = np.linalg.lstsq(edges, -v[0], rcond=None)[0]
    proposals.append(v[0] + edges @ weight)
    z = np.asarray(proposals)
    dots = z @ v.T
    dot_error = np.abs(z) @ verr.T + 32 * np.finfo(float).eps * (np.abs(z) @ np.abs(v.T))
    norm_upper = np.sum(z ** 2, axis=1) * (1 + 32 * np.finfo(float).eps)
    dual = 2 * (dots - dot_error).min(axis=1) - norm_upper
    return max(0., float(np.nextafter(dual.max(), -np.inf)))


def texture_bounds(image, uv_low, uv_high):
    """Exact component extrema on a periodic bilinear UV rectangle.

    A bilinear function attains its extrema at rectangle corners. Partition
    at every texel interpolation knot, including wrap crossings. Unlike a
    raw pixel min/max, this bound converges on subpixel footprints.
    """
    h, w = image.shape[:2]
    low = np.array([uv_low[0] * w - .5, (1 - uv_high[1]) * h - .5])
    high = np.array([uv_high[0] * w - .5, (1 - uv_low[1]) * h - .5])
    axes = []
    for j, period in enumerate((w, h)):
        if high[j] - low[j] >= period:
            axes.append(np.arange(period, dtype=float))
        else:
            axes.append(np.r_[low[j], np.arange(np.ceil(low[j]), np.floor(high[j]) + 1), high[j]])
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    # Blocks control memory only; all knots are visited.
    for start in range(0, len(axes[1]), 256):
        for xs in range(0, len(axes[0]), 256):
            xx, yy = np.meshgrid(axes[0][xs:xs + 256], axes[1][start:start + 256])
            uv = np.column_stack(((xx.ravel() + .5) / w, 1 - (yy.ravel() + .5) / h))
            values = bilinear(image, uv)
            minimum = np.minimum(minimum, values.min(0))
            maximum = np.maximum(maximum, values.max(0))
    return minimum, maximum


def split_cell(bary, triangle, metric=None):
    world = bary @ triangle
    edges = np.roll(world, -1, axis=0) - world
    lengths = np.sum(edges ** 2, axis=1) if metric is None else np.einsum('ni,ij,nj->n', edges, metric, edges)
    c = int(lengths.argmax())
    a, b, other = c, (c + 1) % 3, (c + 2) % 3
    midpoint = (bary[a] + bary[b]) / 2
    actual = midpoint @ triangle
    if np.array_equal(actual, world[a]) or np.array_equal(actual, world[b]):
        raise FloatingPointError('Verification subdivision reached coordinate precision')
    return np.stack((bary[a], midpoint, bary[other])), np.stack((midpoint, bary[b], bary[other]))


class GrowingArray:
    """Contiguous rows that grow as required; capacity is never a sample limit."""
    def __init__(self, shape=(), dtype=float, values=None):
        self.shape, self.dtype = shape, np.dtype(dtype)
        self.size = 0 if values is None else len(values)
        self.data = np.empty((max(256, self.size), *shape), self.dtype)
        if values is not None:
            self.data[:self.size] = values

    def append(self, value):
        self.reserve(self.size + 1)
        self.data[self.size] = value
        self.size += 1

    def reserve(self, count):
        if count > len(self.data):
            expanded = np.empty((max(256, 2 * len(self.data), count), *self.shape), self.dtype)
            expanded[:self.size] = self.data[:self.size]
            self.data = expanded

    def __len__(self):
        return self.size

    def __getitem__(self, rows):
        return self.data[:self.size][rows]

    def __iter__(self):
        return iter(self.data[:self.size])

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.data[:self.size], dtype=dtype)
        return array.copy() if copy else array


class Sampler:
    """One object/material group; chart ownership excludes unrelated sheets.

    Planar convex charts use directional half-plane boundary constraints.
    Other charts use a conservative circular collar. There is no iteration
    or Gaussian limit. Checkpoints include the pending target-cell queue.
    """
    def __init__(self, triangles, vertex_ids, uv=None, image=None, colour=(.5, .5, .5),
                 config=None, state=None):
        self.config = config or CoverageConfig()
        self.original_face_count = len(triangles)
        self.surface = charts(triangles, vertex_ids, uv, crease_degrees=0, retain_slivers=True, uv_tolerance=0)
        self.image = image  # scene-linear, top-down RGB
        self.image_colour_bounds = None
        if image is not None:
            low = np.asarray(image.min(axis=(0, 1)), float)
            high = np.asarray(image.max(axis=(0, 1)), float)
            # Every periodic bilinear value is a convex combination of texels.
            # Round outwards to also cover floating-point interpolation error.
            pad = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(np.abs(low), np.abs(high)))
            self.image_colour_bounds = low - pad, high + pad
        self.boundary_corner = np.zeros((len(self.surface.areas), 3), bool)
        self.boundary_corner[self.surface.boundary_faces, (self.surface.boundary_corners + 2) % 3] = True
        self.colour = np.asarray(srgb_to_linear(np.asarray(colour, float)))
        self.geometry_hash = array_digest(self.surface.triangles, self.surface.labels,
                                         np.empty(0) if self.surface.uv is None else self.surface.uv)
        self.material_hash = array_digest(self.colour, np.empty(0) if image is None else image)
        self.index = spatial_index()
        self.edges = {}
        self.texture_charts = {}
        self.source_index = None
        self.jacobians = None
        if uv is not None:
            from .adaptive import uv_jacobians
            self.jacobians = uv_jacobians(self.surface, strict=True)
        self.row_specs = dict(points=((3,), float), scales=((3,), float), rotations=((3, 3), float),
                              opacities=((), float), faces=((), np.int64), bary=((3,), float),
                              colours=((3,), float), kinds=((), 'U32'), cert_faces=((), np.int64),
                              cert_bary=((3, 3), float), cert_bounds=((), float))
        for key, (shape, dtype) in self.row_specs.items():
            setattr(self, key, GrowingArray(shape, dtype))
        self.processed = 0
        root_barycentric = np.eye(3)
        self.pending = [(i, root_barycentric) for i in reversed(range(len(self.surface.areas)))]
        if state is not None:
            self.restore(state)

    def chart_edges(self, chart):
        if chart in self.edges:
            return self.edges[chart]
        s = self.surface
        ids = s.boundary_groups[chart]
        f, c = s.boundary_faces[ids], s.boundary_corners[ids]
        a, b = s.triangles[f, c], s.triangles[f, (c + 1) % 3]
        owner = s.face_groups[chart]
        normal = s.normals[owner[0]]
        m = np.cross(normal, b - a)
        m /= np.linalg.norm(m, axis=1)[:, None]
        vertices = s.triangles[owner].reshape(-1, 3)
        # Coplanarity and convexity are established from ALL chart vertices.
        tol = 128 * np.finfo(float).eps * max(1., np.abs(vertices).max())
        planar = np.max(np.abs((vertices - vertices[0]) @ normal)) <= tol
        convex = planar
        if convex:
            for start in range(0, len(a), 256):
                if np.any(np.einsum('vki,ki->vk', vertices[:, None] - a[None, start:start + 256],
                                    m[start:start + 256]) < -tol):
                    convex = False
                    break
        collar_factor = 1.
        if convex and len(a):
            # Parallel edge offsets form mitres at convex corners. Scale the
            # offset by sin(interior_angle/2), so even the furthest mitre is
            # within the declared Euclidean boundary tolerance.
            incident = {}
            for j in range(len(a)):
                for vertex in (a[j], b[j]):
                    incident.setdefault(tuple(vertex), []).append(j)
            for pair in incident.values():
                if len(pair) != 2:
                    convex = False
                if len(pair) == 2:
                    cosine = np.clip(m[pair[0]] @ m[pair[1]], -1., 1.)
                    collar_factor = min(collar_factor, np.sqrt(max(0., (1 + cosine) / 2)))
            if collar_factor <= 1e-7:
                # Sliver angles can lose the mitre factor to rounding. The
                # circular distance collar stays valid and retains the face.
                convex = False
        self.edges[chart] = a, b, m, convex, collar_factor
        return self.edges[chart]

    def candidates(self, vertices, chart=None):
        ids = np.fromiter(self.index.intersection(box(vertices)), dtype=np.int64)
        if chart is not None and len(ids):
            ids = ids[self.surface.labels[self.faces[ids]] == chart]
        return ids

    def parameters(self, ids):
        return self.points[ids], self.scales[ids], self.rotations[ids], self.opacities[ids]

    def reserve_rows(self, values):
        # Reserve every column before changing any row count. A failed
        # allocation therefore leaves a consistent resumable checkpoint.
        for key in values:
            array = getattr(self, key)
            array.reserve(len(array) + 1)

    def append_rows(self, values):
        sizes = {key: len(getattr(self, key)) for key in values}
        try:
            for key, value in values.items():
                getattr(self, key).append(value)
        except BaseException:
            for key, size in sizes.items():
                getattr(self, key).size = size
            raise

    def footprint_colour_error(self, point, covariance, face, colour):
        """Bound source colour on every possibly intersected face of a chart.

        UV continuity along an edge does not imply an identical Jacobian on
        its two faces. A nearly affine whole-chart map has a vertex-error
        certificate; otherwise query every source-face AABB and use that
        face's own UV affine map. No nearest-face cap is used.
        """
        if self.image_colour_bounds is not None:
            low, high = self.image_colour_bounds
            global_error = float(np.maximum(np.abs(low - colour), np.abs(high - colour)).max())
            if global_error <= .9 * self.config.colour_error:
                # A whole-image certificate covers every face mapping and UV
                # repeat. Avoid rescanning large, low-contrast texture regions.
                # Otherwise retain the precise local footprint calculation.
                return global_error
        s, radius = self.surface, self.config.support_radius
        chart = s.labels[face]
        if chart not in self.texture_charts:
            faces = s.face_groups[chart]
            reference = faces[0]
            jac = self.jacobians[reference]
            uv0, world0 = s.uv[reference, 0], s.triangles[reference, 0]
            predicted = (s.triangles[faces] - world0) @ jac.T + uv0
            rounding = 32 * np.finfo(float).eps * ((np.abs(s.triangles[faces]) + np.abs(world0)) @ np.abs(jac.T)
                                                  + np.abs(uv0) + np.abs(s.uv[faces]))
            error = (np.abs(predicted - s.uv[faces]) + rounding).max(axis=(0, 1))
            self.texture_charts[chart] = (reference, error) if error.max() <= 1e-10 else None
        affine = self.texture_charts[chart]
        if affine is not None:
            faces = [affine[0]]
        else:
            if self.source_index is None:
                self.source_index = spatial_index()
                for j, triangle in enumerate(s.triangles):
                    self.source_index.insert(j, box(triangle))
            extent = radius * np.sqrt(np.maximum(np.diag(covariance), 0))
            pad = 32 * np.finfo(float).eps * np.maximum(np.abs(point) + extent, 1.)
            ids = np.fromiter(self.source_index.intersection(tuple(np.r_[point - extent - pad, point + extent + pad])), np.int64)
            faces = ids[s.labels[ids] == chart]
            # Always include the birth face despite a rounding-only AABB miss.
            faces = np.unique(np.r_[faces, face])
        eigvalues, rotation = np.linalg.eigh(covariance)
        if eigvalues.min() <= 0:
            raise FloatingPointError('Footprint colour query reached covariance precision')
        scales = np.sqrt(eigvalues)
        worst = 0.
        for target_face in faces:
            if affine is None and target_face != face:
                if minimum_triangle_energy_lower(s.triangles[target_face], point, scales, rotation) > radius ** 2:
                    continue
            jac = self.jacobians[target_face]
            uv = s.uv[target_face, 0] + jac @ (point - s.triangles[target_face, 0])
            extent = radius * np.sqrt(np.maximum(np.diag(jac @ covariance @ jac.T), 0))
            rounding = 32 * np.finfo(float).eps * (np.abs(s.uv[target_face, 0])
                       + np.abs(jac) @ (np.abs(point) + np.abs(s.triangles[target_face, 0])))
            extent += rounding + (affine[1] if affine is not None else 0.)
            lo, hi = texture_bounds(self.image, uv - extent, uv + extent)
            worst = max(worst, float(np.maximum(np.abs(lo - colour), np.abs(hi - colour)).max()))
        return worst

    def add(self, face, bary, kind='deficit', covariance=None, colour=None, opacity=None):
        """Constrain a new point or a learned seed to the same quality contract."""
        s, cfg = self.surface, self.config
        point, normal = bary @ s.triangles[face], s.normals[face]
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = np.cross(axis, normal)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        frame = np.column_stack((tangent, bitangent, normal))
        C = np.eye(2) * (.8 * cfg.max_spacing) ** 2
        if covariance is not None:
            C = frame[:, :2].T @ np.asarray(covariance, float) @ frame[:, :2]
            C = (C + C.T) / 2
            values = np.linalg.eigvalsh(C)
            if not np.isfinite(C).all() or values.min() <= 0:
                raise ValueError('Learned seed must have positive finite tangent covariance')
            C *= min(1., (.8 * cfg.max_spacing) ** 2 / values.max())
        a, b, m3, convex, collar_factor = self.chart_edges(s.labels[face])
        # Finite support is certified at cfg.support_radius. Leave an extra
        # boundary margin for the renderer's ordinary tails: many thin
        # projected sheets can otherwise accumulate visible 3-sigma tails.
        boundary_radius = max(4., cfg.support_radius)
        if len(a):
            if convex:
                distance = np.maximum((m3 * (point - a)).sum(1), 0.)
                directions = m3 @ frame[:, :2]
                limit = (distance + .8 * cfg.edge_tolerance * collar_factor) / boundary_radius
                C = project(C[None], directions[None], limit[None])[0]
            else:
                e = b - a
                t = np.clip(((point - a) * e).sum(1) / (e * e).sum(1), 0, 1)
                distance = np.linalg.norm(point - (a + t[:, None] * e), axis=1).min()
                sigma = min(.8 * cfg.max_spacing, (distance + .8 * cfg.edge_tolerance) / boundary_radius)
                C *= min(1., sigma ** 2 / np.linalg.eigvalsh(C).max())
        # Limit curvature by plane approximation error, without normal inflation.
        kappa = s.curvature[face]
        if kappa > 0:
            limit2 = 2 * cfg.plane_error / kappa / cfg.support_radius ** 2
            C *= min(1., limit2 / np.linalg.eigvalsh(C).max())
        linear = self.colour.copy()
        if self.image is not None:
            uv = bary @ s.uv[face]
            linear = bilinear(self.image, uv[None])[0]
        if colour is not None:
            if not np.isfinite(colour).all():
                raise ValueError('Learned seed colour must be finite')
            linear = np.clip(colour, np.maximum(0., linear - .5 * cfg.colour_error),
                             np.minimum(1., linear + .5 * cfg.colour_error))
        if self.image is not None:
            jac = self.jacobians[face] @ frame[:, :2]
            guided = False
            # Directional exact bilinear derivative estimate guides shrinking;
            # the final footprint rectangle bound, not this probe, certifies it.
            while True:
                values, vectors = np.linalg.eigh(C)
                if values.min() <= 0:
                    raise FloatingPointError('Texture projection reached covariance precision')
                root = frame[:, :2] @ (vectors * np.sqrt(values))
                covariance = root @ root.T + np.outer(normal, normal) * cfg.normal_fraction ** 2 * values.min()
                error = self.footprint_colour_error(point, covariance, face, linear)
                if error <= cfg.colour_error * .9:
                    break
                if not guided:
                    # A large derivative of low-amplitude texture noise need
                    # not violate the colour tolerance. Check the exact bound
                    # first, and use gradients only to repair a failing footprint.
                    step = max(cfg.max_spacing * 1e-4, 1e-10)
                    gradient = np.column_stack([(bilinear(self.image, (uv + step * jac[:, j])[None])[0]
                                                 - bilinear(self.image, (uv - step * jac[:, j])[None])[0]) / (2 * step)
                                                for j in range(2)])
                    gvalue, gvector = np.linalg.eigh(gradient.T @ gradient)
                    limits = cfg.colour_error * .5 / (cfg.support_radius * np.sqrt(np.maximum(gvalue, 1e-30)))
                    C = project(C[None], gvector.T[None], limits[None])[0]
                    guided = True
                    continue
                # Shrink unresolved UV directions, rather than multiplying an
                # already-small gradient eigenvalue by the major-axis shrink.
                # Uniform shrinking here creates needless extreme anisotropy.
                extent = cfg.support_radius * np.sqrt(np.maximum(np.diag(jac @ C @ jac.T), 0))
                axis_error = []
                for j in range(2):
                    delta = np.eye(2)[j] * extent[j]
                    lo, hi = texture_bounds(self.image, uv - delta, uv + delta)
                    axis_error.append(np.maximum(np.abs(lo - linear), np.abs(hi - linear)).max())
                norm = np.linalg.norm(jac, axis=1)
                take = norm > 0
                if max(axis_error) > cfg.colour_error * .25:
                    take &= np.asarray(axis_error) >= .7 * max(axis_error)
                if take.any():
                    directions = jac[take] / norm[take, None]
                    limits = np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', directions, C, directions), 0) * .5)
                    C = project(C[None], directions[None], limits[None])[0]
                else:
                    C *= .5
                if np.linalg.eigvalsh(C).min() <= np.finfo(float).tiny:
                    raise FloatingPointError('Texture footprint reached covariance precision')
        values, vectors = np.linalg.eigh(C)
        if values.min() <= 0:
            raise FloatingPointError('Boundary constraints produced singular covariance')
        rotation = frame.copy()
        rotation[:, :2] = frame[:, :2] @ vectors
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        scales = np.r_[np.sqrt(values), cfg.normal_fraction * np.sqrt(values.min())]
        # Use the exact double-precision PLY parameterization already in construction.
        quat = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        scales = np.exp(np.log(scales))
        i = len(self.points)
        alpha = cfg.opacity if opacity is None else float(opacity)
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('Learned seed opacity must be finite and strictly between 0 and 1')
        # gsplat 1.5.3 stops before compositing a contributor when its next
        # transmittance is <=1e-4. Near-one opacity can therefore leave a
        # visible hole at the previous alpha. At .99 this can only happen
        # once the previous alpha is already >=.99.
        alpha = min(alpha, cfg.opacity)
        row = dict(points=point, scales=scales, rotations=rotation, opacities=alpha,
                   faces=face, bary=bary, colours=linear, kinds=kind)
        self.reserve_rows(row)
        bounds = footprint_box(point, scales, rotation, cfg.support_radius)
        try:
            self.index.insert(i, bounds)
            self.append_rows(row)
        except BaseException:
            self.index.delete(i, bounds)
            for key in row:
                getattr(self, key).size = i
            raise

    def run(self, checkpoint=None, progress=None):
        cfg, s = self.config, self.surface
        probes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0],
                           [0, .5, .5], [.5, 0, .5], [1/3, 1/3, 1/3]])
        active = None
        try:
            while self.pending:
                # Keep the target in the queue until the iteration commits.
                # An interrupt during a column write or queue update can then
                # restore it without losing or duplicating certificate leaves.
                face, bary = self.pending[-1]
                active_pending_size, active_cert_size = len(self.pending), len(self.cert_faces)
                active = (face, bary)
                children, retry = [], False
                vertices = bary @ s.triangles[face]
                ids = self.candidates(vertices, s.labels[face])
                parameters = self.parameters(ids)
                bound = triangle_lower_bound(vertices, *parameters, cfg.support_radius,
                                             cell_vertex_error(bary, s.triangles[face])) if len(ids) else 0.
                if bound >= cfg.threshold:
                    row = dict(cert_faces=face, cert_bary=bary, cert_bounds=bound)
                    self.reserve_rows(row)
                    self.append_rows(row)
                else:
                    # Split large target regions before placing quality-scale samples.
                    longest = np.linalg.norm(vertices - np.roll(vertices, -1, axis=0), axis=1).max()
                    if longest > cfg.max_spacing:
                        children = [(face, child) for child in split_cell(bary, s.triangles[face])]
                    else:
                        pb = probes @ bary
                        world = pb @ s.triangles[face]
                        actual = support(energy(world, *parameters[:3]),
                                         parameters[3], cfg.support_radius) if len(ids) else np.zeros(len(pb))
                        worst = int(actual.argmin())
                        if actual[worst] < cfg.construction_threshold:
                            on_boundary = np.any(self.boundary_corner[face] & (pb[worst] == 0))
                            self.add(face, pb[worst], 'boundary' if on_boundary else 'deficit')
                            retry = True
                        else:
                            # Loose bound: verification subdivision alone, with the
                            # whitened metric of an influential actual contributor.
                            metric = None
                            if len(ids):
                                p, sc, r, _ = parameters
                                k = int(energy(world[[worst]], p, sc, r)[:, 0].argmin())
                                metric = (r[k] / sc[k]) @ (r[k] / sc[k]).T
                            children = [(face, child) for child in split_cell(bary, s.triangles[face], metric)]
                if not retry:
                    self.pending.pop()
                    self.pending.extend(children)
                self.processed += 1
                active = None
                if self.processed % cfg.checkpoint_cells == 0:
                    if checkpoint:
                        self.save(checkpoint)
                    if progress:
                        progress(self)
        except BaseException:
            # Put the active region back if it did not become a certificate or
            # children. Revalidation on resume is safe and never reduces targets.
            if active is not None:
                for key in ('cert_faces', 'cert_bary', 'cert_bounds'):
                    getattr(self, key).size = active_cert_size
                del self.pending[active_pending_size - 1:]
                self.pending.append(active)
            if checkpoint:
                self.save(checkpoint)
            raise
        if checkpoint:
            self.save(checkpoint)
        return self

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, geometry_hash=self.geometry_hash, material_hash=self.material_hash,
                 config=json.dumps(asdict(self.config)), processed=self.processed,
                 points=np.asarray(self.points).reshape(-1, 3), scales=np.asarray(self.scales).reshape(-1, 3),
                 rotations=np.asarray(self.rotations).reshape(-1, 3, 3), opacities=self.opacities,
                 faces=self.faces, bary=np.asarray(self.bary).reshape(-1, 3),
                 colours=np.asarray(self.colours).reshape(-1, 3), kinds=self.kinds,
                 cert_faces=self.cert_faces, cert_bary=np.asarray(self.cert_bary).reshape(-1, 3, 3),
                 cert_bounds=self.cert_bounds, pending_faces=[f for f, _ in self.pending],
                 pending_bary=np.asarray([b for _, b in self.pending]).reshape(-1, 3, 3))
        temporary.replace(path)

    def restore(self, path):
        with np.load(path, allow_pickle=False) as state:
            if (str(state['geometry_hash']) != self.geometry_hash or
                    str(state['material_hash']) != self.material_hash or
                    json.loads(str(state['config'])) != asdict(self.config)):
                raise ValueError('Checkpoint source, material or quality protocol changed')
            for key in ('points', 'scales', 'rotations', 'opacities', 'faces', 'bary', 'colours',
                        'kinds', 'cert_faces', 'cert_bary', 'cert_bounds'):
                shape, dtype = self.row_specs[key]
                setattr(self, key, GrowingArray(shape, dtype, state[key]))
            self.pending = list(zip(state['pending_faces'], state['pending_bary']))
            self.processed = int(state['processed'])
        for i, (p, sc, r) in enumerate(zip(self.points, self.scales, self.rotations)):
            self.index.insert(i, footprint_box(p, sc, r, self.config.support_radius))

    def export(self, directory):
        """Write float64 geometry, reload it and certify the serialized model."""
        from plyfile import PlyData, PlyElement
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                      'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        p = np.asarray(self.points).reshape(-1, 3)
        sc = np.asarray(self.scales).reshape(-1, 3)
        r = np.asarray(self.rotations).reshape(-1, 3, 3)
        q = Rotation.from_matrix(r).as_quat(scalar_first=True) if len(r) else np.empty((0, 4))
        colour = (np.asarray(linear_to_srgb(np.asarray(self.colours).reshape(-1, 3))) - .5) / .28209479177387814
        opacity = np.asarray(self.opacities)
        values = np.column_stack((p, r[:, :, 2], colour, np.log(opacity / (1 - opacity)), np.log(sc), q))
        data = np.empty(len(p), dtype=[(key, '<f8') for key in attributes])
        for i, key in enumerate(attributes):
            data[key] = values[:, i]
        model = directory / 'model.ply'
        PlyData([PlyElement.describe(data, 'vertex')], text=False, byte_order='<').write(model)
        target = directory / 'surface'
        target.mkdir(exist_ok=True)
        s = self.surface
        arrays = dict(triangles=s.triangles, chart=s.labels, source=s.source,
                      loop_triangle=s.source,
                      rejected_source_face=np.setdiff1d(np.arange(self.original_face_count), s.source),
                      uv=np.zeros((len(s.areas), 3, 2)) if s.uv is None else s.uv,
                      face=np.asarray(self.faces, np.int64), barycentric=np.asarray(self.bary).reshape(-1, 3)[:, :2],
                      boundary=np.any(self.boundary_corner[np.asarray(self.faces)] &
                                      (np.asarray(self.bary).reshape(-1, 3) == 0), axis=1), sample=np.arange(len(p)),
                      target_face=np.arange(len(s.areas)), target_state=np.ones(len(s.areas), np.uint8),
                      certificate_face=np.asarray(self.cert_faces, np.int64),
                      certificate_barycentric=np.asarray(self.cert_bary).reshape(-1, 3, 3),
                      repair_reason=np.asarray(self.kinds))
        for key, value in arrays.items():
            np.save(target / f'{key}.npy', value)
        report = certify_model(model, target, self.config)
        quality = self.validate_quality(model)
        if quality['status'] != 'complete':
            report['status'] = 'incomplete'
        report.update(geometry_sha256=self.geometry_hash, material_sha256=self.material_hash,
                      quality_protocol=asdict(self.config), skipped_degenerate_faces=s.skipped,
                      construction_boundary_radius=max(4., self.config.support_radius),
                      count_stop_condition=False, processed_cells=self.processed, quality=quality)
        (directory / 'coverage_report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report

    def validate_quality(self, model):
        """Recheck each actual footprint against source geometry/material.

        Every Gaussian is constrained to its own chart's edge collar. Their
        union therefore has zero effective contribution outside the union of
        all collars, including contributors from other charts. This proof
        does not discard foreign contributors from an outside upper bound.
        Ordinary infinite tails and gsplat's screen filter are measured by
        rendering, separately from finite spatial support.
        """
        p, sc, r, opacity = read_model(model)
        from plyfile import PlyData
        stored = PlyData.read(model, mmap='r')['vertex'].data
        stored_rgb = np.column_stack([stored[f'f_dc_{j}'] for j in range(3)]) * .28209479177387814 + .5
        stored_linear = srgb_to_linear(np.clip(stored_rgb, 0, 1))
        cfg, s = self.config, self.surface
        colour_error = center_error = boundary_excess = normal_error = 0.
        for i, face in enumerate(self.faces):
            a, b, m, convex, factor = self.chart_edges(s.labels[face])
            radius = cfg.support_radius
            if convex:
                distance = ((p[i] - a) * m).sum(1)
                sigma = np.sqrt(((m @ r[i]) ** 2 * sc[i] ** 2).sum(1))
                excess = radius * sigma - distance - cfg.edge_tolerance * factor
                boundary_excess = max(boundary_excess, float(excess.max()))
            elif len(a):
                e = b - a
                t = np.clip(((p[i] - a) * e).sum(1) / (e * e).sum(1), 0, 1)
                distance = np.linalg.norm(p[i] - a - t[:, None] * e, axis=1).min()
                boundary_excess = max(boundary_excess, radius * sc[i].max() - distance - cfg.edge_tolerance)
            rebuilt = self.bary[i] @ s.triangles[face]
            center_error = max(center_error, float(np.linalg.norm(p[i] - rebuilt)))
            normal_error = max(normal_error, float(np.linalg.norm(r[i, :, 2] - s.normals[face])))
            if self.image is not None:
                covariance = (r[i] * sc[i]) @ (r[i] * sc[i]).T
                colour_error = max(colour_error, self.footprint_colour_error(p[i], covariance, face, stored_linear[i]))
            else:
                colour_error = max(colour_error, float(np.abs(stored_linear[i] - self.colour).max()))
        complete = (boundary_excess <= 1e-12 and center_error <= cfg.plane_error
                    and colour_error <= cfg.colour_error and normal_error <= 1e-10
                    and np.isfinite(opacity).all() and np.all((opacity > 0) & (opacity <= cfg.opacity + 1e-14)))
        return dict(status='complete' if complete else 'incomplete',
                    maximum_footprint_colour_error_linear=colour_error,
                    maximum_center_error_world=center_error,
                    maximum_normal_error=normal_error,
                    maximum_boundary_excess_world=boundary_excess,
                    maximum_opacity=float(opacity.max()) if len(opacity) else None,
                    opacity_ceiling=cfg.opacity,
                    outside_effective_support_upper_bound=0. if boundary_excess <= 1e-12 else None,
                    outside_domain='outside union of retained chart Euclidean edge collars in source planes',
                    colour_bound='exact extrema on periodic bilinear UV rectangles enclosing every effective footprint',
                    normals='source face normals; normal thickness is not inflated for certification')


def read_model(path):
    from plyfile import PlyData
    data = PlyData.read(path, mmap='r')['vertex'].data
    columns = lambda keys: np.column_stack([data[k] for k in keys]).astype(float)
    points = columns('xyz')
    scales = np.exp(columns([f'scale_{i}' for i in range(3)]))
    rotations = (Rotation.from_quat(columns([f'rot_{i}' for i in range(4)]), scalar_first=True).as_matrix()
                 if len(data) else np.empty((0, 3, 3)))
    opacity = 1 / (1 + np.exp(-np.asarray(data['opacity'], float)))
    return points, scales, rotations, opacity


def certify_model(model, target, config=None):
    """Recheck independent complete targets with current serialized parameters.

    A certificate partition must conserve EACH source face's barycentric
    area, so deleting a target or a failed leaf cannot create a passing report.
    """
    cfg = config or CoverageConfig()
    target = Path(target)
    triangles = np.load(target / 'triangles.npy', mmap_mode='r')
    chart = np.load(target / 'chart.npy')
    faces = np.load(target / 'face.npy')
    cf = np.load(target / 'certificate_face.npy')
    cb = np.load(target / 'certificate_barycentric.npy', mmap_mode='r')
    p, sc, r, opacity = read_model(model)
    index = spatial_index()
    for i in range(len(p)):
        index.insert(i, footprint_box(p[i], sc[i], r[i], cfg.support_radius))
    bounds = np.zeros(len(cf))
    # The cell batch only bounds temporary memory. Every R-tree contributor
    # remains eligible, including broad distant-centre splats and tiny seeds.
    for start in range(0, len(cf), 128):
        stop = min(start + 128, len(cf))
        source = triangles[cf[start:stop]]
        bary = cb[start:stop]
        vertices = np.matmul(bary, source)
        candidates = []
        for face, v in zip(cf[start:stop], vertices):
            ids = np.fromiter(index.intersection(box(v)), np.int64)
            candidates.append(ids[chart[faces[ids]] == chart[face]])
        bounds[start:stop] = cell_lower_bounds(vertices, candidates, p, sc, r, opacity, cfg.support_radius,
                                              8 * np.finfo(float).eps * np.matmul(np.abs(bary), np.abs(source)))
    determinants = np.abs(np.linalg.det(cb))
    partition = np.bincount(cf, weights=determinants, minlength=len(triangles))
    partition_ok = np.all(np.abs(partition - 1) <= 1e-9)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    passed = bounds >= cfg.threshold
    complete = bool(partition_ok and np.all(passed))
    np.save(target / 'certificate_lower_bound.npy', bounds)
    return dict(status='complete' if complete else 'incomplete',
                model_sha256=digest(model), target_sha256=array_digest(triangles, chart, faces, cf, cb),
                gaussians=len(p), target_faces=len(triangles), certificate_cells=len(cf),
                minimum_lower_bound=float(bounds.min()) if len(bounds) else None,
                certified_area=float(np.sum(area[cf[passed]] * determinants[passed])),
                target_area=float(area.sum()), unresolved_cells=int((~passed).sum()),
                partition_complete=bool(partition_ok),
                partition_max_area_error=float(np.max(np.abs(partition - 1))) if len(partition) else 0.,
                coverage_definition='continuous spatial alpha in finite effective ellipsoids; not screen alpha',
                certificate_stage='serialized float64 PLY',
                boundary_definition='effective-support collar; Gaussian tails and screen filtering validated separately')


def probe_model(model, target, config=None, seed=20261103, probes_per_face=32):
    """Independent diagnostics, including all original vertices and edges.

    Probes can reveal a bug in certification; they never replace the bound.
    """
    cfg = config or CoverageConfig()
    target = Path(target)
    triangles = np.load(target / 'triangles.npy', mmap_mode='r')
    chart, faces = np.load(target / 'chart.npy'), np.load(target / 'face.npy')
    p, sc, r, opacity = read_model(model)
    index = spatial_index()
    for i in range(len(p)):
        index.insert(i, footprint_box(p[i], sc[i], r[i], cfg.support_radius))
    rng = np.random.default_rng(seed)
    fixed = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5],
                      [.5, 0, .5], [1/3, 1/3, 1/3]])
    worst = 1.
    failures = count = 0
    for face, triangle in enumerate(triangles):
        bary = np.r_[fixed, rng.dirichlet([1, 1, 1], probes_per_face)]
        world = bary @ triangle
        ids = np.fromiter(index.intersection(box(triangle)), np.int64)
        ids = ids[chart[faces[ids]] == chart[face]]
        value = support(energy(world, p[ids], sc[ids], r[ids]), opacity[ids], cfg.support_radius)
        worst = min(worst, float(value.min()))
        failures += int((value < cfg.threshold).sum())
        count += len(value)
    return dict(seed=seed, probes=count, minimum_support=worst, failed_probes=failures,
                includes='every source vertex, edge midpoint, centroid and independent random interior probes',
                replaces_continuous_certificate=False)

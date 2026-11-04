"""Finite, independent Mesh probes and batched hole repair; no continuous proof.

Sampling density follows world area/edge length, with a representative for
each connected chart. There is no per-source-triangle probe allocation.
Nearest neighbours give a lower estimate; failed probes are rechecked against
every centre within the maximum effective radius before generating new seeds.
"""
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
import json
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .adaptive import bilinear, uv_jacobians
from .color import linear_to_srgb, srgb_to_linear
from .surface import charts
from .support_constraints import project


@dataclass(frozen=True)
class SampledConfig:
    spacing: float = .024
    probe_spacing_ratio: float = .5
    threshold: float = .95
    opacity: float = .99
    overlap: float = .9
    normal_fraction: float = .02
    support_radius: float = 3.
    edge_tolerance: float = .001
    texture_weight: float = 2.
    colour_error: float = .02
    repair_rounds: int = 4
    failure_fraction: float = .001
    neighbours: int = 32
    batch_size: int = 8192
    seed: int = 20261104

    def __post_init__(self):
        for name in ('spacing', 'probe_spacing_ratio', 'overlap', 'normal_fraction',
                     'support_radius', 'edge_tolerance', 'colour_error'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.threshold < self.opacity < 1 or not 0 <= self.failure_fraction < 1:
            raise ValueError('Require 0 < threshold < opacity < 1 and failure_fraction in [0,1)')
        if not np.isfinite(self.texture_weight) or self.texture_weight < 0:
            raise ValueError('texture_weight must be finite and nonnegative')
        for name, minimum in (('neighbours', 1), ('batch_size', 1), ('repair_rounds', 0), ('seed', 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')


def frame(normals):
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent = np.cross(axis, normals)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return np.stack((tangent, np.cross(normals, tangent), normals), axis=-1)


def distributed_rows(weights, owner, counts, rng, *, return_fraction=False):
    """Stratify chart mass; use local sums when global sums lose small charts."""
    order = np.argsort(owner, kind='stable')
    mass = np.bincount(owner, weights=weights, minlength=len(counts))
    cumulative = np.cumsum(weights[order])
    starts = np.r_[0., np.cumsum(mass)[:-1]]
    labels = np.repeat(np.arange(len(counts)), counts)
    offsets = np.arange(len(labels)) - np.repeat(np.cumsum(counts) - counts, counts)
    targets = starts[labels] + (offsets + rng.random(len(labels))) / counts[labels] * mass[labels]
    # Zero-mass charts never receive rows. Outward rounding cannot cross the
    # end of a chart into a foreign sheet.
    targets = np.minimum(targets, np.nextafter(starts[labels] + mass[labels], -np.inf))
    index = np.minimum(np.searchsorted(cumulative, targets, side='right'), len(order) - 1)
    previous = np.r_[0., cumulative[:-1]]
    fraction = (targets - previous[index]) / weights[order[index]]
    # Adding a tiny chart to a large global sum can erase its entire mass.
    # Also detect summation-order drift before it silently biases selection.
    begin = np.searchsorted(owner[order], np.arange(len(counts)))
    end = np.searchsorted(owner[order], np.arange(len(counts)), side='right')
    active = np.flatnonzero(counts)
    represented = cumulative[end[active] - 1] - previous[begin[active]]
    unstable = active[np.abs(represented - mass[active]) > 1e-10 * mass[active]]
    unstable = np.union1d(unstable, labels[owner[order[index]] != labels])
    for chart in unstable:
        rows = np.flatnonzero(labels == chart)
        members = order[begin[chart]:end[chart]]
        local = np.cumsum(weights[members])
        target = (np.arange(len(rows)) + rng.random(len(rows))) / len(rows) * local[-1]
        target = np.minimum(target, np.nextafter(local[-1], -np.inf))
        chosen = np.searchsorted(local, target, side='right')
        index[rows] = begin[chart] + chosen
        fraction[rows] = (target - np.r_[0., local[:-1]][chosen]) / weights[members[chosen]]
    result = order[index], labels
    return (*result, fraction.clip(0, 1)) if return_fraction else result


def probe_summary(values, weights, boundary, threshold):
    failed = ~np.isfinite(values) | (values < threshold)
    result = dict(probes=len(values), failed_probes=int(failed.sum()),
                  minimum_support=float(values.min()) if len(values) else None)
    for name, mask in (('interior', ~boundary), ('boundary', boundary)):
        total = float(weights[mask].sum())
        bad = float(weights[mask & failed].sum())
        result[name] = dict(probes=int(mask.sum()), failed_probes=int((mask & failed).sum()),
                            measure=total, failed_measure=bad, failure_fraction=bad / total if total else 0.)
    endpoints = boundary & (weights == 0)
    result['endpoints'] = dict(probes=int(endpoints.sum()), failed_probes=int((endpoints & failed).sum()))
    return result


class SampledSurface:
    def __init__(self, triangles, vertex_ids, uv=None, texture=None, colour=(.5, .5, .5), config=None):
        self.config = config or SampledConfig()
        self.surface = charts(triangles, vertex_ids, uv, retain_slivers=True)
        if not len(self.surface.areas):
            raise ValueError('No nondegenerate surface to sample')
        self.texture = texture
        self.image = None if texture is None else texture.image
        self.colour = srgb_to_linear(np.asarray(colour, float))
        s, cfg = self.surface, self.config
        self.importance = np.ones(len(s.areas))
        self.jacobian = None
        self.constant_texture = False
        if self.image is not None:
            if s.uv is None or not np.isfinite(s.uv).all():
                raise ValueError('A texture requires finite face-corner UV coordinates')
            if self.image.ndim != 3 or self.image.shape[2] != 3 or not self.image.size or not np.isfinite(self.image).all():
                raise ValueError('Texture must be a nonempty finite linear RGB image')
            low, high = self.image.min((0, 1)), self.image.max((0, 1))
            self.constant_texture = bool(np.max(high - low) <= cfg.colour_error)
            if not self.constant_texture:
                self.jacobian = uv_jacobians(s, strict=True)
                for start in range(0, len(s.areas), cfg.batch_size):
                    sl = slice(start, start + cfg.batch_size)
                    score = texture.score(s.uv[sl].mean(1), self.jacobian[sl], cfg.spacing)
                    self.importance[sl] += cfg.texture_weight * score
        self.spacing = cfg.spacing / np.sqrt(self.importance)
        self.points = np.empty((0, 3))
        self.scales = np.empty((0, 3))
        self.rotations = np.empty((0, 3, 3))
        self.colours = np.empty((0, 3))
        self.faces = np.empty(0, np.int64)
        self.bary = np.empty((0, 3))
        self.boundary = np.empty(0, bool)
        self.rounds = []
        self._boundary_tree = None
        self._prepare_boundary()

    def _prepare_boundary(self):
        s, cfg = self.surface, self.config
        f, c = s.boundary_faces, s.boundary_corners
        self.edge_a = s.triangles[f, c]
        self.edge_b = s.triangles[f, (c + 1) % 3]
        self.chart_stride = 4 * cfg.support_radius * cfg.overlap * cfg.spacing * max(1., cfg.normal_fraction) + 1.
        if not len(f):
            return
        # Midpoint pieces make long segments discoverable near either endpoint.
        count = np.maximum(1, np.ceil(s.boundary_lengths / cfg.spacing).astype(np.int64))
        self.edge_ids = np.repeat(np.arange(len(f)), count)
        offset = np.arange(len(self.edge_ids)) - np.repeat(np.cumsum(count) - count, count)
        fraction = (offset + .5) / count[self.edge_ids]
        mid = self.edge_a[self.edge_ids] + fraction[:, None] * (self.edge_b - self.edge_a)[self.edge_ids]
        self._boundary_tree = cKDTree(np.c_[mid, s.labels[f[self.edge_ids]] * self.chart_stride])

    def probes(self, ratio, seed):
        s, cfg = self.surface, self.config
        rng = np.random.default_rng(seed)
        mass = s.areas * self.importance
        chart_mass = np.bincount(s.labels, weights=mass, minlength=len(s.component_areas))
        counts = np.maximum(1, np.ceil(chart_mass / (cfg.spacing * ratio) ** 2).astype(np.int64))
        face, labels = distributed_rows(mass, s.labels, counts, rng)
        random = rng.random((len(face), 2))
        root = np.sqrt(random[:, 0])
        bary = np.c_[1 - root, root * (1 - random[:, 1]), root * random[:, 1]]
        # Importance-weighted selection with inverse density weights estimates
        # surface area, rather than overweighting detailed charts.
        weights = chart_mass[labels] / counts[labels] / self.importance[face]
        boundary = np.zeros(len(face), bool)
        if len(s.boundary_faces):
            owner = s.labels[s.boundary_faces]
            lengths = s.boundary_lengths
            count = np.ceil(s.perimeter / (cfg.spacing * ratio)).astype(np.int64)
            edge, label, t = distributed_rows(lengths, owner, count, rng, return_fraction=True)
            eb = np.zeros((len(edge), 3))
            c = s.boundary_corners[edge]
            eb[np.arange(len(edge)), c] = 1 - t
            eb[np.arange(len(edge)), (c + 1) % 3] = t
            face = np.r_[face, s.boundary_faces[edge]]
            bary = np.r_[bary, eb]
            weights = np.r_[weights, s.perimeter[label] / count[label]]
            boundary = np.r_[boundary, np.ones(len(edge), bool)]
            # Explicit endpoints of edges resolvable at this world spacing.
            # Tiny charts already have independent area/edge probes; avoid
            # turning every original tessellation vertex into a probe.
            long = np.flatnonzero(lengths >= cfg.spacing * ratio)
            if len(long):
                cf = np.repeat(s.boundary_faces[long], 2)
                cc = np.stack((s.boundary_corners[long], (s.boundary_corners[long] + 1) % 3), 1).ravel()
                cb = np.eye(3)[cc]
                xyz = np.einsum('ni,nij->nj', cb, s.triangles[cf])
                _, keep = np.unique(np.c_[xyz, s.labels[cf]], axis=0, return_index=True)
                face, bary = np.r_[face, cf[keep]], np.r_[bary, cb[keep]]
                weights, boundary = np.r_[weights, np.zeros(len(keep))], np.r_[boundary, np.ones(len(keep), bool)]
        return face, bary, weights, boundary

    def append(self, faces, bary, boundary):
        if not len(faces):
            return
        s, cfg = self.surface, self.config
        points, scales, rotations, colours = [], [], [], []
        for start in range(0, len(faces), cfg.batch_size):
            f, ba = faces[start:start + cfg.batch_size], bary[start:start + cfg.batch_size]
            p = np.einsum('ni,nij->nj', ba, s.triangles[f])
            r = frame(s.normals[f])
            sigma = cfg.overlap * self.spacing[f]
            C = np.eye(2)[None] * sigma[:, None, None] ** 2
            if self._boundary_tree is not None:
                k = min(8, len(self.edge_ids))
                d, ids = self._boundary_tree.query(np.c_[p, s.labels[f] * self.chart_stride], k=k,
                    distance_upper_bound=cfg.support_radius * cfg.overlap * cfg.spacing + cfg.spacing, workers=1)
                ids, d = np.asarray(ids).reshape(len(p), k), np.asarray(d).reshape(len(p), k)
                valid = np.isfinite(d)
                e = self.edge_ids[np.minimum(ids, len(self.edge_ids) - 1)]
                a, vector = self.edge_a[e], (self.edge_b - self.edge_a)[e]
                n2 = np.sum(vector * vector, -1)
                t = np.clip(np.sum((p[:, None] - a) * vector, -1) / np.maximum(n2, 1e-300), 0, 1)
                delta = p[:, None] - a - t[..., None] * vector
                m = np.cross(s.normals[f, None], vector)
                # Radial endpoint constraints also handle corners.
                at_endpoint = np.linalg.norm(delta, axis=-1) <= 8 * np.finfo(float).eps * np.maximum(1., np.linalg.norm(p, axis=-1))[:, None]
                m = np.where((((t > 0) & (t < 1)) | at_endpoint)[..., None], m, delta)
                norm = np.linalg.norm(m, axis=-1)
                m /= np.maximum(norm, 1e-300)[..., None]
                perpendicular = np.abs(np.sum(delta * m, -1))
                directions = np.einsum('nki,nij->nkj', m, r[:, :, :2])
                limits = np.where(valid & (norm > 0), (perpendicular + cfg.edge_tolerance) / cfg.support_radius, np.inf)
                C = project(C, directions, limits)
            values, axes = np.linalg.eigh(C)
            values = np.maximum(values, 1e-24)
            r[:, :, :2] = r[:, :, :2] @ axes
            r[np.linalg.det(r) < 0, :, 0] *= -1
            sc = np.c_[np.sqrt(values), cfg.normal_fraction * np.sqrt(values[:, 0])]
            if self.image is None:
                rgb = np.broadcast_to(self.colour, (len(f), 3)).copy()
            else:
                uv = np.einsum('ni,nij->nj', ba, s.uv[f])
                rgb = bilinear(self.image, uv)
            points.append(p); scales.append(sc); rotations.append(r); colours.append(rgb)
        for key, chunks in (('points', points), ('scales', scales), ('rotations', rotations), ('colours', colours)):
            setattr(self, key, np.concatenate([getattr(self, key), *chunks]))
        self.faces = np.r_[self.faces, faces]
        self.bary = np.r_[self.bary, bary]
        self.boundary = np.r_[self.boundary, boundary]

    def evaluate(self, faces, bary, *, model=None):
        cfg, s = self.config, self.surface
        points, scales, rotations, opacity = ((self.points, self.scales, self.rotations,
                                              np.full(len(self.points), cfg.opacity)) if model is None else model)
        if len(points) != len(self.faces):
            raise ValueError('Serialized model rows do not match surface ownership')
        if not all(np.isfinite(a).all() for a in (points, scales, rotations, opacity)) or np.any(scales <= 0) or np.any((opacity < 0) | (opacity >= 1)):
            raise ValueError('Invalid Gaussian parameters')
        values = np.zeros(len(faces))
        if not len(points):
            return values, dict(full_radius_fallbacks=0)
        radius = np.nextafter(cfg.support_radius * scales.max(), np.inf)
        stride = max(self.chart_stride, 2 * radius + 1.)
        tree = cKDTree(np.c_[points, s.labels[self.faces] * stride])
        k = min(cfg.neighbours, len(points))
        fallbacks = 0
        def contributions(p, ids, rows):
            local = np.einsum('ni,nij->nj', p[rows] - points[ids], rotations[ids]) / scales[ids]
            q = np.sum(local * local, -1)
            u = np.where(q <= cfg.support_radius ** 2, opacity[ids] * np.exp(-q / 2), 0.)
            return np.log1p(-u)
        for start in range(0, len(faces), cfg.batch_size):
            f, ba = faces[start:start + cfg.batch_size], bary[start:start + cfg.batch_size]
            p = np.einsum('ni,nij->nj', ba, s.triangles[f])
            query = np.c_[p, s.labels[f] * stride]
            d, ids = tree.query(query, k=k, distance_upper_bound=radius, workers=1)
            ids, d = np.asarray(ids).reshape(len(p), k), np.asarray(d).reshape(len(p), k)
            valid = np.isfinite(d)
            rows = np.broadcast_to(np.arange(len(p))[:, None], ids.shape)[valid]
            total = np.bincount(rows, weights=contributions(p, ids[valid], rows), minlength=len(p))
            alpha = -np.expm1(total)
            failed = np.flatnonzero(alpha < cfg.threshold)
            if len(failed):
                lists = tree.query_ball_point(query[failed], radius, workers=1)
                counts = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
                if counts.sum():
                    neighbors = np.concatenate(lists).astype(np.int64)
                    rr = np.repeat(np.arange(len(failed)), counts)
                    total = np.bincount(rr, weights=contributions(p[failed], neighbors, rr), minlength=len(failed))
                    alpha[failed] = -np.expm1(total)
                fallbacks += len(failed)
            values[start:start + len(p)] = alpha
        return values, dict(full_radius_fallbacks=fallbacks)

    def fit(self, progress=None):
        cfg = self.config
        tick = time.perf_counter()
        initial = self.probes(1., cfg.seed)
        self.append(initial[0], initial[1], initial[3])
        repair = self.probes(cfg.probe_spacing_ratio, cfg.seed + 1)
        for iteration in range(cfg.repair_rounds + 1):
            values, queries = self.evaluate(repair[0], repair[1])
            failed = np.flatnonzero(values < cfg.threshold)
            event = dict(round=iteration, gaussians=len(self.points),
                         **probe_summary(values, repair[2], repair[3], cfg.threshold), **queries,
                         seconds=time.perf_counter() - tick)
            self.rounds.append(event)
            if progress:
                progress(event)
            if not len(failed) or iteration == cfg.repair_rounds:
                break
            # Only merge identical deficits. A world-size voxel can be much
            # larger than boundary-constrained supports, hiding nearby holes.
            xyz = np.einsum('ni,nij->nj', repair[1][failed], self.surface.triangles[repair[0][failed]])
            key = np.c_[self.surface.labels[repair[0][failed]], xyz, repair[3][failed]]
            _, keep = np.unique(key, axis=0, return_index=True)
            add = failed[np.sort(keep)]
            self.append(repair[0][add], repair[1][add], repair[3][add])
        return self

    def validate(self, *, model=None):
        # This seed is never used for fitting. Keep failures as diagnostics;
        # validation samples must not silently become additional training data.
        probes = self.probes(self.config.probe_spacing_ratio, self.config.seed + 2)
        values, query = self.evaluate(probes[0], probes[1], model=model)
        summary = probe_summary(values, probes[2], probes[3], self.config.threshold)
        passed = all(summary[k]['failure_fraction'] <= self.config.failure_fraction for k in ('interior', 'boundary'))
        passed &= summary['endpoints']['failed_probes'] == 0 and np.isfinite(values).all()
        summary.update(status='passed' if passed else 'failed', seed=self.config.seed + 2,
                       continuous_certificate=False, **query)
        failures = values < self.config.threshold
        return summary, dict(face=probes[0][failures], barycentric=probes[1][failures],
                             boundary=probes[3][failures], support=values[failures])

    def write_ply(self, path):
        from plyfile import PlyData, PlyElement
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        data = np.empty(len(self.points), dtype=[(key, '<f8') for key in names])
        q = Rotation.from_matrix(self.rotations).as_quat(scalar_first=True)
        values = np.c_[self.points, self.rotations[:, :, 2], (linear_to_srgb(self.colours) - .5) / .28209479177387814,
                       np.full(len(self.points), np.log(self.config.opacity / (1 - self.config.opacity))),
                       np.log(self.scales), q]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite serialized Gaussian')
        for i, name in enumerate(names):
            data[name] = values[:, i]
        PlyData([PlyElement.describe(data, 'vertex')]).write(path)

    def report(self, *, model=None):
        summary, failures = self.validate(model=model)
        return dict(method='sampled_surface_coverage', continuous_certificate=False,
                    configuration=asdict(self.config), validation=summary, rounds=self.rounds,
                    gaussians=len(self.points), target_faces=len(self.surface.areas),
                    charts=len(self.surface.component_areas), target_area=float(self.surface.areas.sum()),
                    boundary_length=float(self.surface.perimeter.sum()),
                    skipped_degenerate_faces=self.surface.skipped,
                    texture_mode=('constant_colour' if self.image is None else
                                  'global_range' if self.constant_texture else 'mip_gradient_density'),
                    limits=['Finite probes can miss subspacing holes', 'Boundary and colour rules are approximate',
                            'A fixed repair-round limit can leave reported deficits; no Gaussian count budget']), failures

    def export(self, directory):
        """Export a diagnostic model even on failure; validate actual PLY rows."""
        from .coverage import digest, read_model
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'model.ply'
        self.write_ply(path)
        report, failures = self.report(model=read_model(path))
        report.update(status=report['validation']['status'], model_sha256=digest(path),
                      validation_stage='serialized float64 PLY')
        np.savez(directory / 'surface.npz', triangles=self.surface.triangles,
                 source_face=self.surface.source, chart=self.surface.labels,
                 face=self.faces, barycentric=self.bary, boundary=self.boundary,
                 uv=self.surface.uv if self.surface.uv is not None else np.empty((0, 3, 2)))
        np.savez(directory / 'failed_probes.npz', **failures)
        report['surface_sha256'] = digest(directory / 'surface.npz')
        (directory / 'sampled_coverage_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return report

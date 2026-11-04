"""Project optimized seeds within their owning chart, then certify uncapped repair."""
import numpy as np


def closest_chart_points(points, owning_faces, surface):
    """Exact float64 triangle/segment search over every face in each chart.

    Returns current face and barycentrics, including clamped edges/corners.
    A nearby foreign sheet is never eligible, even if it is closer.
    """
    points = np.asarray(points, float)
    owning_faces = np.asarray(owning_faces, np.int64)
    if len(points) != len(owning_faces) or not np.isfinite(points).all():
        raise ValueError('Invalid learned point/face correspondence')
    if np.any((owning_faces < 0) | (owning_faces >= len(surface.triangles))):
        raise ValueError('Learned seed ownership is outside source targets')
    faces = owning_faces.copy()
    barycentric = np.empty((len(points), 3))
    for chart in np.unique(surface.labels[owning_faces]):
        rows = np.flatnonzero(surface.labels[owning_faces] == chart)
        p = points[rows]
        best = np.full(len(rows), np.inf)
        for face in surface.face_groups[chart]:
            triangle = surface.triangles[face]
            a, b, c = triangle
            ab, ac = b - a, c - a
            # Cross-product coordinates avoid subtracting nearly equal Gram
            # products for thin but positive-area source triangles.
            normal = np.cross(ab, ac)
            unit = normal / np.hypot(np.hypot(*normal[:2]), normal[2])
            v = np.cross(p - a, ac) @ unit / (normal @ unit)
            w = np.cross(ab, p - a) @ unit / (normal @ unit)
            bary = np.column_stack((1 - v - w, v, w))
            inside = (bary >= 0).all(1)
            candidates = [(bary @ triangle, bary, inside)]
            for j in range(3):
                edge = triangle[(j + 1) % 3] - triangle[j]
                length = np.hypot(np.hypot(*edge[:2]), edge[2])
                t = np.clip(((p - triangle[j]) @ (edge / length)) / length, 0, 1)
                segment_bary = np.zeros((len(p), 3))
                segment_bary[:, j] = 1 - t
                segment_bary[:, (j + 1) % 3] = t
                candidates.append((segment_bary @ triangle, segment_bary, np.ones(len(p), bool)))
            for q, bary, eligible in candidates:
                distance = np.linalg.norm(p - q, axis=1)
                take = eligible & (distance < best)
                best[take] = distance[take]
                faces[rows[take]] = face
                barycentric[rows[take]] = bary[take]
    projected = np.einsum('ni,nij->nj', barycentric, surface.triangles[faces])
    return projected, faces, barycentric


def seed_sampler(sampler, points, scales, rotations, opacity, colours, owning_faces,
                 checkpoint=None, progress=None):
    """Keep learned opacity and permitted colour/covariance, repairing geometry.

    The checkpoint's leading ``optimized`` rows are the committed seeds;
    pending source triangles remain untouched until all seeds are loaded.
    """
    projected, faces, bary = closest_chart_points(points, owning_faces, sampler.surface)
    committed = int((np.asarray(sampler.kinds) == 'optimized').sum())
    if committed > len(points):
        raise ValueError('Seed checkpoint belongs to a different learned model')
    try:
        for i in range(committed, len(points)):
            covariance = (rotations[i] * scales[i]) @ (rotations[i] * scales[i]).T
            sampler.add(faces[i], bary[i], 'optimized', covariance, colours[i], opacity[i])
            if checkpoint and (i + 1) % sampler.config.checkpoint_cells == 0:
                sampler.save(checkpoint)
                if progress:
                    progress(sampler)
    except BaseException:
        if checkpoint:
            sampler.save(checkpoint)
        raise
    if checkpoint:
        sampler.save(checkpoint)
    delta = np.linalg.norm(points - projected, axis=1)
    return dict(optimized_seeds=len(points), maximum_projection_distance=float(delta.max()) if len(delta) else 0.,
                mean_projection_distance=float(delta.mean()) if len(delta) else 0.,
                changed_owning_face=int((faces != owning_faces).sum()),
                projection='closest point over every triangle of owning chart; no foreign sheet or nearest-k cap')

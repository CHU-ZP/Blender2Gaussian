"""Per-Gaussian chart-boundary constraints rebuilt from surface metadata.

The initializer limits each Gaussian's tangent covariance across nearby chart
boundary segments: along the in-plane unit direction m (segment normal, or the
radial direction beyond a segment endpoint) sigma_m <= distance / multiple.
`constraints` rebuilds these limits for every Gaussian as (m, limit) pairs in
its fixed tangent frame, so that a support optimizer can project the
covariance back onto the feasible set after every step.

Segments are subdivided to at most `piece` length and indexed globally by
midpoint; each Gaussian keeps the K nearest pieces of its own chart within a
search radius. The radius should exceed the largest support the optimizer may
reach (e.g. 2 * 1.5 * reference sigma), so that boundaries an enlarged
footprint could cross are included. Segments shared with a coplanar face of
another chart (`seam`) may use a different multiple, as --seam-sigmas does.
"""
import numpy as np
from scipy.spatial import cKDTree


def chart_boundaries(triangles, chart, seam=None):
    """Boundary edges of every chart as (a, b, chart id, is_coplanar_seam)."""
    _, vertex = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    vertex = vertex.reshape(-1, 3)
    edges = np.sort(np.stack((vertex, np.roll(vertex, -1, axis=1)), -1).reshape(-1, 2), axis=1)
    face = np.repeat(np.arange(len(triangles)), 3)
    key = np.column_stack((edges, chart[face]))
    order = np.lexsort(key.T[::-1])
    key = key[order]
    start = np.r_[0, np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1]
    count = np.diff(np.r_[start, len(order)])
    # An edge is interior to a chart when that chart owns it twice.
    index = np.sort(order[start[count == 1]])
    f, c = index // 3, index % 3
    is_seam = np.zeros(len(index), bool) if seam is None else np.asarray(seam).reshape(-1)[index]
    return triangles[f, c], triangles[f, (c + 1) % 3], chart[f], is_seam


def constraints(points, normals, tangent, bitangent, sample_chart, boundaries, radius, k=8,
                multiple=2., seam_multiple=None, piece=.02, chunk=1 << 20, chart_local=False):
    """m [N,k,2] directions; limit [N,k] max sigma (inf = unused).

    chart_local=True reserves nearest-neighbour slots for the owning chart
    using a separated fourth coordinate. The historical global query remains
    the default for existing refinement runs; texture ablations opt in.
    """
    a, b, owner, seam = boundaries
    seam_multiple = multiple if seam_multiple is None else seam_multiple
    weight = np.where(seam, seam_multiple, multiple)
    keep = weight > 0
    a, b, owner, weight = a[keep], b[keep], owner[keep], weight[keep]
    m_out = np.zeros((len(points), k, 2), np.float32)
    limit_out = np.full((len(points), k), np.inf, np.float32)
    if not len(a) or not len(points):
        return m_out, limit_out
    parts = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / piece).astype(np.int64))
    sid = np.repeat(np.arange(len(a)), parts)
    offset = np.arange(parts.sum()) - np.repeat(np.cumsum(parts) - parts, parts)
    pa = a[sid] + (offset / parts[sid])[:, None] * (b - a)[sid]
    pb = pa + ((b - a) / parts[:, None])[sid]
    # Distances are measured to the parent segment, so piece ends are not treated as endpoints.
    midpoint = (pa + pb) / 2
    if chart_local:
        # A fourth coordinate separates charts by more than the entire query
        # radius. Foreign pieces cannot occupy the finite nearest-k slots.
        # Post-filtering a 3D nearest-k list by owner does not have this property.
        stride = float(np.max(radius)) + 2*piece + 1e-6
        midpoint = np.column_stack((midpoint, owner[sid].astype(float)*stride))
    tree = cKDTree(midpoint)
    kq = min(4 * k, len(pa))
    kout = min(k, kq)
    for s in range(0, len(points), chunk):
        p, n = points[s:s + chunk], normals[s:s + chunk]
        query = np.column_stack((p, sample_chart[s:s + chunk].astype(float)*stride)) if chart_local else p
        dist, idx = tree.query(query, k=kq, distance_upper_bound=float(radius[s:s + chunk].max()) + piece, workers=-1)
        dist, idx = dist.reshape(len(p), kq), idx.reshape(len(p), kq)
        valid = idx < len(pa)
        j = np.where(valid, idx, 0)
        seg = sid[j]
        valid &= owner[seg] == sample_chart[s:s + chunk, None]
        aa, ee = a[seg], (b - a)[seg]
        t = np.clip(np.einsum('nki,nki->nk', p[:, None] - aa, ee) / np.maximum(np.einsum('nki,nki->nk', ee, ee), 1e-30), 0, 1)
        d = p[:, None] - (aa + t[..., None] * ee)
        d -= np.einsum('nki,ni->nk', d, n)[..., None] * n[:, None]
        inside = (t > 0) & (t < 1)
        m3 = np.where(inside[..., None], np.cross(n[:, None], ee), d)
        m3 /= np.maximum(np.linalg.norm(m3, axis=2)[..., None], 1e-30)
        perpendicular = np.abs(np.einsum('nki,nki->nk', d, m3))
        valid &= perpendicular <= radius[s:s + chunk, None]
        # Several pieces of one segment give the same constraint; keep one per segment.
        seg_key = np.where(valid, seg, -1)
        ordered = np.sort(seg_key, axis=1)
        duplicate = np.zeros_like(valid)
        rank = np.argsort(seg_key, axis=1, kind='stable')
        duplicate[np.arange(len(p))[:, None], rank[:, 1:]] = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
        valid &= ~duplicate
        score = np.where(valid, perpendicular, np.inf)
        pick = np.argsort(score, axis=1)[:, :k]
        rows = np.arange(len(p))[:, None]
        chosen = valid[rows, pick]
        m3p = m3[rows, pick]
        m_out[s:s + chunk, :kout, 0] = np.where(chosen, np.einsum('nki,ni->nk', m3p, tangent[s:s + chunk]), 0)
        m_out[s:s + chunk, :kout, 1] = np.where(chosen, np.einsum('nki,ni->nk', m3p, bitangent[s:s + chunk]), 0)
        limit_out[s:s + chunk, :kout] = np.where(chosen, np.maximum(perpendicular[rows, pick] / weight[seg[rows, pick]], 1e-9), np.inf)
    return m_out, limit_out


def project(C, m, limit, sweeps=1):
    """Sequentially shrink 2x2 covariances C [N,2,2] so that m^T C m <= limit^2 for each constraint.

    Same rank-one update as the initializer; works for numpy arrays and torch tensors.
    Each update only shrinks C, so later constraints never undo earlier ones:
    shrinking can only lower every m^T C m. One sweep therefore suffices;
    `sweeps` > 1 only guards against floating-point round-off.
    """
    for column in [c for _ in range(sweeps) for c in range(m.shape[1])]:
        mm = m[:, column]
        Cm = (C * mm[:, None, :]).sum(-1)
        variance = (Cm * mm).sum(-1)
        bound = limit[:, column] ** 2
        safe = variance.clip(1e-30, None)
        shrink = (1 - bound / safe).clip(0, None)
        C = C - (shrink / safe)[:, None, None] * Cm[:, :, None] * Cm[:, None, :]
    return C

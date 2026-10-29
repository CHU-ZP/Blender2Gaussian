"""Topology-aware, stratified surface initialization, in world length units.

Only geometric neighbours within one connected material/UV chart are used.
No camera images or photometric optimization are involved.
"""

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def allocate(weights, total, minimum=False):
    """Systematic area allocation: exact budget, no per-triangle ceil overhead."""
    weights = np.asarray(weights, np.float64)
    if len(weights) == 0 or total == 0:
        return np.zeros(len(weights), np.int64)
    base = int(minimum and total >= len(weights))
    cumulative = np.concatenate(([0.], np.cumsum(weights)))
    if cumulative[-1] <= 0:
        return np.zeros(len(weights), np.int64)
    rounded = np.floor(cumulative / cumulative[-1] * (total - base * len(weights)) + .5).astype(np.int64)
    return np.diff(rounded) + base


def triangle_strata(count, rng, triangle=None, importance=None):
    if count == 1:
        return np.full((1, 3), 1 / 3)
    if triangle is not None:
        # Exact equal-area cells, splitting the longest *physical* edge.
        # A barycentric lattice inherits a sliver's aspect ratio and can put
        # all nearest neighbours in one row, hiding the gaps between rows.
        cells = np.eye(3)[None]
        sizes = np.array([count],np.int64)
        leaves = []
        while len(sizes):
            final = sizes == 1
            if final.any():
                leaves.append(cells[final])
            cells,sizes = cells[~final],sizes[~final]
            if not len(sizes):
                break
            world = cells @ triangle
            lengths = np.sum((world-np.roll(world,-1,axis=1))**2,axis=2)
            edge = lengths.argmax(1)
            rows = np.arange(len(cells))
            a,b,c = cells[rows,edge],cells[rows,(edge+1)%3],cells[rows,(edge+2)%3]
            left = sizes//2
            fraction = left/sizes if importance is None else np.full(len(sizes),.5)
            split = (1-fraction[:,None])*a+fraction[:,None]*b
            child_left=np.stack((a,split,c),axis=1)
            child_right=np.stack((split,b,c),axis=1)
            if importance is not None:
                # Integrate each child's field at four interior locations.
                probes=np.array([[1/3]*3,[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
                l=importance(np.einsum('ki,nij->nkj',probes,child_left).reshape(-1,3)).reshape(-1,4).mean(1)
                r=importance(np.einsum('ki,nij->nkj',probes,child_right).reshape(-1,3)).reshape(-1,4).mean(1)
                left=np.clip(np.floor(sizes*l/(l+r)+.5).astype(np.int64),1,sizes-1)
            cells = np.concatenate((child_left,child_right))
            sizes = np.concatenate((left,sizes-left))
        cells = np.concatenate(leaves)
        random = rng.random((count,2))
        root = np.sqrt(random[:,0])
        local = np.stack((1-root,root*(1-random[:,1]),root*random[:,1]),axis=1)
        local = .7/3+.3*local
        return np.einsum('ni,nij->nj',local,cells)
    side = math.ceil(math.sqrt(count))
    cells = ((2 * np.arange(count, dtype=np.int64) + 1) * side ** 2) // (2 * count)
    row = side - np.ceil(np.sqrt(side ** 2 - cells)).astype(np.int64)
    offset = cells - (side ** 2 - (side - row) ** 2)
    center = (1 + offset % 2) / 3
    jitter = .3 * (rng.random((count, 2)) - .5)
    v = (offset // 2 + center + jitter[:, 0]) / side
    w = (row + center + jitter[:, 1]) / side
    return np.column_stack((1 - v - w, v, w))


@dataclass
class Surface:
    triangles: np.ndarray
    uv: np.ndarray | None
    normals: np.ndarray
    areas: np.ndarray
    labels: np.ndarray
    component_areas: np.ndarray
    boundary_faces: np.ndarray
    boundary_corners: np.ndarray
    boundary_lengths: np.ndarray
    perimeter: np.ndarray
    skipped: int
    curvature: np.ndarray | None = None

    @cached_property
    def face_groups(self):
        order = np.argsort(self.labels, kind='stable')
        return np.split(order, np.cumsum(np.bincount(self.labels))[:-1])

    @cached_property
    def boundary_groups(self):
        labels = self.labels[self.boundary_faces]
        order = np.argsort(labels, kind='stable')
        return np.split(order, np.cumsum(np.bincount(labels, minlength=len(self.component_areas)))[:-1])


def charts(triangles, vertex_ids, uv=None, crease_degrees=20):
    triangles = np.asarray(triangles, np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    edge2 = np.max(np.sum((triangles - np.roll(triangles, 1, axis=1)) ** 2, axis=2), axis=1)
    valid = (lengths > np.maximum(edge2 * 1e-12, 1e-18))
    skipped = int((~valid).sum())
    triangles, vertex_ids, lengths, cross = triangles[valid], vertex_ids[valid], lengths[valid], cross[valid]
    # Blender assets often duplicate vertices along shading splits. Restore
    # exact geometric adjacency inside this object/material before testing
    # crease angle and face-corner UV continuity. No proximity-based welding:
    # nearby parallel surfaces must remain separate.
    _, vertex_ids = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    vertex_ids = vertex_ids.reshape(-1, 3)
    uv = None if uv is None else np.asarray(uv)[valid]
    normals = cross / lengths[:, None]
    area = lengths / 2
    n = len(triangles)
    if not n:
        return Surface(triangles, uv, normals, area, np.empty(0, int), np.empty(0),
                       np.empty(0, int), np.empty(0, int), np.empty(0), np.empty(0), skipped)
    edges = vertex_ids[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1, len(order)]
    pairs = starts[:-1][np.diff(starts) == 2]
    left, right = order[pairs], order[pairs + 1]
    f0, f1 = left // 3, right // 3
    dot=np.einsum('ij,ij->i', normals[f0], normals[f1]).clip(-1,1)
    join = dot >= math.cos(math.radians(crease_degrees))
    curvature=np.zeros(n)
    # Measure the dual distance across the shared edge. Full centroid distance
    # contains an arbitrary along-edge offset on elongated triangles and would
    # underestimate curvature on e.g. a tall, finely tessellated cylinder.
    edge_vector=triangles[f0,(left%3+1)%3]-triangles[f0,left%3]
    dual=np.cross(edge_vector,normals[f0]+normals[f1])
    dual/=np.maximum(np.linalg.norm(dual,axis=1)[:,None],1e-30)
    separation=np.abs(np.sum((triangles[f0].mean(1)-triangles[f1].mean(1))*dual,axis=1))
    kappa=np.arccos(dot)/np.maximum(separation,1e-12)
    kappa[dot>1-1e-12]=0
    np.maximum.at(curvature,f0[join],kappa[join]);np.maximum.at(curvature,f1[join],kappa[join])
    if uv is not None:
        corners = np.stack((np.arange(len(edges)) % 3, (np.arange(len(edges)) % 3 + 1) % 3), axis=1)
        edge_uv = uv[np.arange(len(edges))[:, None] // 3, corners]
        edge_uv[edges[:, 0] > edges[:, 1]] = edge_uv[edges[:, 0] > edges[:, 1], ::-1]
        join &= np.max(np.abs(edge_uv[left] - edge_uv[right]), axis=(1, 2)) < 1e-5
    graph = coo_matrix((np.ones(int(join.sum()), np.uint8), (f0[join], f1[join])), shape=(n, n))
    components, labels = connected_components(graph, directed=False)
    boundary = np.ones(len(edges), bool)
    boundary[left[join]] = boundary[right[join]] = False
    indices = np.flatnonzero(boundary)
    faces, corners = indices // 3, indices % 3
    boundary_lengths = np.linalg.norm(triangles[faces, corners] - triangles[faces, (corners + 1) % 3], axis=1)
    return Surface(triangles, uv, normals, area, labels,
                   np.bincount(labels, weights=area, minlength=components), faces, corners,
                   boundary_lengths, np.bincount(labels[faces], weights=boundary_lengths, minlength=components), skipped,curvature)


def sample_component(surface, component, count, seed,texture=None,return_faces=False):
    faces = surface.face_groups[component]
    boundary = surface.boundary_groups[component]
    area = surface.component_areas[component]
    spacing = math.sqrt(area / count)
    boundary_count = min(count // 4, math.ceil(surface.perimeter[component] / max(.6 * spacing, 1e-15))) if len(boundary) else 0
    weights=surface.areas[faces]*surface.importance[faces] if hasattr(surface,'importance') else surface.areas[faces]
    allocations = allocate(weights, count - boundary_count)
    point_faces, barycentrics = [], []
    rng = np.random.default_rng(seed)
    for face, n in zip(faces, allocations):
        if n:
            point_faces.append(np.full(n, face, np.int32))
            field=None
            if texture is not None:
                from .adaptive import face_sampler
                field=face_sampler(surface,face,texture)
            barycentrics.append(triangle_strata(int(n), rng, surface.triangles[face],field))
    if boundary_count:
        lengths = surface.boundary_lengths[boundary]
        cumulative = np.r_[0, np.cumsum(lengths)]
        distances = (np.arange(boundary_count) + .5) / boundary_count * cumulative[-1]
        selected = np.minimum(np.searchsorted(cumulative, distances, side='right') - 1, len(boundary) - 1)
        fraction = (distances - cumulative[selected]) / lengths[selected]
        chosen = boundary[selected]
        f, c = surface.boundary_faces[chosen], surface.boundary_corners[chosen]
        # Inset by a fraction of the point spacing, not a fixed fraction of
        # triangle altitude (which can put wall-edge samples decimetres away).
        centroid_distance = 2*surface.areas[f]/(3*lengths[selected])
        inset = np.minimum(.1, .3*spacing/np.maximum(centroid_distance,1e-30))
        b = np.broadcast_to(inset[:,None]/3,(boundary_count,3)).copy()
        b[np.arange(boundary_count), c] += (1-inset) * (1 - fraction)
        b[np.arange(boundary_count), (c + 1) % 3] += (1-inset) * fraction
        point_faces.append(f.astype(np.int32))
        barycentrics.append(b)
    ids, bary = np.concatenate(point_faces), np.concatenate(barycentrics)
    points = np.einsum('ni,nij->nj', bary, surface.triangles[ids])
    normals = surface.normals[ids]
    uv = None if surface.uv is None else np.einsum('ni,nij->nj', bary, surface.uv[ids])
    segments = np.stack((surface.triangles[surface.boundary_faces[boundary], surface.boundary_corners[boundary]],
                         surface.triangles[surface.boundary_faces[boundary], (surface.boundary_corners[boundary] + 1) % 3]), axis=1)
    result=(points, normals, uv, segments, spacing, boundary_count)
    return result+(ids,) if return_faces else result


def covariance(points, normals, segments, spacing, overlap=.8, boundary_sigmas=2., chunk=32768,
               return_areas=False,max_normal_rotation=None,directional_support=False):
    """Estimate tangent spacing locally; cap covariance across chart boundaries."""
    count = len(points)
    axes = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent = np.cross(axes, normals)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    bitangent = np.cross(normals, tangent)
    tree = cKDTree(points)
    k = min(count,49 if directional_support else 13)
    # Subdivide long edges so nearest-midpoint queries also find long boundaries.
    if len(segments):
        parts = np.maximum(1, np.ceil(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) / (1.5 * spacing)).astype(int))
        segment_ids = np.repeat(np.arange(len(segments)), parts)
        offsets = np.arange(parts.sum()) - np.repeat(np.cumsum(parts) - parts, parts)
        fraction = offsets / parts[segment_ids]
        vec = segments[segment_ids, 1] - segments[segment_ids, 0]
        a = segments[segment_ids, 0] + fraction[:, None] * vec
        b = a + vec / parts[segment_ids, None]
        boundary_tree = cKDTree((a + b) / 2)
    else:
        boundary_tree = None
    scales = np.empty((count, 3), np.float32)
    quats = np.empty((count, 4), np.float32)
    capped = 0
    density_areas=np.empty(count,np.float32) if return_areas else None
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        p, n, t, bt = points[start:stop], normals[start:stop], tangent[start:stop], bitangent[start:stop]
        if k > 1:
            _, indices = tree.query(p, k=k, workers=-1 if len(p) > 4096 else 1)
            delta = points[indices[:, 1:]] - p[:, None]
            good = (np.einsum('nki,ni->nk', normals[indices[:, 1:]], n) > .94)
            good &= np.abs(np.einsum('nki,ni->nk', delta, n)) < max(.3 * spacing, 1e-9)
            local = np.stack((np.einsum('nki,ni->nk', delta, t), np.einsum('nki,ni->nk', delta, bt)), axis=2)
            valid_count = good.sum(1)
            C = np.einsum('nki,nkj,nk->nij', local, local, good) / np.maximum(valid_count, 1)[:, None, None]
            values, vectors = np.linalg.eigh(C)
            local_area = np.pi * np.max(np.sum(local ** 2, axis=2) * good, axis=1) / np.maximum(valid_count, 1)
            if directional_support:
                # Approximate the local Voronoi cell in eight directions.
                # This measures uncovered directions, instead of mistaking
                # a dense line of samples for a dense two-dimensional field.
                angles=np.arange(8)*np.pi/4
                direction=np.column_stack((np.cos(angles),np.sin(angles)))
                radii=[];distance2=np.sum(local**2,axis=2)
                for u in direction:
                    projection=np.einsum('nki,i->nk',local,u)
                    candidates=np.where(good&(projection>1e-9),.5*distance2/np.maximum(projection,1e-30),np.inf)
                    radii.append(np.min(candidates,axis=1))
                radii=np.stack(radii,axis=1)
                # A dense row can fill the nearest-neighbor list before
                # any sample across the next row is seen. Extend only
                # points with an unbounded or unusually broad direction.
                refine=np.any(~np.isfinite(radii)|(radii>1.25*spacing),axis=1)
                if refine.any() and count>k:
                    _,extra=tree.query(p[refine],k=min(count,257),workers=-1 if refine.sum()>4096 else 1)
                    dd=points[extra[:,1:]]-p[refine,None]
                    nn=n[refine]
                    accepted=np.einsum('nki,ni->nk',normals[extra[:,1:]],nn)>.94
                    accepted &= np.abs(np.einsum('nki,ni->nk',dd,nn))<max(.3*spacing,1e-9)
                    xy=np.stack((np.einsum('nki,ni->nk',dd,t[refine]),np.einsum('nki,ni->nk',dd,bt[refine])),axis=2)
                    rr=np.sum(xy**2,axis=2)
                    for column,u in enumerate(direction):
                        projection=np.einsum('nki,i->nk',xy,u)
                        candidate=np.where(accepted&(projection>1e-9),.5*rr/np.maximum(projection,1e-30),np.inf)
                        radii[refine,column]=np.min(candidate,axis=1)
                radii=np.where(np.isfinite(radii),radii,2*spacing).clip(.05*spacing,8*spacing)
                local_area=np.pi/8*np.sum(radii**2,axis=1)
                moment=np.einsum('nk,ki,kj->nij',radii**4,direction,direction)/np.maximum(np.sum(radii**2,axis=1),1e-30)[:,None,None]
                values,vectors=np.linalg.eigh(moment)
            local_area = np.clip(local_area, .25 * spacing ** 2, (16 if directional_support else 4) * spacing ** 2)
            local_area[valid_count<2]=spacing**2
            target = overlap ** 2 * local_area
            shape = np.sqrt(np.maximum(values[:, 1], 1e-30) / np.maximum(values[:, 0], 1e-30)).clip(1, 8 if directional_support else 4)
            values = np.stack((target / shape, target * shape), axis=1)
            values[valid_count < 2] = (overlap * spacing) ** 2
            C = np.einsum('nik,nk,njk->nij', vectors, values, vectors)
        else:
            C = np.tile(np.eye(2) * (overlap * spacing) ** 2, (len(p), 1, 1))
            local_area=np.full(len(p),spacing**2)
        if return_areas:density_areas[start:stop]=local_area
        if max_normal_rotation is not None and k>1:
            cosine=np.einsum('nki,ni->nk',normals[indices[:,1:]],n).clip(-1,1)
            theta=np.arccos(cosine);theta[cosine>1-1e-12]=0
            kappa=np.max(theta/np.maximum(np.linalg.norm(delta,axis=2),1e-12)*good,axis=1)
            limit=max_normal_rotation/np.maximum(kappa,1e-12)
            largest=np.linalg.eigvalsh(C)[:,1]
            C*=np.minimum(1,limit**2/np.maximum(largest,1e-30))[:,None,None]
        if boundary_tree is not None:
            nearest_count = min(4, len(a))
            _, near = boundary_tree.query(p, k=nearest_count, workers=-1 if len(p) > 4096 else 1)
            near = np.asarray(near).reshape(len(p), nearest_count)
            for column in range(nearest_count):
                aa, bb = a[near[:, column]], b[near[:, column]]
                edge = bb - aa
                fraction = np.einsum('ni,ni->n', p - aa, edge) / np.maximum(np.sum(edge * edge, axis=1), 1e-30)
                displacement = p - (aa + fraction.clip(0, 1)[:, None] * edge)
                m3 = np.cross(n, edge)
                m3 /= np.maximum(np.linalg.norm(m3, axis=1)[:, None], 1e-30)
                # A finite edge's endpoint is not an infinite crease. Outside
                # the segment constrain towards the endpoint, otherwise an
                # unrelated collinear edge can collapse a distant footprint.
                radial = displacement - np.sum(displacement*n, axis=1)[:, None]*n
                radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-30)
                m3 = np.where(((fraction < 0) | (fraction > 1))[:, None], radial, m3)
                m = np.stack((np.sum(m3 * t, axis=1), np.sum(m3 * bt, axis=1)), axis=1)
                distance = np.abs(np.sum(displacement * m3, axis=1))
                Cm = np.einsum('nij,nj->ni', C, m)
                variance = np.sum(m * Cm, axis=1)
                limit = np.maximum(distance / boundary_sigmas, spacing * 1e-5) ** 2
                shrink = np.maximum(1 - limit / np.maximum(variance, 1e-30), 0)
                C -= shrink[:, None, None] * np.einsum('ni,nj->nij', Cm, Cm) / np.maximum(variance, 1e-30)[:, None, None]
                capped += int((shrink > 1e-6).sum())
        values, vectors = np.linalg.eigh(C)
        values = np.maximum(values, (spacing * 1e-6) ** 2)
        major = vectors[:, :, 1]
        axis0 = major[:, 0:1] * t + major[:, 1:2] * bt
        axis1 = np.cross(n, axis0)
        rotation = np.stack((axis0, axis1, n), axis=2)
        quats[start:stop] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        scales[start:stop, 0] = np.sqrt(values[:, 1])
        scales[start:stop, 1] = np.sqrt(values[:, 0])
        scales[start:stop, 2] = .02 * np.sqrt(values[:, 0])
    result=(scales, quats, capped)
    return result+(density_areas,) if return_areas else result


def small_single_face_components(surface, components, count, seed, group_index,
                                 overlap=.8, boundary_sigmas=2.):
    """Batch disconnected 1--3 point facets without a tree per tiny chart.

    Neighbours are still strictly within each chart. All three triangle edges
    constrain the footprint, including slivers whose third edge a nearest-edge
    query might miss. RNG keys agree with the general per-component path.
    """
    faces = np.array([surface.face_groups[c][0] for c in components])
    triangles, normal = surface.triangles[faces], surface.normals[faces]
    if count == 1:
        bary = np.full((len(faces), 1, 3), 1/3)
    else:
        bary = np.stack([triangle_strata(count, np.random.default_rng(
            np.random.SeedSequence([seed, group_index, int(c)]))) for c in components])
    points = np.einsum('nki,nij->nkj', bary, triangles)
    h = np.sqrt(surface.areas[faces] / count)
    axes = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
    tangent = np.cross(axes, normal)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    bitangent = np.cross(normal, tangent)
    if count < 3:
        C = np.broadcast_to(np.eye(2), (len(faces), count, 2, 2)).copy()
        C *= (overlap*h)[:, None, None, None]**2
    else:
        delta = points[:, None] - points[:, :, None]
        local = np.stack((np.einsum('nkji,ni->nkj', delta, tangent),
                          np.einsum('nkji,ni->nkj', delta, bitangent)), axis=-1)
        C = np.einsum('nkji,nkjl->nkil', local, local)/(count-1)
        values, vectors = np.linalg.eigh(C)
        area = np.pi*np.max(np.sum(local**2, axis=-1), axis=2)/(count-1)
        target = overlap**2*np.clip(area, .25*h[:, None]**2, 4*h[:, None]**2)
        shape = np.sqrt(np.maximum(values[..., 1], 1e-30)/np.maximum(values[..., 0], 1e-30)).clip(1,4)
        values = np.stack((target/shape, target*shape), axis=-1)
        C = np.einsum('nkij,nkj,nklj->nkil', vectors, values, vectors)
    capped = 0
    for corner in range(3):
        edge = triangles[:, (corner+1)%3] - triangles[:, corner]
        m3 = np.cross(normal, edge)
        m3 /= np.linalg.norm(m3, axis=1)[:, None]
        m = np.stack((np.sum(m3*tangent, axis=1), np.sum(m3*bitangent, axis=1)), axis=1)
        distance = np.abs(np.einsum('nki,ni->nk', points-triangles[:, None, corner], m3))
        Cm = np.einsum('nkij,nj->nki', C, m)
        variance = np.einsum('nki,ni->nk', Cm, m)
        limit = np.maximum(distance/boundary_sigmas, h[:, None]*1e-5)**2
        shrink = np.maximum(1-limit/np.maximum(variance,1e-30),0)
        C -= (shrink/np.maximum(variance,1e-30))[...,None,None]*np.einsum('nki,nkj->nkij',Cm,Cm)
        capped += int((shrink>1e-6).sum())
    values, vectors = np.linalg.eigh(C)
    values = np.maximum(values,(h[:,None,None]*1e-6)**2)
    major = vectors[..., 1]
    axis0 = major[...,0,None]*tangent[:,None]+major[...,1,None]*bitangent[:,None]
    ns = np.broadcast_to(normal[:,None], points.shape)
    axis1 = np.cross(ns,axis0)
    q = Rotation.from_matrix(np.stack((axis0,axis1,ns),axis=-1).reshape(-1,3,3)).as_quat(scalar_first=True)
    scales = np.stack((np.sqrt(values[...,1]),np.sqrt(values[...,0]),.02*np.sqrt(values[...,0])),axis=-1)
    uv = None if surface.uv is None else np.einsum('nki,nij->nkj',bary,surface.uv[faces]).reshape(-1,2)
    return points.reshape(-1,3),ns.reshape(-1,3),uv,scales.reshape(-1,3).astype('float32'),q.astype('float32'),capped


def singleton_components(surface, components, overlap=.8, boundary_sigmas=2.):
    """Batch tiny multi-face charts with one point and no usable neighbours."""
    weights=surface.areas*(surface.importance if hasattr(surface,'importance') else 1)
    faces = np.array([surface.face_groups[c][np.searchsorted(
        np.cumsum(weights[surface.face_groups[c]]), .5*weights[surface.face_groups[c]].sum(), side='left')]
        for c in components])
    p,n = surface.triangles[faces].mean(1),surface.normals[faces]
    h = np.sqrt(surface.component_areas[components])
    axis = np.eye(3)[np.argmin(np.abs(n),axis=1)]
    t = np.cross(axis,n);t /= np.linalg.norm(t,axis=1)[:,None]
    bt = np.cross(n,t)
    C = np.eye(2)[None]*(overlap*h)[:,None,None]**2
    boundaries = [surface.boundary_groups[c] for c in components]
    sizes = np.array([len(b) for b in boundaries])
    capped = 0
    for corner in range(int(sizes.max(initial=0))):
        rows = np.flatnonzero(sizes>corner)
        edges = np.array([boundaries[r][corner] for r in rows])
        f,c = surface.boundary_faces[edges],surface.boundary_corners[edges]
        a,b = surface.triangles[f,c],surface.triangles[f,(c+1)%3]
        edge = b-a
        fraction = np.sum((p[rows]-a)*edge,axis=1)/np.maximum(np.sum(edge*edge,axis=1),1e-30)
        displacement = p[rows]-(a+fraction.clip(0,1)[:,None]*edge)
        m3 = np.cross(n[rows],edge)
        m3 /= np.maximum(np.linalg.norm(m3,axis=1)[:,None],1e-30)
        radial = displacement-np.sum(displacement*n[rows],axis=1)[:,None]*n[rows]
        radial /= np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-30)
        m3 = np.where(((fraction<0)|(fraction>1))[:,None],radial,m3)
        m = np.stack((np.sum(m3*t[rows],axis=1),np.sum(m3*bt[rows],axis=1)),axis=1)
        distance = np.abs(np.sum(displacement*m3,axis=1))
        Cm = np.einsum('nij,nj->ni',C[rows],m)
        variance = np.sum(m*Cm,axis=1)
        limit = np.maximum(distance/boundary_sigmas,h[rows]*1e-5)**2
        shrink = np.maximum(1-limit/np.maximum(variance,1e-30),0)
        C[rows] -= (shrink/np.maximum(variance,1e-30))[:,None,None]*np.einsum('ni,nj->nij',Cm,Cm)
        capped += int((shrink>1e-6).sum())
    values,vectors = np.linalg.eigh(C)
    values = np.maximum(values,(h[:,None]*1e-6)**2)
    major = vectors[:,:,1]
    axis0 = major[:,0,None]*t+major[:,1,None]*bt
    axis1 = np.cross(n,axis0)
    q = Rotation.from_matrix(np.stack((axis0,axis1,n),axis=-1)).as_quat(scalar_first=True)
    sc = np.stack((np.sqrt(values[:,1]),np.sqrt(values[:,0]),.02*np.sqrt(values[:,0])),axis=1)
    uv = None if surface.uv is None else surface.uv[faces].mean(1)
    return p,n,uv,sc.astype('float32'),q.astype('float32'),capped


def linear_texture(texture):
    image = np.asarray(texture, np.float32) / 255
    return np.where(image <= .04045, image / 12.92, ((image + .055) / 1.055) ** 2.4)


def texture_colors(texture, uv, linear=False):
    """Repeat/linear image sampling in scene-linear RGB, returned as sRGB."""
    image = texture if linear else linear_texture(texture)
    height, width = image.shape[:2]
    xy = np.column_stack((uv[:, 0] * width - .5, (1 - uv[:, 1]) * height - .5))
    lower = np.floor(xy).astype(np.int64)
    fraction = xy - lower
    x, y = lower[:, 0], lower[:, 1]
    rgb = ((1 - fraction[:, 0:1]) * (1 - fraction[:, 1:2]) * image[y % height, x % width]
           + fraction[:, 0:1] * (1 - fraction[:, 1:2]) * image[y % height, (x + 1) % width]
           + (1 - fraction[:, 0:1]) * fraction[:, 1:2] * image[(y + 1) % height, x % width]
           + fraction[:, 0:1] * fraction[:, 1:2] * image[(y + 1) % height, (x + 1) % width])
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - .055).astype(np.float32)

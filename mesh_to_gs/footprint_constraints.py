"""Training-time tangent-footprint limits at physical source-patch edges.

This uses source geometry only. Coplanar interior triangle edges are ignored.
Limits follow the original ancestor through splits and update with its centre.
"""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .surface import charts
from .support_constraints import chart_boundaries, constraints, project
from .joint_refinement import quaternion_matrix


def project_parameters(log_tangent, angle, directions, limits):
    variance = log_tangent.exp().square()
    cosine, sine = angle[:, 0].cos(), angle[:, 0].sin()
    covariance = torch.stack((
        cosine.square()*variance[:, 0]+sine.square()*variance[:, 1],
        cosine*sine*(variance[:, 0]-variance[:, 1]),
        cosine*sine*(variance[:, 0]-variance[:, 1]),
        sine.square()*variance[:, 0]+cosine.square()*variance[:, 1]), -1).reshape(-1, 2, 2)
    covariance = project(covariance, directions, limits)
    difference = covariance[:, 0, 0]-covariance[:, 1, 1]
    cross = covariance[:, 0, 1]
    root = torch.sqrt(difference.square()+4*cross.square())
    trace = covariance[:, 0, 0]+covariance[:, 1, 1]
    eigenvalues = torch.stack(((trace+root)/2, (trace-root)/2), -1).clamp(min=1e-10)
    theta = .5*torch.atan2(2*cross, difference)
    # Swap the tangent axes for equivalent orientations outside the existing
    # +/-45 degree parameter range, preserving the covariance and normal.
    mapped = torch.remainder(theta+math.pi/4, math.pi/2)-math.pi/4
    swapped = (theta-mapped).abs() > math.pi/4
    eigenvalues = torch.where(swapped[:, None], eigenvalues.flip(-1), eigenvalues)
    return .5*eigenvalues.log(), mapped[:, None]


class SourcePatchFootprints:
    def __init__(self, model, triangles, groups, group_ids, *, origin=0., multiple=2., floor=.001, radius=.125):
        self.parts = []
        self.multiple, self.floor = multiple, floor
        records = []
        active = torch.zeros(len(model.position), dtype=torch.bool, device=model.position.device)
        for group_id in group_ids:
            group = groups[group_id]
            first, last = group['faces']
            ids = torch.where((model.faces >= first) & (model.faces < last))[0]
            if not len(ids):
                continue
            if hasattr(model, 'transparent') and model.transparent[ids].any():
                raise ValueError('Physical footprint groups must be opaque')
            if not torch.allclose(model.log_ref[ids, 0], model.log_ref[ids, 1], atol=1e-6, rtol=0):
                raise ValueError('Footprint limits require isotropic raw source seeds')
            tri = np.asarray(triangles[first:last])-origin
            surface = charts(tri, np.arange(len(tri)*3).reshape(-1, 3),
                             crease_degrees=20, retain_slivers=True)
            if len(surface.source) != len(tri):
                raise ValueError('Source footprint geometry contains invalid triangles')
            face = model.faces[ids].cpu().numpy()-first
            positions = model.position_ref[ids].cpu().numpy().astype(float)
            rotation = Rotation.from_quat(model.q_ref[ids].cpu().numpy(), scalar_first=True).as_matrix()
            m, limit = constraints(positions, rotation[:, :, 2], rotation[:, :, 0], rotation[:, :, 1],
                surface.labels[face], chart_boundaries(surface.triangles, surface.labels),
                np.full(len(ids), radius), multiple=multiple, chart_local=True, chunk=32768)
            source_ids = ids.cpu().numpy()
            if not np.array_equal(source_ids, np.arange(source_ids[0], source_ids[-1]+1)):
                raise ValueError('Footprint groups require contiguous raw source rows')
            self.parts.append((first, last, int(source_ids[0]),
                torch.as_tensor(m, device=model.position.device),
                torch.as_tensor(limit*multiple, device=model.position.device)))
            active[ids] = True
            records.append(dict(group=group_id, object=group['object'], material=group['material'],
                source_rows=len(ids), physical_patches=len(surface.component_areas),
                boundary_rows=int(np.isfinite(limit).any(-1).sum())))
        self.record = dict(groups=records, boundary_sigmas=multiple, minimum_directional_sigma_m=floor,
            search_radius_m=radius, crease_degrees=20, interior_triangle_diagonals_ignored=True,
            behaviour='Contract tangent covariance after each update and split; allow sub-baseline edge scales',
            centre_motion='Source signed edge distance plus original-frame centre displacement; conservative endpoint planes')
        model.register_buffer('footprint_active', active)

    def save(self, directory):
        from .mesh_supervision import digest
        for record, (first, last, start, directions, distance) in zip(self.record['groups'], self.parts):
            key=str(record['group'])
            record.update(first_face=first, last_face=last, source_start=start)
            for label, value in [('directions', directions), ('distance', distance)]:
                path=directory/f'footprint_{key}_{label}.npy'
                np.save(path, value.cpu().numpy())
                record[label+'_sha256']=digest(path)

    @torch.no_grad()
    def project_(self, model):
        for first, last, source_start, directions, distance in self.parts:
            ids = torch.where((model.faces >= first) & (model.faces < last))[0]
            if not len(ids):
                continue
            slot = model.roots[ids]-source_start
            m = directions[slot]
            frame = quaternion_matrix(model.q_ref[ids])[:, :, :2]
            offset = model.position[ids]-model.position_ref[ids]
            planar = (offset[:, :, None]*frame).sum(1)
            d = distance[slot]+(m*planar[:, None]).sum(-1)
            limit = (d/self.multiple).clamp(min=self.floor)
            logs, rotation = project_parameters(model.log_tangent[ids], model.rotation[ids], m, limit)
            model.log_tangent[ids] = logs
            model.rotation[ids] = rotation

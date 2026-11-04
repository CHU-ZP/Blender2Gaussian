"""Per-Gaussian surface correspondences and the evaluated geometry they index."""
import json

import numpy as np
from numpy.lib.format import open_memmap


class SurfaceMetadata:
    """Stream a geometry cache and one correspondence row per Gaussian, in PLY order.

    `face` indexes the cache globally: object/material groups are stored
    consecutively, each with only the valid faces kept by `charts()`. The row
    index is the Gaussian's stable identity within this model; `sample` maps it
    to the unfiltered sampling order, which differs only when hidden samples
    were removed. Barycentrics keep the first two coordinates in float64; the
    third is 1 minus their sum.
    """

    def __init__(self, directory, surfaces, groups, count):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.face_offsets = np.r_[0, np.cumsum([len(s.areas) for s in surfaces])]
        self.chart_offsets = np.r_[0, np.cumsum([len(s.component_areas) for s in surfaces])]
        faces = int(self.face_offsets[-1])

        def array(name, dtype, shape):
            return open_memmap(directory / f'{name}.npy', 'w+', dtype, shape)
        self.triangles = array('triangles', np.float64, (faces, 3, 3))
        self.uv = array('uv', np.float32, (faces, 3, 2))
        self.chart = array('chart', np.int32, (faces,))
        self.loop_triangle = array('loop_triangle', np.int32, (faces,))
        self.face = array('face', np.int32, (count,))
        self.barycentric = array('barycentric', np.float64, (count, 2))
        self.boundary = array('boundary', np.bool_, (count,))
        self.sample = array('sample', np.int32, (count,))
        self.groups = []
        for i, (group, surface) in enumerate(zip(groups.values(), surfaces)):
            description = group['description']
            image = description['image']
            self.groups.append(dict(
                object=group['object_name'], object_index=group['object_index'],
                material=description['name'], image=None if image is None else image.name,
                color=description['color'], has_uv=surface.uv is not None,
                faces=self.face_offsets[i:i + 2].tolist(), charts=self.chart_offsets[i:i + 2].tolist()))

    def write_group(self, index, surface, loop_triangles):
        start, stop = self.face_offsets[index:index + 2]
        self.triangles[start:stop] = surface.triangles
        if surface.uv is not None:
            self.uv[start:stop] = surface.uv
        self.chart[start:stop] = surface.labels + self.chart_offsets[index]
        self.loop_triangle[start:stop] = loop_triangles

    def write_samples(self, index, start, faces, barycentric, boundary, samples):
        stop = start + len(faces)
        self.face[start:stop] = faces + self.face_offsets[index]
        self.barycentric[start:stop] = barycentric[:, :2]
        self.boundary[start:stop] = boundary
        self.sample[start:stop] = samples

    def finish(self, model, count, chunk=1 << 20):
        """Flush, then rebuild centres and normals from the cache and compare with the PLY."""
        from plyfile import PlyData
        arrays = (self.triangles, self.uv, self.chart, self.loop_triangle, self.face, self.barycentric,
                  self.boundary, self.sample)
        for array in arrays:
            array.flush()
        if count < len(self.face):
            # Rows were preallocated for the whole budget; drop the unused tail.
            for name in ('face', 'barycentric', 'boundary', 'sample'):
                kept = np.array(getattr(self, name)[:count])
                setattr(self, name, None)
                np.save(self.directory / f'{name}.npy', kept)
                setattr(self, name, np.load(self.directory / f'{name}.npy', mmap_mode='r'))
        (self.directory / 'groups.json').write_text(json.dumps(self.groups, indent=1) + '\n')
        data = PlyData.read(model, mmap='r')['vertex'].data
        worst_ulp = worst_normal = 0.
        inexact = 0
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            stored = np.column_stack([data[k][start:stop] for k in 'xyz'])
            normal = np.column_stack([data[k][start:stop] for k in ('nx', 'ny', 'nz')])
            points, rebuilt = surface_points(self, slice(start, stop))
            ulp = np.abs(points.astype(np.float32) - stored) / np.spacing(np.abs(stored))
            worst_ulp = max(worst_ulp, float(ulp.max()))
            inexact += int((ulp.max(1) > 0).sum())
            worst_normal = max(worst_normal, float(np.abs(rebuilt.astype(np.float32) - normal).max()))
        if worst_ulp > 1 or worst_normal > 0:
            raise ValueError(f'Metadata does not reproduce the PLY: {worst_ulp} ULP, normal {worst_normal}')
        return dict(directory=str(self.directory), gaussians=count, faces=int(self.face_offsets[-1]),
                    charts=int(self.chart_offsets[-1]), boundary_samples=int(np.count_nonzero(self.boundary)),
                    max_position_error_float32_ulp=worst_ulp, positions_not_bit_identical=inexact,
                    max_normal_error=worst_normal,
                    files={'face': 'int32 [N] global cache face of each Gaussian (row = stable id)',
                           'barycentric': 'float64 [N,2] first two barycentrics; third = 1 - sum',
                           'boundary': 'bool [N] chart-boundary sample',
                           'sample': 'int32 [N] index in the unfiltered sampling order',
                           'triangles': 'float64 [F,3,3] evaluated world triangles',
                           'uv': 'float32 [F,3,2] corner UV; zero where groups.json has_uv is false',
                           'chart': 'int32 [F] global chart id',
                           'loop_triangle': 'int32 [F] Blender loop-triangle index within the object',
                           'groups.json': 'object/material per group with [start, stop) face and chart ranges'})


def surface_points(metadata, rows):
    """World centres and geometric normals of the given Gaussian rows."""
    face = metadata.face[rows]
    partial = metadata.barycentric[rows]
    barycentric = np.column_stack((partial, 1 - partial.sum(1)))
    triangles = metadata.triangles[face]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return (np.einsum('ni,nij->nj', barycentric, triangles),
            cross / np.linalg.norm(cross, axis=1)[:, None])

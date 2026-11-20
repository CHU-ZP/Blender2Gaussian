"""Geometry, probe ownership and serialization for the retained initializer."""
from dataclasses import dataclass
from types import SimpleNamespace
import json
import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial.transform import Rotation


@dataclass
class Probes:
    face: np.ndarray
    bary: np.ndarray
    weight: np.ndarray
    boundary: np.ndarray

    def take(self, rows):
        return Probes(*(getattr(self, k)[rows] for k in ('face', 'bary', 'weight', 'boundary')))

    def points(self, surface):
        return np.einsum('ni,nij->nj', self.bary, surface.triangles[self.face])

    @classmethod
    def join(cls, *parts):
        return cls(*(np.concatenate([getattr(p, k) for p in parts])
                     for k in ('face', 'bary', 'weight', 'boundary')))


def model_arrays(rows):
    p = np.column_stack([rows[k] for k in 'xyz']).astype(float)
    scales = np.exp(np.column_stack([rows[f'scale_{k}'] for k in range(3)]).astype(float))
    q = np.column_stack([rows[f'rot_{k}'] for k in range(4)]).astype(float)
    rotations = Rotation.from_quat(q, scalar_first=True).as_matrix() if len(rows) else np.empty((0, 3, 3))
    opacity = 1/(1+np.exp(-rows['opacity'].astype(float)))
    if not all(np.isfinite(a).all() for a in (p, scales, rotations, opacity)) or np.any(scales <= 0) or np.any(opacity >= 1):
        raise ValueError('Invalid PLY parameters')
    return p, scales, rotations, opacity


def validate_attachment(rows, triangles, faces, barycentric, batch_size=262144):
    """Check float32 serialized centres against their source correspondence."""
    if len(rows) != len(faces) or np.shape(barycentric) != (len(rows), 2):
        raise ValueError('Model/metadata row mismatch')
    if np.any(faces < 0) or np.any(faces >= len(triangles)):
        raise ValueError('Invalid source-face ownership')
    for start in range(0, len(rows), batch_size):
        sl = slice(start, start+batch_size)
        b = barycentric[sl]
        b = np.c_[b, 1-b.sum(1)]
        if not np.isfinite(b).all() or np.any(b < -1e-12):
            raise ValueError('Invalid barycentric correspondence')
        expected = np.einsum('ni,nij->nj', b, triangles[faces[sl]])
        actual = np.column_stack([rows[k][sl] for k in 'xyz'])
        tolerance = 2*np.abs(np.spacing(actual))+1e-9
        if not np.isfinite(actual).all() or np.any(np.abs(expected-actual) > tolerance):
            raise ValueError('Input is not an attached initializer')


def frame(normals):
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent = np.cross(axis, normals)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    return np.stack((tangent, np.cross(normals, tangent), normals), axis=-1)


def prepare_geometry(raw_groups, old_groups, metadata, directory):
    """Restore positive-area slivers omitted by the legacy cache, with no resampling.

    Loop-triangle identity maps every old row to its unchanged source face.
    Added slivers receive separate conservative charts. All raw nondegenerate
    faces, including groups/charts with no initial Gaussian, remain targets.
    """
    if len(raw_groups) != len(old_groups):
        raise ValueError('Source object/material groups changed')
    targets, groups = [], []
    old_to_new = np.empty(len(metadata.triangles), np.int64)
    face_offset = chart_offset = restored = raw_count = degenerate = 0
    for raw, old in zip(raw_groups.values(), old_groups):
        desc = raw['description']
        if (raw['object_name'], desc['name']) != (old['object'], old['material']):
            raise ValueError('Source group identity changed')
        triangles = np.concatenate(raw.pop('triangles')); loop = np.concatenate(raw.pop('loop_triangles'))
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        area2 = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])
        valid = np.isfinite(triangles).all((1, 2)) & np.isfinite(area2) & (area2 > 0)
        raw_count += len(valid); degenerate += int((~valid).sum())
        uv = np.concatenate(raw.pop('uvs'))[valid] if raw['uvs'] else np.zeros((int(valid.sum()), 3, 2))
        triangles, loop = triangles[valid], loop[valid]
        a, b = old['faces']; order = np.argsort(loop)
        old_loop = metadata.loop_triangle[a:b]
        found = np.searchsorted(loop[order], old_loop)
        if np.any(found >= len(loop)) or not np.array_equal(loop[order[found]], old_loop):
            raise ValueError('Cached loop-triangle identities missing from source')
        local = order[found]
        np.testing.assert_array_equal(triangles[local], metadata.triangles[a:b])
        labels = np.full(len(loop), -1, np.int64)
        labels[local] = metadata.chart[a:b]-old['charts'][0]
        missing = labels < 0
        labels[missing] = np.arange(int(missing.sum()))+(old['charts'][1]-old['charts'][0])
        _, labels = np.unique(labels, return_inverse=True)
        charts = int(labels.max(initial=-1))+1
        old_to_new[a:b] = face_offset+local
        groups.append(dict(old, faces=[face_offset, face_offset+len(loop)],
                           charts=[chart_offset, chart_offset+charts], restored_sliver_faces=int(missing.sum())))
        targets.append(SimpleNamespace(triangles=triangles, uv=uv, chart=labels,
                                       loop=loop, image=desc['image'], colour=desc['color']))
        face_offset += len(loop); chart_offset += charts; restored += int(missing.sum())
    directory.mkdir()
    specifications = {'triangles':(np.float64,(face_offset,3,3)), 'uv':(np.float32,(face_offset,3,2)),
                      'chart':(np.int32,(face_offset,)), 'loop_triangle':(np.int32,(face_offset,))}
    arrays = {k:open_memmap(directory/f'{k}.npy','w+',dtype,shape) for k,(dtype,shape) in specifications.items()}
    for target, group in zip(targets, groups):
        a,b=group['faces']; arrays['triangles'][a:b]=target.triangles; arrays['uv'][a:b]=target.uv
        arrays['chart'][a:b]=target.chart+group['charts'][0]; arrays['loop_triangle'][a:b]=target.loop
    for a in arrays.values():a.flush()
    (directory/'groups.json').write_text(json.dumps(groups,indent=1,ensure_ascii=False)+'\n')
    return targets, groups, old_to_new, dict(raw_faces=raw_count, target_faces=face_offset,
        skipped_degenerate_faces=degenerate, restored_sliver_faces=restored, target_groups=len(groups)), arrays

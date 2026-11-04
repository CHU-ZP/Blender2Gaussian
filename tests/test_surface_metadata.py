"""Surface correspondences written alongside the Gaussian PLY."""
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement

from mesh_to_gs.surface import charts
from mesh_to_gs.surface_state import SurfaceMetadata, surface_points
from scripts.initialize_blend_surface import group_samples


def mixed_surface():
    """One subdivided chart, a single-face chart, a two-face singleton chart and a degenerate face."""
    xy = np.mgrid[0:5, 0:5].reshape(2, -1).T / 4
    grid = np.column_stack((xy, np.zeros(len(xy))))
    ids = []
    for i in range(4):
        for j in range(4):
            a = i*5+j
            ids.extend([[a, a+5, a+1], [a+5, a+6, a+1]])
    triangles = [grid[np.array(ids)]]
    triangles.append(np.array([[[2., 0, 0], [3, 0, 0], [2, 1, 0]]]))
    triangles.append(np.array([[[4., 0, 0], [4, 0, 0], [5, 1, 0]]]))  # degenerate, dropped
    triangles.append(np.array([[[0., 0, 1], [.1, 0, 1], [0, .1, 1]], [[.1, 0, 1], [.1, .1, 1], [0, .1, 1]]]))
    triangles = np.concatenate(triangles)
    surface = charts(triangles, np.arange(3*len(triangles)).reshape(-1, 3))
    sizes = np.bincount(surface.labels)
    counts = np.where(sizes == 32, 300, np.where(sizes == 1, 3, 1))
    return surface, counts


def samples(surface, counts, record):
    args = SimpleNamespace(seed=7, overlap=.8, boundary_sigmas=2., surface_metadata=record)
    return list(group_samples(surface, counts, args, 0))


def test_metadata_does_not_change_samples():
    surface, counts = mixed_surface()
    plain, recorded = samples(surface, counts, False), samples(surface, counts, True)
    assert len(plain) == len(recorded) == 3  # singleton, single-face, general paths
    for a, b in zip(plain, recorded):
        assert a[7] is None and b[7] is not None
        for x, y in zip(a[:7], b[:7]):
            np.testing.assert_array_equal(x, y)


def test_metadata_rebuilds_centres_normals_and_source_faces(tmp_path):
    surface, counts = mixed_surface()
    assert surface.skipped == 1 and surface.source.tolist() == [*range(33), 34, 35]
    batches = samples(surface, counts, True)
    groups = {0: dict(object_name='object', object_index=0,
                      description=dict(name='material', image=None, color=[.5, .5, .5]))}
    total = sum(len(b[0]) for b in batches)
    metadata = SurfaceMetadata(tmp_path/'surface_metadata', [surface], groups, total)
    metadata.write_group(0, surface, surface.source)
    done = 0
    for p, *_, faces in batches:
        metadata.write_samples(0, done, *faces, np.arange(done, done + len(p)))
        done += len(p)
    p = np.concatenate([b[0] for b in batches]).astype(np.float32)
    n = np.concatenate([b[1] for b in batches]).astype(np.float32)
    vertex = np.empty(len(p), [(k, 'f4') for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')])
    for i, k in enumerate('xyz'):
        vertex[k], vertex['n'+k] = p[:, i], n[:, i]
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(tmp_path/'model.ply')
    report = metadata.finish(tmp_path/'model.ply', total)
    assert report['max_position_error_float32_ulp'] <= 1 and report['max_normal_error'] == 0
    assert report['boundary_samples'] == sum(b[6] for b in batches) > 0
    np.testing.assert_array_equal(np.load(tmp_path/'surface_metadata/loop_triangle.npy'), surface.source)
    # Every centre lies on its recorded face and chart.
    points, _ = surface_points(metadata, slice(None))
    np.testing.assert_allclose(points, np.concatenate([b[0] for b in batches]), atol=1e-12)
    chart = np.load(tmp_path/'surface_metadata/chart.npy')[np.load(tmp_path/'surface_metadata/face.npy')]
    assert np.bincount(chart, minlength=len(counts)).tolist() == counts.tolist()


def test_filtered_rows_are_trimmed_and_traced_to_samples(tmp_path):
    surface, counts = mixed_surface()
    batches = samples(surface, counts, True)
    groups = {0: dict(object_name='object', object_index=0,
                      description=dict(name='material', image=None, color=[.5, .5, .5]))}
    total = sum(len(b[0]) for b in batches)
    metadata = SurfaceMetadata(tmp_path/'surface_metadata', [surface], groups, total)
    metadata.write_group(0, surface, surface.source)
    done = sampled = 0
    kept_points = []
    for p, *_, faces in batches:
        keep = np.arange(sampled, sampled + len(p)) % 3 != 0   # stand-in for removed hidden samples
        rows = np.arange(sampled, sampled + len(p))[keep]
        metadata.write_samples(0, done, *(f[keep] for f in faces), rows)
        kept_points.append(p[keep])
        done += len(rows)
        sampled += len(p)
    p = np.concatenate(kept_points).astype(np.float32)
    vertex = np.zeros(len(p), [(k, 'f4') for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')])
    for i, k in enumerate('xyz'):
        vertex[k] = p[:, i]
    vertex['nz'] = 1
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(tmp_path/'model.ply')
    report = metadata.finish(tmp_path/'model.ply', done)
    assert report['gaussians'] == done < total and report['max_position_error_float32_ulp'] <= 1
    sample = np.load(tmp_path/'surface_metadata/sample.npy')
    assert len(sample) == len(np.load(tmp_path/'surface_metadata/face.npy')) == done
    assert np.all(np.diff(sample) > 0) and not np.any(sample % 3 == 0)

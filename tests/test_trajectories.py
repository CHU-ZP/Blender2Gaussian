"""Camera motion checks cover interpolation, closed-loop framing and thin obstacles."""
import numpy as np
import pytest

from mesh_to_gs.trajectories import dense_path, make_cameras, orbit_path, path_clear, resample_path


def test_arclength_sampling_keeps_endpoints_and_ignores_duplicate_waypoints():
    points = [[0, 0, 1], [0, 0, 1], [2, 0, 1], [2, 1, 1]]
    expected = [[0, 0, 1], [1, 0, 1], [2, 0, 1], [2, 1, 1]]
    np.testing.assert_allclose(resample_path(points, 4), expected)
    eased = resample_path(points, 101, ease=True)
    assert np.linalg.norm(eased[1]-eased[0]) < np.linalg.norm(eased[51]-eased[50])
    with pytest.raises(ValueError, match='positive length'):
        resample_path([[1, 2, 3], [1, 2, 3]], 10)


def test_dense_checks_cover_a_long_segment_and_the_closing_edge():
    points = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
    dense = dense_path(points, .05, closed=True)
    assert (np.linalg.norm(np.diff(dense, axis=0), axis=1) <= .05+1e-12).all()
    assert (np.linalg.norm(dense-[.5, .5, 1], axis=1) < .05).any()


def test_complete_mesh_check_rejects_crossing_a_wall_between_safe_frames():
    from mesh_to_gs.visibility import VisibilityTester
    triangles = np.array([[[0, -2, 0], [0, 2, 0], [0, 2, 3]],
                          [[0, -2, 0], [0, 2, 3], [0, -2, 3]]], dtype=float)
    tester = VisibilityTester(triangles, [[-2, -2, 0], [2, 2, 3]], spacing=1.,
                              min_enclosure=0, device='cpu')
    clear, checks = path_clear(tester, triangles, [[-1, 0, 1.5], [1, 0, 1.5]])
    assert not clear and checks > 2
    assert path_clear(tester, triangles, [[-.6, -1, 1.5], [-.6, 1, 1.5]])[0]


@pytest.mark.parametrize('turns,height_variation', [(1, 0.), (2, 1.5)])
def test_orbit_cameras_fit_all_corners_and_form_a_continuous_closed_loop(turns, height_variation):
    bounds = [[-4, -3, 0], [4, 3, 2]]
    points, target, _ = orbit_path(bounds, 120, 42, 1280, 720,
                                    turns=turns, height_variation=height_variation)
    cameras = make_cameras(points, 1280, 720, 24, target=target)
    corners = np.array([[x, y, z, 1] for x in (-4, 4) for y in (-3, 3) for z in (0, 2)])
    for camera in cameras:
        world = np.asarray(camera['camera_to_world'])
        view = np.asarray(camera['view_matrix'])
        np.testing.assert_allclose(view@world, np.eye(4), atol=1e-10)
        local = (corners@view.T)[:, :3]
        pixel = local@np.asarray(camera['K']).T
        pixel = pixel[:, :2]/pixel[:, 2:]
        assert (pixel > [0, 0]).all() and (pixel < [1280, 720]).all()
    steps = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    if height_variation == 0:
        np.testing.assert_allclose(steps, steps[0], rtol=1e-10)
    else:
        assert steps[-1] <= steps.max()
        assert abs(points[-1, 2]-points[0, 2]) < .01
        assert np.ptp(points[:, 2]) == pytest.approx(height_variation)
        assert np.linalg.norm(points[0]-points[60]) == pytest.approx(height_variation)
    assert len({c['name'] for c in cameras}) == 120

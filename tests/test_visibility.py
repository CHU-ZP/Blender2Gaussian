"""Coincident hidden layers are dropped; visible or farther hidden surfaces are kept."""
import numpy as np
import pytest

pytest.importorskip('warp')
from mesh_to_gs.visibility import VisibilityTester
from scripts.initialize_blend_surface import coincident_hidden


def quad(x0, x1, y0, y1, z0, z1, axis):
    """Axis-aligned rectangle as two triangles; `axis` is the constant coordinate."""
    if axis == 2:
        corners = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]
    elif axis == 1:
        corners = [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]
    else:
        corners = [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]
    c = np.array(corners, float)
    return np.stack((c[[0, 1, 2]], c[[0, 2, 3]]))


def grid(x0, x1, y0, y1, z, step=.1):
    x, y = np.meshgrid(np.arange(x0 + step/2, x1, step), np.arange(y0 + step/2, y1, step))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))


@pytest.fixture(scope='module')
def scene():
    room = np.concatenate([quad(-2, 2, -2, 2, -.05, -.05, 2), quad(-2, 2, -2, 2, 3, 3, 2),
                           quad(-2, 2, -2, -2, -.05, 3, 1), quad(-2, 2, 2, 2, -.05, 3, 1),
                           quad(-2, -2, -2, 2, -.05, 3, 0), quad(2, 2, -2, 2, -.05, 3, 0)])
    slab = quad(-1, 1, -1, 1, .001, .001, 2)        # thin floor board
    sheet = quad(-1, 1.5, -1, 1, 0, 0, 2)           # 1 mm under it, partly uncovered
    panel = quad(-1, 1, 1.97, 1.97, 1, 2, 1)        # 3 cm in front of the wall
    triangles = np.concatenate([room, slab, sheet, panel])
    tester = VisibilityTester.for_scene(triangles, [[-2, -2, -.05], [2, 2, 3]], 'auto', spacing=.5, neighbours=16, distant=16)
    assert tester.mode == 'indoor'
    return tester


def drop(tester, points, normal):
    normals = np.tile(normal, (len(points), 1)).astype(float)
    scales = np.full((len(points), 3), 1e-3, np.float32)
    quats = np.tile([1, 0, 0, 0], (len(points), 1)).astype(np.float32)
    return coincident_hidden(tester, points, normals, scales, quats, .01)


def test_ray_helpers(scene):
    eye = np.array([[0., 0, 1.5]])
    assert np.isclose(scene.first_hit(eye, np.array([[0, 0, -1.]]))[0], 1.5 - .001, atol=1e-4)   # board top
    assert np.isclose(scene.clearance(eye)[0], 1.5, atol=.1)


def test_viewpoints_stay_in_free_space(scene):
    v = scene.viewpoints
    assert len(v) > 0 and v[:, 2].min() > .05 and np.abs(v[:, :2]).max() < 1.9


def test_layer_under_thin_board_is_dropped_only_where_covered(scene):
    sheet = grid(-1, 1.5, -1, 1, 0)
    dropped = drop(scene, sheet, [0, 0, -1])
    covered = np.abs(sheet[:, 0]) < .9
    uncovered = sheet[:, 0] > 1.1
    assert dropped[covered].all() and not dropped[uncovered].any()


def test_visible_and_distant_hidden_surfaces_are_kept(scene):
    assert not drop(scene, grid(-1, 1, -1, 1, .001), [0, 0, 1]).any()          # board top
    assert not drop(scene, grid(-1, 1, -1, 1, -.05), [0, 0, 1]).any()          # floor 5 cm below
    wall = np.column_stack((np.linspace(-.8, .8, 9), np.full(9, 2.), np.full(9, 1.5)))
    assert not drop(scene, wall, [0, -1, 0]).any()                             # 3 cm behind the panel


def test_open_scene_mode_finds_viewpoints_outside_the_bounds():
    # A box on a ground plate, no walls: only viewpoints around the scene can see it.
    box = np.concatenate([quad(-.5, .5, -.5, .5, 0, 0, 2), quad(-.5, .5, -.5, .5, .4, .4, 2),
                          quad(-.5, .5, -.5, -.5, 0, .4, 1), quad(-.5, .5, .5, .5, 0, .4, 1),
                          quad(-.5, -.5, -.5, .5, 0, .4, 0), quad(.5, .5, -.5, .5, 0, .4, 0)])
    plate = quad(-1, 1, -1, 1, -.003, -.003, 2)          # 3 mm under the box bottom
    # Bounds start at the plate: an open scene is seen from above its ground, not from below.
    bounds = [[-3, -3, -.003], [3, 3, 2.]]
    # Inside the closed box the grid finds enclosed points, but they are a small
    # share of a grid that extends around it (as for a furnished open scene).
    tester = VisibilityTester.for_scene(np.concatenate([box, plate]), bounds, 'auto', spacing=.25, open_margin=1.,
                                        neighbours=16, distant=16)
    assert tester.mode == 'open' and tester.report['margin'] == 1. and (tester.viewpoints[:, 2] > .5).any()
    assert tester.viewpoints[:, 2].min() > -.003           # never below the ground
    under_box = grid(-.4, .4, -.4, .4, -.003)
    assert drop(tester, under_box, [0, 0, -1]).all()            # plate hidden under the box
    top = grid(-.4, .4, -.4, .4, .4)
    assert not drop(tester, top, [0, 0, 1]).any()

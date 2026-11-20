"""Estimate the unattended training budget from evaluated Mesh triangles."""
import json
import math
from pathlib import Path
import subprocess
import sys


REFERENCE_TRIANGLES = 4_936_252
INITIALIZER = Path(__file__).resolve().parents[1] / 'configs' / 'area_sampling_initialization.json'
DEFAULT_SPACING = json.loads(INITIALIZER.read_text())['spacing']  # Blender world units.
AUTO_FIELDS = ('views', 'validation_views', 'steps')
QUALITY_DEFAULTS = dict(width=1280, height=720, samples=16, colour_passes=4, seed=20261102)


def triangle_budget(triangles):
    if triangles < 1:
        raise ValueError('No render-enabled mesh triangles found')
    # Sublinear growth limits sensitivity to differences in tessellation.
    scale = (triangles / REFERENCE_TRIANGLES) ** (1 / 3)
    views = min(256, max(24, round(196 * scale)))
    return dict(views=views,
                validation_views=max(4, math.ceil(12 * views / 196)),
                steps=round(7000 * views / 196))


def count_scene_triangles(path):
    import bpy
    from mesh_to_gs.blend import evaluated_mesh_objects

    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    graph = bpy.context.evaluated_depsgraph_get()
    count = 0
    for _, evaluated in evaluated_mesh_objects(graph):
        mesh = evaluated.data
        # A polygon with n corners yields n-2 triangles, including degenerates.
        count += len(mesh.loops) - 2 * len(mesh.polygons)
    return count


def inspect_triangles(scene):
    # Release Blender and its scene memory before starting the training stages.
    probe = subprocess.run([sys.executable, '-m', 'mesh_to_gs.scene_budget', str(scene)],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    if probe.returncode:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        raise ValueError(f'Cannot inspect {scene}: {detail[-1] if detail else probe.returncode}')
    for line in probe.stdout.splitlines():
        if line.startswith('TRIANGLES='):
            return int(line.removeprefix('TRIANGLES='))
    raise ValueError(f'Blender did not return a triangle count for {scene}')


def new_plan(scene, overrides):
    if 'spacing' in overrides and 'count' in overrides:
        raise ValueError('Choose either spacing or a fixed count')
    defaults = QUALITY_DEFAULTS.copy()
    if 'count' not in overrides:
        defaults['spacing'] = DEFAULT_SPACING
    if all(field in overrides for field in AUTO_FIELDS):
        return dict(method='explicit', triangles=None, budgets=defaults | overrides,
                    overrides=overrides)
    triangles = inspect_triangles(scene)
    return dict(method='triangle_training_budget_v2', triangles=triangles,
                budgets=defaults | triangle_budget(triangles) | overrides, overrides=overrides)


if __name__ == '__main__':
    print(f'TRIANGLES={count_scene_triangles(sys.argv[1])}', flush=True)

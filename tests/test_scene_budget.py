import json
import pytest

from mesh_to_gs import scene_budget
from scripts import run_scene


def test_resume_uses_saved_samples_after_template_default_changes(tmp_path, monkeypatch):
    scene = tmp_path/'scene.blend'
    scene.touch()
    config = json.loads(run_scene.CONFIG.read_text())
    assert config['reference_samples'] == 16
    template = tmp_path/'template.json'
    monkeypatch.setattr(run_scene, 'CONFIG', template)
    out = tmp_path/'out'
    out.mkdir()
    run_scene.write_json(template, dict(config, reference_samples=64))
    state = run_scene.prepare_run(scene, out, dict(views=6, validation_views=3, steps=28, samples=64))
    run_scene.write_json(out/'run.json', state)
    run_scene.write_json(template, config)
    assert run_scene.prepare_run(scene, out, {}) == state
    run_scene.write_json(template, dict(config, depth_weight=0.1))
    with pytest.raises(ValueError, match='different scene or budget'):
        run_scene.prepare_run(scene, out, {})


def test_estimate_scales_through_reference_and_clamps():
    baseline = scene_budget.triangle_budget(scene_budget.REFERENCE_TRIANGLES)
    assert baseline == dict(views=196, validation_views=12, steps=7000)
    small, medium, large = map(scene_budget.triangle_budget, (24, 300_000, 14_000_000))
    for name in baseline:
        assert small[name] < medium[name] < baseline[name] < large[name]
    assert scene_budget.triangle_budget(1) == small
    assert scene_budget.triangle_budget(10**9) == large
    with pytest.raises(ValueError, match='triangles'):
        scene_budget.triangle_budget(0)


def test_count_uses_evaluated_visible_mesh_and_triangulates_quads(tmp_path):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.object.modifiers.new('Two copies', 'ARRAY').count = 2
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.object.hide_render = True
    path = tmp_path/'input.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    assert scene_budget.inspect_triangles(path) == 24


def test_resume_pins_automatic_and_legacy_budgets(tmp_path, monkeypatch):
    source = tmp_path/'scene.blend'
    source.touch()
    out = tmp_path/'out'
    out.mkdir()
    monkeypatch.setattr(scene_budget, 'inspect_triangles', lambda _: 300_000)
    first = run_scene.prepare_run(source, out, dict(steps=123))
    assert first['budget_plan']['triangles'] == 300_000
    assert first['settings']['budgets']['steps'] == 123
    assert first['settings']['budgets']['samples'] == 16
    assert first['settings']['budgets']['spacing'] == .024
    assert 'count' not in first['settings']['budgets']
    run_scene.write_json(out/'run.json', first)
    monkeypatch.setattr(scene_budget, 'inspect_triangles', lambda _: pytest.fail('unexpected reinspection'))
    assert run_scene.prepare_run(source, out, {}) == first
    with pytest.raises(ValueError, match='different scene or budget'):
        run_scene.prepare_run(source, out, dict(steps=124))
    with pytest.raises(ValueError, match='different scene or budget'):
        run_scene.prepare_run(source, out, dict(spacing=.02))
    with pytest.raises(ValueError, match='different scene or budget'):
        run_scene.prepare_run(source, out, dict(count=500))
    saved = first['settings']['budgets']
    del saved['spacing']
    saved['count'] = 500
    del first['budget_plan']
    run_scene.write_json(out/'run.json', first)
    legacy = run_scene.prepare_run(source, out, {})
    assert legacy['budget_plan']['method'] == 'saved_fixed_budget'
    assert legacy['budget_plan']['budgets'] == first['settings']['budgets']
    with pytest.raises(ValueError, match='different scene or budget'):
        run_scene.prepare_run(source, out, dict(spacing=.024))

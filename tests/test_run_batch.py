import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_batch


def test_scene_paths_and_dry_run(tmp_path, monkeypatch, capsys):
    inputs = tmp_path/'input'
    files = [inputs/'a/scene.blend', inputs/'b/scene.blend']
    for path in files:
        path.parent.mkdir(parents=True)
        path.touch()
    manifest = inputs/'list.json'
    manifest.write_text(json.dumps(['a/scene.blend', 'b/scene.blend']))
    out = inputs/'batch'
    expected = run_batch.discover(inputs, out)
    assert expected == run_batch.discover(manifest, out)
    assert expected[0]['output'] != expected[1]['output']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,4')
    monkeypatch.setattr('mesh_to_gs.scene_budget.inspect_triangles',
                        lambda scene: 100_000 if scene == files[0] else 1_000_000)
    monkeypatch.setattr('sys.argv', ['run_batch.py', str(manifest), '--output-dir', str(out),
                                     '--gpus', '4', '--steps', '21', '--dry-run'])
    assert run_batch.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['gpus'] == ['4'] and plan['budgets']['steps'] == 21
    plans = [job['budget_plan'] for job in plan['scenes']]
    assert all(entry['budgets']['spacing'] == .024 and 'count' not in entry['budgets'] for entry in plans)
    assert plans[0]['budgets']['views'] < plans[1]['budgets']['views']
    assert [entry['budgets']['steps'] for entry in plans] == [21, 21]
    assert [job['scene'] for job in plan['scenes']] == [job['scene'] for job in expected]
    assert not out.exists()
    out.mkdir()
    (out/'archive.blend').touch()
    assert expected == run_batch.discover(inputs, out)
    with pytest.raises(ValueError, match='CUDA_VISIBLE_DEVICES'):
        run_batch.gpu_list('0')


def test_failed_scene_releases_gpu_and_success_uses_selected_scores(tmp_path, monkeypatch):
    inputs, out = tmp_path/'inputs', tmp_path/'out'
    inputs.mkdir()
    out.mkdir()
    for name in ('a', 'b', 'c'):
        (inputs/f'{name}.blend').touch()
    jobs = run_batch.discover(inputs, out)
    calls = []

    def launch(command, **kwargs):
        device = kwargs['env']['CUDA_VISIBLE_DEVICES']
        calls.append((Path(command[3]).stem, device))
        assert command[-2:] == ['--colour-passes', '2']
        if len(calls) == 1:
            return SimpleNamespace(poll=lambda: 1)
        directory = Path(command[command.index('--output-dir')+1])
        (directory/'colour').mkdir(parents=True)
        (directory/'init').mkdir()
        (directory/'init/initialization_report.json').write_text(json.dumps(dict(
            method='mesh_area_spacing_barycentric_texture', gaussians=101, model_sha256='input',
            surface_area_world_units_squared=.058)))
        (directory/'model.ply').touch()
        selected = 'base' if len(calls) == 2 else 'colour'
        (directory/'run.json').write_text(json.dumps(dict(status='complete', model_sha256='test',
                          model=f'{selected}/model.ply', decision=selected, budget_plan={'triangles':123})))
        (directory/'colour/decision.json').write_text(json.dumps(dict(
            base_validation=dict(mae=.2), candidate_validation=dict(mae=.1))))
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(run_batch.subprocess, 'Popen', launch)
    monkeypatch.setattr(run_batch.time, 'sleep', lambda _: None)
    assert run_batch.execute(jobs, ['2','4'], dict(colour_passes=2), out) == 1
    assert calls == [('a', '2'), ('b', '4'), ('c', '2')]
    rows = json.loads((out/'batch.json').read_text())['scenes']
    assert [row['status'] for row in rows] == ['failed', 'complete', 'complete']
    assert [row['validation']['mae'] for row in rows[1:]] == [.2, .1]
    assert [row['budget_plan']['triangles'] for row in rows[1:]] == [123, 123]
    assert [row['initialization']['gaussians'] for row in rows[1:]] == [101, 101]

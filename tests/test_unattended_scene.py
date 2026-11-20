import argparse
import json
from pathlib import Path

import pytest

from scripts import run_scene
from scripts.refit_scene_colours import accept_refit


def test_colour_acceptance_requires_local_and_full_view_improvement():
    base = dict(local_pixels=10, local_mae=.3, mae=.2, mse=.1)
    better = dict(local_pixels=10, local_mae=.25, mae=.19, mse=.099)
    assert accept_refit(base, better)
    for regression in (dict(local_pixels=9), dict(local_mae=.3), dict(mae=.21), dict(mse=.11)):
        assert not accept_refit(base, dict(better, **regression))
    assert not accept_refit(dict(base, local_pixels=0, local_mae=None),
                            dict(better, local_pixels=0, local_mae=None))


@pytest.mark.parametrize(('allocation', 'expected'), [
    ({}, ('--spacing', .024)), ({'spacing': .012}, ('--spacing', .012)),
    ({'count': 200}, ('--count', 200)),
])
def test_resumes_failed_stage_and_publishes_base_on_rejection(tmp_path, monkeypatch, allocation, expected):
    source = tmp_path/'example.blend'
    source.write_bytes(b'mesh source')
    out = tmp_path/'out'
    out.mkdir()
    args = argparse.Namespace(scene=source, output_dir=out, views=4, validation_views=2,
                              steps=14, width=40, height=32, samples=2, seed=17, colour_passes=2, **allocation)
    calls = []
    def worker(name, *arguments):
        calls.append(name)
        if name == 'build_area_sampled_model.py':
            assert arguments[-2:] == expected
            assert ('--spacing' in arguments) != ('--count' in arguments)
        if name == 'audit_model.py':
            return
        if name == 'prepare_scene_references.py':
            directory = arguments[1]
        elif name == 'refit_scene_colours.py':
            directory = arguments[1]
            directory.mkdir()
            if calls.count(name) == 1:
                raise RuntimeError('interrupted')
            run_scene.write_json(directory/'decision.json', dict(model='base/model.ply',
                                 reason='validation did not improve', model_sha256=run_scene.digest(out/'base/model.ply')))
            return
        else:
            directory = arguments[arguments.index('--output-dir')+1]
        Path(directory).mkdir()
        if name == 'refine_joint.py':
            (directory/'model.ply').write_bytes(b'base model')

    monkeypatch.setattr(run_scene, 'run_script', worker)
    with pytest.raises(RuntimeError, match='interrupted'):
        run_scene.run(args)
    assert not (out/'model.ply').exists()
    run_scene.run(args)
    assert calls.count('refine_joint.py') == 1
    assert calls.count('refit_scene_colours.py') == 2
    assert (out/'model.ply').resolve() == out/'base/model.ply'
    n = len(calls)
    run_scene.run(args)
    assert len(calls) == n
    assert json.loads((out/'run.json').read_text())['status'] == 'complete'
    args.seed += 1
    with pytest.raises(ValueError, match='different scene or budget'):
        run_scene.run(args)

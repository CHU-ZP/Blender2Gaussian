"""The explicit legacy build retains joint training and reproducible cameras."""
import json
from pathlib import Path

import pytest

from scripts import build_model
from scripts.refine_joint import parse_args


def test_joint_defaults_match_selected_configuration_and_removed_modes_are_rejected():
    quality = json.loads((build_model.ROOT/'configs/joint_quality.json').read_text())
    args = parse_args(['init', '--output-dir', 'out'])
    assert all(getattr(args, key) == value for key, value in quality['refinement'].items())
    removed_options = [('--free-normals',), ('--no-split',), ('--no-move',), ('--split-schedule', 'two-phase'),
                       ('--densification', 'split'), ('--densification', 'project'),
                       ('--densification', 'resample'), ('--densification', 'resample-reinit'),
                       ('--record-control', 'control'), ('--replay-control', 'control'),
                       ('--resample-candidates', '16'), ('--resample-radius', '1')]
    for options in removed_options:
        with pytest.raises(SystemExit):
            parse_args(['init', '--output-dir', 'out', *options])


def test_build_runs_covariance_before_joint_and_removes_successful_intermediates(tmp_path, monkeypatch):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index('--output-dir')+1])
        destination.mkdir(parents=True)
        if command[1].endswith('initialize_blend_surface.py'):
            (destination/'initialization_report.json').write_text('{}')
        elif command[1].endswith('apply_texture_covariance.py'):
            (destination/'covariance_report.json').write_text('{}')
        else:
            queries = Path(command[command.index('--query-cache')+1])
            queries.mkdir()
            (queries/'queries.json').write_text('{}')
            (destination/'model.ply').write_bytes(b'completed')
    monkeypatch.setattr(build_model.subprocess, 'run', run)
    cameras = tmp_path/'cameras.json'
    cameras.write_text('{}')
    args = build_model.parse_args(['scene.blend', '--output-dir', str(tmp_path/'final'),
                                  '--sampling', 'legacy',
                                  '--training-cameras', str(cameras)])
    build_model.build(args)
    assert [Path(c[1]).name for c in commands] == ['initialize_blend_surface.py', 'apply_texture_covariance.py', 'refine_joint.py']
    assert '--training-cameras' in commands[-1]
    assert (tmp_path/'final/model.ply').read_bytes() == b'completed'
    assert (tmp_path/'final/provenance/query_manifest.json').exists()
    assert not list(tmp_path.glob('.final.build-*'))
    with pytest.raises(SystemExit):
        build_model.build(args)


def test_failed_build_keeps_diagnostics(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('training failed')
    monkeypatch.setattr(build_model.subprocess, 'run', fail)
    args = build_model.parse_args(['scene.blend', '--output-dir', str(tmp_path/'final'), '--sampling', 'legacy'])
    with pytest.raises(RuntimeError, match='training failed'):
        build_model.build(args)
    assert list(tmp_path.glob('.final.build-*'))


def test_optimized_coverage_build_requires_final_certificate_and_preserves_raw_model(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(Path(command[1]).name)
        directory = Path(command[command.index('--output-dir') + 1])
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'model.ply').write_bytes(b'raw' if calls[-1] == 'refine_joint.py' else b'certified')
        if calls[-1] == 'repair_coverage.py':
            (directory/'coverage_report.json').write_text(json.dumps(dict(status='incomplete', quality_protocol={})))
    monkeypatch.setattr(build_model.subprocess, 'run', run)
    output = tmp_path/'final'
    args = build_model.parse_args(['scene.blend', '--output-dir', str(output), '--sampling', 'coverage', '--optimize'])
    with pytest.raises(RuntimeError, match='final recertification'):
        build_model.build(args)
    assert calls == ['initialize_coverage_surface.py', 'refine_joint.py', 'repair_coverage.py']
    assert (output/'intermediates/optimized/model.ply').read_bytes() == b'raw'
    assert not (output/'pipeline.json').exists()

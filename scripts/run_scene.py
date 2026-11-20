"""Build a surface-bound SH0 model and validate automatic colour refinement."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid

from mesh_to_gs.mesh_supervision import digest
from mesh_to_gs.scene_budget import AUTO_FIELDS, DEFAULT_SPACING, INITIALIZER, QUALITY_DEFAULTS, new_plan

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/pbr_v10_area_surface_bound.json'


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def run_script(name, *args):
    subprocess.run([sys.executable, '-u', str(ROOT/'scripts'/name), *map(str, args)],
                   check=True, cwd=ROOT)


def budget_overrides(args):
    fields = ('spacing', 'count', *AUTO_FIELDS, *QUALITY_DEFAULTS)
    return {name: value for name, value in vars(args).items() if name in fields and value is not None}


def prepare_run(scene, out, overrides):
    manifest = out/'run.json'
    settings = dict(version=1, scene=str(scene), source_sha256=digest(scene),
                    init_config_sha256=digest(INITIALIZER), train_config_sha256=digest(CONFIG),
                    lighting_sha256=digest(ROOT/'configs/material_lighting_v10.json'))
    if manifest.exists():
        state = json.loads(manifest.read_text())
        saved = state['settings']['budgets']
        saved_hash = state['settings']['train_config_sha256']
        if settings['train_config_sha256'] != saved_hash and 'samples' in saved:
            # Old runs pinned the 64-sample template and passed their saved budget to both stages.
            previous = json.loads(CONFIG.read_text())
            previous['reference_samples'] = 64
            encoded = (json.dumps(previous, indent=2)+'\n').encode()
            if hashlib.sha256(encoded).hexdigest() == saved_hash:
                settings['train_config_sha256'] = saved_hash
        if dict(settings, budgets=saved | overrides) != state['settings']:
            raise ValueError('This output belongs to a different scene or budget. Choose a new directory.')
        state.setdefault('budget_plan', dict(method='saved_fixed_budget', triangles=None, budgets=saved))
        return state
    if out.exists() and any(p.name != '.run.lock' for p in out.iterdir()):
        raise ValueError('Use a new output or resume a directory created by run_scene.py')
    plan = new_plan(scene, overrides)
    return dict(settings=dict(settings, budgets=plan['budgets']), completed={}, budget_plan=plan)


def run(args):
    out = args.output_dir
    manifest = out/'run.json'
    state = prepare_run(args.scene, out, budget_overrides(args))
    write_json(manifest, state)
    print('Budget', json.dumps(state['budget_plan']), flush=True)
    args = argparse.Namespace(scene=args.scene, **state['settings']['budgets'])

    def stage(name, action):
        directory = out/name
        if name in state['completed']:
            if not directory.is_dir():
                raise FileNotFoundError(f'Missing completed stage: {directory}')
            return
        if directory.exists():
            directory.rename(out/f'{name}.interrupted-{uuid.uuid4().hex[:10]}')
        print(f'Starting {name}', flush=True)
        tick = time.monotonic()
        action(directory)
        state['completed'][name] = dict(seconds=time.monotonic()-tick)
        write_json(manifest, state)

    sampling_args = (('--count', args.count) if hasattr(args, 'count') else ('--spacing', args.spacing))
    stage('init', lambda d: run_script('build_area_sampled_model.py', args.scene,
                                       '--output-dir', d, *sampling_args))
    stage('references', lambda d: run_script('prepare_scene_references.py', out/'init', d,
          '--views', args.views, '--validation-views', args.validation_views, '--seed', args.seed,
          '--width', args.width, '--height', args.height, '--samples', args.samples))
    stage('material', lambda d: run_script('prepare_material_geometry.py', '--model-dir', out/'init',
                                           '--references', out/'references', '--output-dir', d))

    def train(directory):
        config = json.loads(CONFIG.read_text())
        config.update(model_dir=str(out/'init'), reference_cache=str(out/'references'),
                      material_semantics=str(out/'material'), views=args.views, steps=args.steps,
                      width=args.width, height=args.height, reference_samples=args.samples, seed=args.seed,
                      geometry_freeze_steps=args.steps//7, split_start=max(1, round(1500*args.steps/7000)),
                      split_stop=max(1, round(6500*args.steps/7000)), split_every=max(1, round(args.steps/14)),
                      checkpoint_every=0)
        write_json(out/'training_config.json', config)
        run_script('refine_joint.py', '--config', out/'training_config.json', '--output-dir', directory)
        run_script('audit_model.py', directory)

    stage('base', train)
    stage('colour', lambda d: run_script('refit_scene_colours.py', out, d, '--passes', args.colour_passes))
    decision = json.loads((out/'colour/decision.json').read_text())
    selected = out/decision['model']
    if digest(selected) != decision['model_sha256']:
        raise ValueError('Selected model failed its SHA256 check')
    if not (out/'model.ply').exists():
        link = out/'.model-link'
        link.unlink(missing_ok=True)
        link.symlink_to(selected.relative_to(out))
        link.replace(out/'model.ply')
    if (out/'model.ply').resolve() != selected.resolve():
        raise ValueError('The output model link points to a different file')
    state.update(status='complete', model=decision['model'], model_sha256=decision['model_sha256'],
                 decision=decision['reason'])
    write_json(manifest, state)
    print(f'{out}/model.ply ({decision["reason"]})', flush=True)


def add_budget_arguments(p):
    allocation = p.add_mutually_exclusive_group()
    allocation.add_argument('--spacing', type=float,
                            help=f'Target area spacing in Blender world units (new runs: {DEFAULT_SPACING})')
    allocation.add_argument('--count', type=int, help='Use an explicit point count for fixed-count comparisons')
    for name in AUTO_FIELDS:
        p.add_argument('--'+name.replace('_', '-'), type=int, help='Override the per-scene estimate')
    for name, default in QUALITY_DEFAULTS.items():
        p.add_argument('--'+name.replace('_', '-'), type=int, help=f'Default for a new run: {default}')


def validate_budgets(p, args):
    for name, value in budget_overrides(args).items():
        if name == 'spacing':
            if not math.isfinite(value) or value <= 0:
                p.error('--spacing must be finite and positive')
        elif value < (0 if name == 'seed' else 1):
            p.error('Budgets must be positive and seed non-negative')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true', help='Print the resolved budget without training or writing files')
    add_budget_arguments(p)
    args = p.parse_args()
    args.scene = args.scene.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    validate_budgets(p, args)
    if args.dry_run:
        print(json.dumps(prepare_run(args.scene, args.output_dir, budget_overrides(args))['budget_plan'], indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.exit(1, 'A run in this directory is already active\n')
        run(args)


if __name__ == '__main__':
    main()

"""Run independent Blender scenes on one or more GPUs on a single host."""
import argparse
from collections import deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__:
    from .run_scene import add_budget_arguments, budget_overrides, prepare_run, validate_budgets, write_json
else:
    from run_scene import add_budget_arguments, budget_overrides, prepare_run, validate_budgets, write_json

RUNNER = Path(__file__).with_name('run_scene.py').resolve()


def discover(source, output):
    if source.is_dir():
        if source == output:
            raise ValueError('Use different input and output directories')
        files = []
        for directory, dirs, names in source.walk():
            dirs[:] = [name for name in dirs if directory/name != output]
            files.extend(directory/name for name in names if name.lower().endswith('.blend'))
        files.sort()
    else:
        entries = json.loads(source.read_text())
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError('Expected a JSON array of scene paths')
        files = [source.parent/e for e in entries]
    scenes = [p.resolve(strict=True) for p in files]
    if not scenes or any(not p.is_file() or p.suffix.lower() != '.blend' for p in scenes):
        raise ValueError('Expected one or more Blender scene files')
    if len(set(scenes)) != len(scenes):
        raise ValueError('The input lists a scene more than once')
    jobs = []
    for scene in scenes:
        suffix = hashlib.sha256(os.fsencode(scene)).hexdigest()[:16]
        key = f'{scene.stem}-{suffix}'
        jobs.append(dict(scene=str(scene), output=str(output/'scenes'/key),
                         log=str(output/'logs'/f'{key}.log'), status='pending'))
    return jobs


def gpu_list(value):
    devices = [device.strip() for device in value.split(',')]
    if any(d in ('', '-1') for d in devices) or len(set(devices)) != len(devices):
        raise ValueError('Expected distinct visible GPU IDs, e.g. 0,1')
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        allowed = {d.strip() for d in os.environ['CUDA_VISIBLE_DEVICES'].split(',')}
        if not set(devices) <= allowed:
            raise ValueError('GPU IDs must be from CUDA_VISIBLE_DEVICES')
    return devices


def finish(job, code):
    job['exit_code'] = code
    if code:
        job['status'] = 'failed'
        return
    out = Path(job['output'])
    state = json.loads((out/'run.json').read_text())
    if state.get('status') != 'complete' or not (out/'model.ply').is_file():
        raise ValueError('Missing completed scene model')
    decision = json.loads((out/'colour/decision.json').read_text())
    init = json.loads((out/'init/initialization_report.json').read_text())
    scores = (decision['base_validation'] if state['model'] == 'base/model.ply'
              else decision['candidate_validation'])
    job.update(status='complete', model=str(out/'model.ply'),
               model_sha256=state['model_sha256'], decision=state['decision'], validation=scores,
               budget_plan=state['budget_plan'],
               initialization={k: init[k] for k in ('method', 'gaussians', 'model_sha256',
                                                     'surface_area_world_units_squared')})


def execute(jobs, devices, budgets, output):
    pending = deque(jobs)
    active = {}
    (output/'logs').mkdir(exist_ok=True)
    cli = [item for name, value in budgets.items()
           for item in ('--'+name.replace('_', '-'), str(value))]
    summary = dict(gpus=devices, budgets=budgets, scenes=jobs)
    summary_path = output/'batch.json'
    try:
        while pending or active:
            changed = False
            for device, (proc, job) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                del active[device]
                try:
                    finish(job, code)
                except (OSError, ValueError, KeyError) as exc:
                    job.update(status='failed', error=str(exc))
                print(f"{job['status']} {job['scene']}", flush=True)
                changed = True
            for device in devices:
                if device in active or not pending:
                    continue
                job = pending.popleft()
                job.update(status='running', gpu=device)
                command = [sys.executable, '-u', str(RUNNER), job['scene'],
                           '--output-dir', job['output'], *cli]
                try:
                    with Path(job['log']).open('a') as log:
                        print(f'\n{time.ctime()} GPU {device}', file=log, flush=True)
                        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                                                start_new_session=True)
                    active[device] = (proc, job)
                    print(f"GPU {device} {job['scene']}", flush=True)
                except OSError as exc:
                    job.update(status='failed', error=str(exc))
                changed = True
            if changed:
                write_json(summary_path, summary)
            if active:
                time.sleep(.5)
    finally:
        # Each child starts its renderer and trainer in the same process group.
        for proc, job in active.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            job['status'] = 'interrupted'
        for proc, _ in active.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        write_json(summary_path, summary)
    return int(any(job['status'] != 'complete' for job in jobs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path, help='Directory tree or JSON array of .blend paths')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                   help='Comma-separated device IDs, one active scene per device')
    p.add_argument('--dry-run', action='store_true', help='Print jobs without launching or writing files')
    add_budget_arguments(p)
    args = p.parse_args()
    validate_budgets(p, args)
    output = args.output_dir.resolve()
    try:
        jobs = discover(args.input.resolve(strict=True), output)
        devices = gpu_list(args.gpus)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    budgets = budget_overrides(args)
    if args.dry_run:
        for job in jobs:
            try:
                state = prepare_run(Path(job['scene']), Path(job['output']), budgets)
                job['budget_plan'] = state['budget_plan']
            except (OSError, ValueError) as exc:
                job.update(status='failed', error=str(exc))
        print(json.dumps(dict(gpus=devices, budgets=budgets, scenes=jobs), indent=2))
        return int(any(job['status'] == 'failed' for job in jobs))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.exit(1, 'Another batch is using this output directory\n')
        def stop(signum, frame):
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM, stop)
        return execute(jobs, devices, budgets, output)


if __name__ == '__main__':
    sys.exit(main())

"""Build a finite-probe model, or select continuous certification or legacy training."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-cameras', type=Path, help='Override automatic training poses')
    parser.add_argument('--reference-cache', type=Path, help='Reuse training cameras and Mesh EXRs')
    parser.add_argument('--sampling', choices=('legacy', 'coverage', 'sampled'), default='sampled',
                        help='sampled: finite probes; coverage: continuous certification; legacy: retained training')
    parser.add_argument('--optimize', action='store_true', help='Optimize coverage seeds, then repair and recertify the final model')
    parser.add_argument('--init-args', default='', help='Quoted extra arguments for the selected initializer')
    parser.add_argument('--refine-args', default='', help='Quoted extra arguments for refine_joint.py')
    args = parser.parse_args(argv)
    args.init, args.refine = map(shlex.split, (args.init_args, args.refine_args))
    if args.training_cameras and args.reference_cache:
        parser.error('Choose training cameras or a reference cache')
    if args.sampling == 'sampled' and (args.optimize or args.refine or args.training_cameras or args.reference_cache):
        parser.error('sampled currently supports initialization only; choose --sampling coverage or legacy for optimization')
    return args


def build(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit('Use a new output directory; existing models are never overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.sampling == 'sampled':
        subprocess.run([sys.executable, str(ROOT/'scripts/initialize_sampled_surface.py'),
                        str(args.input.resolve()), '--output-dir', str(output), *args.init], check=True, cwd=ROOT)
        report = json.loads((output/'sampled_coverage_report.json').read_text())
        if report['status'] != 'passed':
            raise RuntimeError('Finite-probe validation failed')
        (output/'pipeline.json').write_text(json.dumps(dict(
            input=str(args.input.resolve()), method='sampled_surface_coverage', model='model.ply',
            continuous_certificate=False, sampled_coverage_report='sampled_coverage_report.json',
            configuration=report['configuration']), indent=2) + '\n')
        print('Model passed finite-probe validation:', output/'model.ply', flush=True)
        return
    if args.sampling == 'coverage':
        optimize = args.optimize or bool(args.refine or args.training_cameras or args.reference_cache)
        if optimize:
            output.mkdir()
            intermediate = output/'intermediates'
            initial, trained = intermediate/'initializer', intermediate/'optimized'
            def run(script, *rest):
                subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str, rest)], check=True, cwd=ROOT)
            run('initialize_coverage_surface.py', args.input.resolve(), '--output-dir', initial, *args.init)
            extra = []
            if args.reference_cache:
                extra += ['--reference-cache', args.reference_cache.resolve()]
            elif args.training_cameras:
                extra += ['--training-cameras', args.training_cameras.resolve()]
            run('refine_joint.py', initial, '--output-dir', trained, '--uncapped-splits',
                '--retain-all-surfaces', *extra, *args.refine)
            run('repair_coverage.py', initial, trained, '--output-dir', output, '--resume')
            report = json.loads((output/'coverage_report.json').read_text())
            if report['status'] != 'complete':
                raise RuntimeError('Optimized coverage build failed final recertification')
            (output/'pipeline.json').write_text(json.dumps(dict(
                input=str(args.input.resolve()), method='uncapped_coverage_joint_optimization_repair',
                model='model.ply', coverage_report='coverage_report.json', repair_report='repair_report.json',
                raw_optimized_model='intermediates/optimized/model.ply', configuration=report['quality_protocol']), indent=2)+'\n')
            print('Certified optimized model:', output/'model.ply', flush=True)
            return
        subprocess.run([sys.executable, str(ROOT/'scripts/initialize_coverage_surface.py'),
                        str(args.input.resolve()), '--output-dir', str(output), *args.init], check=True, cwd=ROOT)
        report = json.loads((output/'coverage_report.json').read_text())
        if report['status'] != 'complete':
            raise RuntimeError('Coverage build is incomplete')
        (output/'pipeline.json').write_text(json.dumps(dict(
            input=str(args.input.resolve()), method='uncapped_coverage', model='model.ply',
            coverage_report='coverage_report.json', configuration=report['quality_protocol']), indent=2)+'\n')
        print('Certified model:', output/'model.ply', flush=True)
        return
    quality = json.loads((ROOT/'configs/joint_quality.json').read_text())
    work = Path(tempfile.mkdtemp(prefix=f'.{output.name}.build-', dir=output.parent))
    def run(script, *rest):
        subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str, rest)], check=True, cwd=ROOT)
    try:
        initial, covariance, queries = (work/name for name in ('init', 'covariance', 'mesh_queries'))
        run('initialize_blend_surface.py', args.input.resolve(), '--output-dir', initial,
            '--surface-metadata', *args.init)
        run('apply_texture_covariance.py', initial, '--output-dir', covariance,
            '--strength', quality['texture_covariance']['strength'],
            '--max-aspect', quality['texture_covariance']['max_aspect'])
        extra = []
        if args.reference_cache:
            extra += ['--reference-cache', args.reference_cache.resolve()]
        else:
            cameras = args.training_cameras or ROOT/f'configs/{args.input.stem}_cameras_train.json'
            if cameras.exists():
                extra += ['--training-cameras', cameras.resolve()]
            elif args.training_cameras:
                raise FileNotFoundError(args.training_cameras)
        run('refine_joint.py', covariance, '--output-dir', output, '--query-cache', queries,
            *extra, *args.refine)
        provenance = output/'provenance'
        provenance.mkdir(exist_ok=True)
        for source, target in ((initial/'initialization_report.json', 'initializer.json'),
                               (covariance/'covariance_report.json', 'texture_covariance.json'),
                               (queries/'queries.json', 'query_manifest.json')):
            shutil.copyfile(source, provenance/target)
        (output/'pipeline.json').write_text(json.dumps(dict(
            input=str(args.input.resolve()), method='views72_priority1500',
            model='model.ply', configuration=quality, intermediates='Regenerated from source; removed after success'),
            indent=2)+'\n')
    except BaseException:
        print(f'Build did not finish; diagnostic inputs remain in {work}', file=sys.stderr)
        raise
    shutil.rmtree(work)
    print('Final model:', output/'model.ply', flush=True)


def main():
    build(parse_args())


if __name__ == '__main__':
    main()

"""Recheck a coverage model's continuous spatial support after any mutation.

This checks coverage only. Changed colours/covariance also require the source
material and boundary quality checks and matched rendering; an old overall
quality certificate never survives a changed model hash.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from mesh_to_gs.coverage import CoverageConfig, certify_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    previous = json.loads((args.directory / 'coverage_report.json').read_text())
    protocol = previous['quality_protocol']
    config = CoverageConfig(**{k: protocol[k] for k in asdict(CoverageConfig())})
    report = certify_model(args.directory / 'model.ply', args.directory / 'surface', config)
    report['previous_model_sha256'] = previous['model_sha256']
    report['model_unchanged'] = report['model_sha256'] == previous['model_sha256']
    report['target_unchanged'] = report['target_sha256'] == previous.get('target_sha256')
    report['overall_quality_certificate_valid'] = (report['model_unchanged'] and report['target_unchanged']
                                                   and previous['status'] == 'complete')
    destination = args.output or args.directory / 'coverage_recheck.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

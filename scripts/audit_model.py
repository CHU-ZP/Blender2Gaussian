"""Validate a retained joint model, its split lineage and PLY parameters."""
import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from mesh_to_gs.mesh_supervision import digest


def audit_model(directory):
    directory = Path(directory)
    report_path = directory/'refinement_report.json'
    if not report_path.exists():
        report_path = directory/'provenance/training_report.json'
    report = json.loads(report_path.read_text())
    path = directory/'model.ply'
    if digest(path) != report['model_sha256']:
        raise ValueError('Model SHA256 differs from the completed training report')
    rows = PlyData.read(path, mmap='r')['vertex'].data
    lineage = np.load(directory/'lineage.npy', mmap_mode='r')
    expected = report['initial_count']+sum(e['parents'] for e in report['splits'])
    if len(rows) != expected or len(lineage) != expected or report['gaussians'] != expected:
        raise ValueError('Model count does not match split history and lineage')
    maximum_thickness_error = maximum_quaternion_error = maximum_normal_error = 0.
    for start in range(0, len(rows), 262144):
        part = rows[start:start+262144]
        ancestors = lineage[start:start+len(part)]
        if (ancestors < 0).any() or (ancestors >= report['initial_count']).any():
            raise ValueError('Invalid initializer lineage')
        if any(not np.isfinite(part[k]).all() for k in part.dtype.names):
            raise ValueError('Nonfinite PLY parameter')
        q = np.column_stack([part[f'rot_{i}'] for i in range(4)]).astype(np.float64)
        maximum_quaternion_error = max(maximum_quaternion_error, float(np.abs(np.linalg.norm(q, axis=1)-1).max()))
        q /= np.linalg.norm(q, axis=1)[:, None]
        w, x, y, z = q.T
        normal = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
        stored = np.column_stack([part[k] for k in ('nx', 'ny', 'nz')])
        maximum_normal_error = max(maximum_normal_error, float(np.linalg.norm(normal-stored, axis=1).max()))
        scales = np.exp(np.column_stack([part[f'scale_{i}'] for i in range(3)]).astype(np.float64))
        if not np.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError('Invalid Gaussian scales')
        maximum_thickness_error = max(maximum_thickness_error, float(np.abs(scales[:, 2]/(.02*scales[:, :2].min(-1))-1).max()))
        rgb = np.column_stack([part[f'f_dc_{i}'] for i in range(3)])*.28209479177387814+.5
        if rgb.min() < -1e-6 or rgb.max() > 1+1e-6:
            raise ValueError('RGB outside [0,1]')
    if maximum_quaternion_error > 1e-5 or maximum_thickness_error > 1e-4 or maximum_normal_error > 1e-5:
        raise ValueError('Quaternion, normal or surface thickness convention violated')
    result = dict(status='pass', gaussians=len(rows), model_sha256=report['model_sha256'],
                  lineage_valid=True, all_rows_finite=True, maximum_quaternion_error=maximum_quaternion_error,
                  maximum_normal_error=maximum_normal_error, maximum_relative_thickness_error=maximum_thickness_error)
    (directory/'model_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_model(args.model_dir)), flush=True)


if __name__ == '__main__':
    main()

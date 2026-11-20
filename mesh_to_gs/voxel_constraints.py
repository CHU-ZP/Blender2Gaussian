"""Immutable voxel ownership and conservative float32 position bounds."""
import json
import numpy as np
import torch

from .mesh_supervision import digest


def inward_float32_bounds(lower, upper):
    """Round each bound toward the interior, never enlarge the original box."""
    lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
    if lower.shape != upper.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('Expected matching finite voxel bounds')
    lo, hi = lower.astype(np.float32), upper.astype(np.float32)
    lo = np.where(lo.astype(np.float64) < lower, np.nextafter(lo, np.float32(np.inf)), lo)
    hi = np.where(hi.astype(np.float64) > upper, np.nextafter(hi, np.float32(-np.inf)), hi)
    if np.any(lo > hi):
        raise ValueError('Voxel contains no representable float32 position')
    return lo, hi


def load_voxel_bounds(model_dir, report, origin, positions, chunk=262144):
    """Use saved leaf ownership, not floor(position/width) after PLY rounding."""
    if report.get('method') != 'octree-voxel-surface':
        raise ValueError('--constrain-to-voxels requires an octree initializer')
    info = report['octree']
    width = float(info['voxel_size'])
    per_cell = report['samples_per_voxel']
    if not np.isfinite(width) or width <= 0 or not isinstance(per_cell, int) or per_cell < 1:
        raise ValueError('Invalid voxel metadata')
    path = model_dir/'leaf_cells.npy'
    cells = np.load(path, mmap_mode='r')
    if (cells.ndim != 2 or cells.shape[1] != 3 or cells.dtype.kind not in 'iu'
            or len(cells)*per_cell != len(positions) or len(positions) != report['gaussians']):
        raise ValueError('Voxel ownership does not match initializer rows')
    if cells.min() < 0 or cells.max() >= 2**info['depth']:
        raise ValueError('Leaf coordinates outside the octree root')
    root = np.asarray(info['root_min'], np.float64)
    lower, upper = np.empty(positions.shape, np.float32), np.empty(positions.shape, np.float32)
    maximum_input_error = 0.
    for start in range(0, len(positions), chunk):
        stop = min(len(positions), start+chunk)
        ids = np.arange(start, stop)//per_cell
        world_low = root+cells[ids]*width
        local_low, local_high = world_low-origin, world_low+width-origin
        point = positions[start:stop]
        error = np.maximum(local_low-point, point-local_high)
        tolerance = 2*np.abs(np.spacing((point+origin).astype(np.float32)))+1e-9
        if np.any(error > tolerance):
            raise ValueError('Initializer positions do not belong to the saved voxel cells')
        maximum_input_error = max(maximum_input_error, float(error.max()))
        lower[start:stop], upper[start:stop] = inward_float32_bounds(local_low, local_high)
    record = dict(mode='hard_original_voxel', voxel_size=width, root_min=root.tolist(),
        depth=info['depth'], leaves=len(cells), samples_per_initial_voxel=per_cell,
        initial_count=len(positions), source_leaf_sha256=digest(path),
        cells_file='voxel_cells.npy', row_ids_file='voxel_ids.npy',
        initial_ply_maximum_boundary_violation_world=maximum_input_error,
        ownership='Children inherit their original ancestor voxel; never recompute from current position',
        update='Project centres after every Adam update and immediately after each split',
        bounds='Float64 grid bounds rounded inward in the centred float32 training frame',
        export='Round world coordinates inward at voxel faces to keep float32 PLY centres inside')
    return lower, upper, record


@torch.no_grad()
def audit_voxel_bounds(model, chunk=262144):
    if model.voxel_lower is None:
        return None
    outside = 0
    maximum = 0.
    for start in range(0, len(model.position), chunk):
        sl = slice(start, start+chunk)
        p = model.position[sl]
        violation = torch.maximum(model.voxel_lower[sl]-p, p-model.voxel_upper[sl]).clamp_min(0)
        outside += int((violation > 0).any(-1).sum())
        maximum = max(maximum, float(violation.max()))
    if outside:
        raise ValueError(f'{outside} trained centres escaped their voxel')
    return dict(rows=len(model.position), outside_rows=outside, maximum_violation_world=maximum)


@torch.no_grad()
def voxel_export_positions(model, xyz, origin, chunk=262144):
    """Avoid pushing a boundary centre out of its voxel when writing world PLY."""
    rounded = xyz.astype(np.float32)
    corrected, maximum = 0, 0.
    for start in range(0, len(xyz), chunk):
        sl = slice(start, start+chunk)
        lo = model.voxel_lower[sl].cpu().numpy().astype(np.float64)+origin
        hi = model.voxel_upper[sl].cpu().numpy().astype(np.float64)+origin
        lo, hi = inward_float32_bounds(lo, hi)
        before = rounded[sl]
        after = np.maximum(lo, np.minimum(hi, before))
        delta = np.abs(after.astype(np.float64)-before.astype(np.float64))
        corrected += int(np.any(delta > 0, axis=1).sum())
        maximum = max(maximum, float(delta.max()))
        rounded[sl] = after
    return rounded, dict(rounding_corrected_rows=corrected, maximum_rounding_correction_world=maximum)


def audit_exported_voxels(directory, rows, lineage, expected, chunk=262144):
    """Check every saved centre in the original float64 world grid, strictly."""
    record = json.loads((directory/'voxel_constraints.json').read_text())
    if record != expected:
        raise ValueError('Voxel manifest differs from the completed training report')
    path = directory/record['cells_file']
    if digest(path) != record['source_leaf_sha256']:
        raise ValueError('Saved voxel grid changed')
    cells = np.load(path, mmap_mode='r')
    ids = np.load(directory/record['row_ids_file'], mmap_mode='r')
    if len(ids) != len(rows) or not np.array_equal(ids, lineage//record['samples_per_initial_voxel']):
        raise ValueError('Split children did not inherit original voxel ownership')
    if ids.min() < 0 or ids.max() >= len(cells):
        raise ValueError('Invalid saved voxel id')
    root = np.asarray(record['root_min'], np.float64)
    outside, maximum = 0, 0.
    for start in range(0, len(rows), chunk):
        sl = slice(start, start+chunk)
        part = rows[sl]
        positions = np.c_[part['x'],part['y'],part['z']].astype(np.float64)
        lower = root+cells[ids[sl]]*record['voxel_size']
        upper = lower+record['voxel_size']
        violation = np.maximum(lower-positions, positions-upper)
        outside += int(np.any(violation > 0, axis=1).sum())
        maximum = max(maximum, float(violation.max()))
    if outside:
        raise ValueError(f'{outside} PLY centres escaped the original voxel')
    return dict(rows=len(rows), outside_rows=outside, maximum_violation_world=maximum,
                inherited_ownership_valid=True, strict_world_bounds=True)

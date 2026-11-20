"""Simple area-proportional Mesh sampling with per-corner UV lookup."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial.transform import Rotation

from .mesh_supervision import digest
from .mesh_geometry import ColourTextures, extract_geometry
from .repair_geometry import frame
from .scene_budget import DEFAULT_SPACING
from .surface import texture_colors


ATTRIBUTES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
              'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
DTYPE = np.dtype([(key, '<f4') for key in ATTRIBUTES])
SH_C0 = 0.28209479177387814


def allocate_area_counts(areas: np.ndarray, total: int | None = None, *, spacing=None) -> np.ndarray:
    """Round cumulative world area at the requested density or a fixed count."""
    areas = np.asarray(areas, dtype=np.float64)
    if areas.ndim != 1 or not len(areas) or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('areas must be a non-empty array of positive finite values')
    cumulative = np.cumsum(areas, dtype=np.float64)
    if spacing is not None:
        if total is not None or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('spacing must be finite, positive and used without total')
        quotas = cumulative / spacing / spacing
        if not np.isfinite(quotas[-1]) or quotas[-1] >= 2**63:
            raise ValueError('spacing produces too many samples for int64 row offsets')
        endpoints = np.floor(quotas + .5).astype(np.int64)
        if endpoints[-1] == 0:
            counts = np.zeros(len(areas), dtype=np.int64)
            counts[areas.argmax()] = 1
            return counts
    else:
        if isinstance(total, bool) or not isinstance(total, (int, np.integer)) or total < 1:
            raise ValueError('total must be a positive integer')
        endpoints = np.floor(cumulative * (int(total) / cumulative[-1]) + .5).astype(np.int64)
        endpoints[-1] = int(total)
    return np.diff(np.r_[0, endpoints])


def uniform_triangle_barycentrics(rng: np.random.Generator, count: int) -> np.ndarray:
    """Draw independent uniform points in a triangle using sqrt warping."""
    if count < 0:
        raise ValueError('count cannot be negative')
    random = rng.random((count, 2))
    root = np.sqrt(random[:, 0])
    return np.column_stack((1-root, root*(1-random[:, 1]), root*random[:, 1]))


def _write_ply_header(path: Path, count: int) -> np.memmap:
    header = ('ply\nformat binary_little_endian 1.0\n' + f'element vertex {count}\n' +
              ''.join(f'property float {key}\n' for key in ATTRIBUTES) + 'end_header\n').encode('ascii')
    with path.open('wb') as stream:
        stream.write(header)
        stream.truncate(len(header) + count * DTYPE.itemsize)
    return np.memmap(path, mode='r+', dtype=DTYPE, offset=len(header), shape=(count,))


def initialize_area_sampled(source, output_dir, *, total=None, spacing=None, seed=20261116,
                            sigma_factor=.8, opacity=.99, chunk_size=262_144):
    """Build an untrained model using only area sampling, barycentric interpolation and BaseColor lookup.

    Every valid source face receives an area-proportional integer quota. Samples
    within each face are independent uniform barycentric draws. There is no
    chart construction, importance weighting, visibility filtering, local
    repair, adaptive covariance, or camera-dependent selection.
    """
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not np.isfinite([sigma_factor, opacity]).all() or sigma_factor <= 0 or not 0 < opacity < 1:
        raise ValueError('sigma_factor must be positive and opacity must be in (0, 1)')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')

    if total is None and spacing is None:
        spacing = DEFAULT_SPACING
    requested_spacing = spacing
    total_start = time.perf_counter()
    source_sha256 = digest(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{output.name}.area-', dir=output.parent))
    staging = work/'model'
    staging.mkdir()
    try:
        geometry_dir = staging/'surface_metadata'
        tick = time.perf_counter()
        triangles, uv, groups, images, geometry = extract_geometry(source, geometry_dir)
        geometry_seconds = time.perf_counter()-tick
        if not len(triangles):
            raise ValueError('No non-degenerate mesh triangles')

        face_count = len(triangles)
        areas = np.empty(face_count, dtype=np.float64)
        normals = np.empty((face_count, 3), dtype=np.float32)
        tick = time.perf_counter()
        for start in range(0, face_count, chunk_size):
            end = min(start+chunk_size, face_count)
            part = triangles[start:end]
            cross = np.cross(part[:, 1]-part[:, 0], part[:, 2]-part[:, 0])
            twice_area = np.linalg.norm(cross, axis=1)
            if not np.isfinite(twice_area).all() or np.any(twice_area <= 0):
                raise ValueError('Extracted mesh contains a degenerate or non-finite triangle')
            areas[start:end] = .5*twice_area
            normals[start:end] = (cross/twice_area[:, None]).astype(np.float32)
        surface_area = float(areas.sum(dtype=np.float64))
        counts = allocate_area_counts(areas, total, spacing=spacing)
        row_offsets = np.empty(face_count+1, dtype=np.int64)
        row_offsets[0] = 0
        np.cumsum(counts, out=row_offsets[1:])
        if total is not None and row_offsets[-1] != total:
            raise ArithmeticError('Sample allocation did not preserve the requested total')
        total = int(row_offsets[-1])
        face_ids = np.repeat(np.arange(face_count, dtype=np.int32), counts)
        allocation_seconds = time.perf_counter()-tick

        tick = time.perf_counter()
        metadata = staging/'surface_metadata'
        face_out = open_memmap(metadata/'face.npy', mode='w+', dtype=np.int32, shape=(total,))
        bary_out = open_memmap(metadata/'barycentric.npy', mode='w+', dtype=np.float64, shape=(total, 2))
        boundary_out = open_memmap(metadata/'boundary.npy', mode='w+', dtype=np.bool_, shape=(total,))
        model = _write_ply_header(staging/'model.ply', total)
        output_setup_seconds = time.perf_counter()-tick

        mean_spacing = math.sqrt(surface_area/total)
        spacing = mean_spacing if requested_spacing is None else requested_spacing
        sigma = sigma_factor*spacing
        alpha_logit = math.log(opacity/(1-opacity))
        rng = np.random.default_rng(seed)
        from .blend import load_texture
        cache = ColourTextures(load_texture)
        sample_seconds = texture_cache_seconds = texture_seconds = pack_write_seconds = 0.
        for group_id, (group, image) in enumerate(zip(groups, images)):
            face_start, face_end = group['faces']
            row_start, row_end = int(row_offsets[face_start]), int(row_offsets[face_end])
            tick = time.perf_counter()
            texture = cache.get(image) if image is not None else None
            texture_cache_seconds += time.perf_counter()-tick
            for start in range(row_start, row_end, chunk_size):
                end = min(start+chunk_size, row_end)
                tick = time.perf_counter()
                sampled_faces = face_ids[start:end]
                bary = uniform_triangle_barycentrics(rng, end-start)
                selected = triangles[sampled_faces]
                points = np.einsum('ni,nij->nj', bary, selected)
                sample_uv = np.einsum('ni,nij->nj', bary, uv[sampled_faces]) if texture is not None else None
                sample_normals = normals[sampled_faces]
                rotations = Rotation.from_matrix(frame(sample_normals)).as_quat(scalar_first=True).astype(np.float32)
                edge_lengths = np.linalg.norm(selected[:, [1, 2, 0]]-selected[:, [2, 0, 1]], axis=-1)
                altitudes = (2*areas[sampled_faces, None])/np.maximum(edge_lengths, 1e-30)
                boundary = (bary*altitudes).min(axis=1) < 1.5*spacing
                sample_seconds += time.perf_counter()-tick

                tick = time.perf_counter()
                if texture is not None:
                    rgb = texture_colors(texture, sample_uv, linear=True)
                else:
                    rgb = np.broadcast_to(np.asarray(group['color'], dtype=np.float32), (end-start, 3))
                texture_seconds += time.perf_counter()-tick

                tick = time.perf_counter()
                rows = np.zeros(end-start, dtype=DTYPE)
                for axis, key in enumerate('xyz'):
                    rows[key] = points[:, axis]
                    rows['n'+key] = sample_normals[:, axis]
                    rows[f'f_dc_{axis}'] = (rgb[:, axis]-.5)/SH_C0
                    rows[f'scale_{axis}'] = math.log(sigma if axis < 2 else sigma*.02)
                rows['opacity'] = alpha_logit
                for axis in range(4):
                    rows[f'rot_{axis}'] = rotations[:, axis]
                model[start:end] = rows
                face_out[start:end] = sampled_faces
                bary_out[start:end] = bary[:, :2]
                boundary_out[start:end] = boundary
                pack_write_seconds += time.perf_counter()-tick
            if group_id % 250 == 0 or group_id+1 == len(groups):
                print(f'Area sampled/wrote {row_end:,}/{total:,} Gaussians ({group_id+1}/{len(groups)} groups)', flush=True)

        tick = time.perf_counter()
        for array in (model, face_out, bary_out, boundary_out):
            array.flush()
        flush_seconds = time.perf_counter()-tick
        del model, face_out, bary_out, boundary_out
        tick = time.perf_counter()
        model_sha256 = digest(staging/'model.ply')
        model_hash_seconds = time.perf_counter()-tick
        elapsed = time.perf_counter()-total_start
        timings = {
            'geometry_extract_and_cache': geometry_seconds,
            'face_areas_normals_and_allocation': allocation_seconds,
            'output_setup': output_setup_seconds,
            'barycentric_position_uv_and_frame': sample_seconds,
            'texture_cache_load_and_linearization': texture_cache_seconds,
            'texture_lookup': texture_seconds,
            'ply_and_attachment_metadata_write': pack_write_seconds,
            'flush': flush_seconds,
            'model_sha256': model_hash_seconds,
            'unaccounted_overhead': 0.,
            'total': elapsed,
        }
        timings['unaccounted_overhead'] = max(0., elapsed-sum(v for k,v in timings.items() if k != 'unaccounted_overhead' and k != 'total'))
        report = {
            'status': 'complete', 'method': 'global_mesh_area_barycentric_texture',
            'source': str(source), 'source_sha256': source_sha256,
            'model': 'model.ply', 'gaussians': int(total), 'seed': int(seed),
            'triangles': int(face_count), 'surface_area_world_units_squared': surface_area,
            'mean_area_spacing_world_units': mean_spacing, 'tangent_sigma_world_units': sigma,
            'target_spacing_world_units': requested_spacing,
            'normal_sigma_world_units': sigma*.02, 'sigma_factor': sigma_factor,
            'opacity': opacity, 'zero_sample_faces': int(np.count_nonzero(counts == 0)),
            'maximum_samples_on_one_face': int(counts.max()),
            'sampling': ('Exact-total per-face area allocation' if requested_spacing is None
                         else 'Cumulative world area at fixed spacing, rounded per face') +
                        '; independent uniform barycentric samples',
            'appearance': 'Per-face-corner UV interpolation; source BaseColor bilinear repeat lookup; SH0',
            'filtering': 'No charting, importance weights, visibility filtering, repair, or training',
            'timings_seconds': timings,
            'source_geometry': geometry,
            'model_sha256': model_sha256,
        }
        if requested_spacing is not None:
            report['method'] = 'mesh_area_spacing_barycentric_texture'
        if digest(source) != source_sha256:
            raise ValueError('Source scene changed during initialization')
        (staging/'area_sampling_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        initialization_report = {
            'status': 'complete', 'method': report['method'],
            'source': str(source), 'input_sha256': source_sha256,
            'model': 'model.ply', 'model_sha256': model_sha256,
            'gaussians': int(total), 'surface_area_world_units_squared': surface_area,
            'source_bounds': geometry['source_bounds'],
            'visibility': {'mode': 'auto', 'max_gap': .007},
            'surface_metadata': {'directory': 'surface_metadata'},
            'area_sampling_report': 'area_sampling_report.json',
            'training': False,
        }
        (staging/'initialization_report.json').write_text(
            json.dumps(initialization_report, ensure_ascii=False, indent=2)+'\n')
        if output.exists():
            raise FileExistsError(f'Output appeared during initialization: {output}')
        staging.rename(output)
        shutil.rmtree(work)
        return report
    except BaseException:
        (work/'failure.json').write_text(json.dumps({'status': 'failed'}, indent=2)+'\n')
        raise

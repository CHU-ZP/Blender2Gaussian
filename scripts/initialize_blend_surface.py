"""Initialize opaque BaseColor Gaussians from connected Blender surfaces.

Default pipeline: stratified chart sampling, local kNN covariance, chart
boundary limits (2 sigma; 1 sigma on coplanar seams), then removal of hidden
coincident layers by GPU ray casts. Kept rows are bit-identical to an
initialization without the removal.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial.transform import Rotation

from mesh_to_gs.surface import (allocate, charts, sample_component, covariance, coplanar_seams,
                               small_single_face_components, singleton_components, linear_texture, texture_colors)
from mesh_to_gs.adaptive import TextureCache, prepare_importance
from mesh_to_gs.surface_state import SurfaceMetadata


ATTRIBUTES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
              'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def group_samples(surface, counts, args, group_index,texture=None):
    # Last item: (faces, barycentrics, boundary flags) when metadata is requested.
    record = args.surface_metadata
    edge_sigmas = getattr(surface, 'edge_sigmas', None)
    sizes = np.bincount(surface.labels, minlength=len(counts))
    boundary_sizes = np.bincount(surface.labels[surface.boundary_faces],minlength=len(counts))
    single = (counts == 1) & (sizes <= 16) & (boundary_sizes <= 32)
    components = np.flatnonzero(single)
    for start in range(0,len(components),32768):
        p,n,uv,sc,q,nc,*faces = singleton_components(surface,components[start:start+32768],args.overlap,args.boundary_sigmas,record,edge_sigmas)
        yield p,n,uv,sc,q,nc,0,(*faces,np.zeros(len(p),bool)) if record else None
    fast = (sizes == 1) & (counts >= 2) & (counts <= 3)
    for count in (2, 3):
        components = np.flatnonzero(fast & (counts == count))
        for start in range(0, len(components), 32768):
            p,n,uv,scales,q,nc,*faces = small_single_face_components(
                surface, components[start:start+32768], count, args.seed, group_index,
                args.overlap, args.boundary_sigmas, record, edge_sigmas)
            yield p,n,uv,scales,q,nc,0,(*faces,np.zeros(len(p),bool)) if record else None
    for component in np.flatnonzero((counts > 0) & ~fast & ~single):
        seed = np.random.SeedSequence([args.seed, group_index, int(component)])
        p,n,uv,segments,spacing,nb,*faces = sample_component(surface, component, int(counts[component]), seed,texture,record)
        sigmas = args.boundary_sigmas
        if edge_sigmas is not None:
            edges = surface.boundary_groups[component]
            sigmas = edge_sigmas[surface.boundary_faces[edges], surface.boundary_corners[edges]]
        scales,q,nc = covariance(p,n,segments,spacing,args.overlap,sigmas)
        # Boundary samples are appended after the interior strata.
        yield p,n,uv,scales,q,nc,nb,(*faces,np.arange(len(p))>=len(p)-nb) if record else None


def coincident_hidden(tester, points, normals, scales, quats, max_gap, reach=4.):
    """Samples to drop: another surface lies within max_gap along the normal and
    neither the centre nor the +-reach sigma tangent footprint is visible.

    Such layers are sorted unreliably against their occluder and bleed its
    colour; hidden surfaces farther behind barely mix and are kept, as they
    still fill seams in the visible layer.
    """
    drop = np.zeros(len(points), bool)
    close = np.flatnonzero(tester.normal_gap(points, normals, max_gap) < max_gap)
    if len(close):
        rotation = Rotation.from_quat(np.asarray(quats[close], np.float64), scalar_first=True).as_matrix()
        sigma = np.asarray(scales[close, :2], np.float64)
        axes = np.stack((rotation[:, :, 0] * sigma[:, :1], rotation[:, :, 1] * sigma[:, 1:]), axis=1)
        drop[close] = ~tester.visible(points[close], axes, reach)
    return drop


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sk_00918_current'))
    budget_group=parser.add_mutually_exclusive_group()
    budget_group.add_argument('--budget', type=int)
    budget_group.add_argument('--base-spacing',type=float,help='World-space spacing on low-detail surfaces; infer total count from scene detail')
    parser.add_argument('--max-budget',type=int,default=25_000_000,help='Cap automatic allocation for the laptop GPU')
    parser.add_argument('--seed', type=int, default=3089248367)
    parser.add_argument('--crease-degrees', type=float, default=20)
    parser.add_argument('--detail-weight', type=float)
    parser.add_argument('--overlap', type=float, help='Tangent sigma / measured local spacing')
    parser.add_argument('--boundary-sigmas', type=float, default=2)
    parser.add_argument('--seam-sigmas', type=float, default=1.,
                        help='Boundary sigmas on chart edges shared with a coplanar face of another chart '
                             '(UV/material/object splits of one surface); 0 ignores them')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--surface-metadata', action='store_true',
                        help='Also write per-Gaussian face/barycentric correspondences and the evaluated geometry cache '
                             '(required by joint refinement)')
    parser.add_argument('--no-visibility', dest='visibility', action='store_false',
                        help='Keep hidden coincident layers (skip the GPU visibility pass)')
    parser.add_argument('--scene-type', choices=('auto', 'indoor', 'open'), default='auto',
                        help='Visibility viewpoints: enclosed interior space, or around objects without walls; '
                             'auto chooses open when almost no grid point is enclosed')
    # 7 mm sits in a sparse gap of the occluder-distance histogram; 10 mm would
    # split the many exactly 1 cm thick boards by micrometre rounding.
    parser.add_argument('--visibility-max-gap', type=float, default=.007,
                        help='Occluder distance along the normal below which a hidden layer is dropped (world units)')
    parser.add_argument('--visibility-grid', type=float, default=.25,
                        help='Spacing of candidate free-space viewpoints (world units)')
    parser.add_argument('--visibility-reach', type=float, default=4.,
                        help='Footprint probes at +-reach tangent sigma; smaller values also drop thin same-colour '
                             'backing layers that fill coverage of lettering and slats')
    parser.add_argument('--texture-weight',type=float)
    parser.add_argument('--curvature-weight',type=float)
    parser.add_argument('--detail-scale',type=float)
    parser.add_argument('--opacity',type=float)
    parser.add_argument('--calibration-profile',type=Path,
                        default=Path(__file__).resolve().parents[1]/'configs/basecolor_quality.json',
                        help='Sampling profile; explicit CLI values override it')
    args = parser.parse_args(argv)
    try:
        args.calibration=json.loads(args.calibration_profile.read_text())
    except (OSError, ValueError) as error:
        parser.error(f'Cannot load profile: {error}')
    defaults=dict(base_spacing=.024,detail_weight=.25,overlap=.8,
                  texture_weight=2.,curvature_weight=1.,detail_scale=1.,opacity=.99)
    if not isinstance(args.calibration,dict):parser.error('Profile must be a JSON object')
    unsupported=set(args.calibration)-defaults.keys()
    if unsupported:
        parser.error(f'Unsupported profile keys: {sorted(unsupported)}')
    requested_count=args.budget
    for key,value in defaults.items():
        if key=='base_spacing' and requested_count is not None:continue
        if getattr(args,key) is None:setattr(args,key,args.calibration.get(key,value))
    if not 0<args.opacity<1:parser.error('Opacity must be between 0 and 1')
    if (args.budget is not None and args.budget <= 0) or args.overlap <= 0 or args.boundary_sigmas <= 0 or min(args.detail_weight,args.texture_weight,args.curvature_weight,args.detail_scale) < 0:
        parser.error('Budget, overlap and boundary sigmas must be positive; detail weight must be nonnegative')
    if args.max_budget<=0 or (args.base_spacing is not None and args.base_spacing<=0):parser.error('Spacing and budget cap must be positive')
    if args.seam_sigmas<0:parser.error('Seam sigmas must be nonnegative')
    if min(args.visibility_max_gap,args.visibility_grid,args.visibility_reach)<=0:parser.error('Visibility gap, grid spacing and reach must be positive')
    return args


def main():
    import bpy
    from mesh_to_gs.blend import extract_scene, load_texture

    args=parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest()
    input_sha256=digest(args.input)
    root=Path(__file__).resolve().parents[1]
    source_hashes={}
    files=['scripts/initialize_blend_surface.py','mesh_to_gs/blend.py',
           'mesh_to_gs/surface.py','mesh_to_gs/adaptive.py','mesh_to_gs/surface_state.py',
           'mesh_to_gs/visibility.py','uv.lock']
    for name in files:
        target=args.output_dir/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
        source_hashes[name]=digest(root/name)
    shutil.copyfile(args.calibration_profile,args.output_dir/'source/profile.json')
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()), load_ui=False)
    groups, bounds, objects = extract_scene()
    print(f'Extracted {len(objects)} objects, {len(groups)} object/material groups', flush=True)
    surfaces = []
    loop_triangles = []
    for i, group in enumerate(groups.values()):
        triangles = np.concatenate(group.pop('triangles'))
        ids = np.concatenate(group.pop('vertex_ids'))
        uv = np.concatenate(group.pop('uvs')) if group['description']['image'] is not None else None
        s = charts(triangles, ids, uv, args.crease_degrees)
        surfaces.append(s)
        loop_triangles.append(np.concatenate(group.pop('loop_triangles'))[s.source])
        if (i + 1) % 250 == 0:
            print(f'Charts {i + 1}/{len(groups)}', flush=True)
    seam_edges = 0
    if args.seam_sigmas != args.boundary_sigmas:
        for surface, seam in zip(surfaces, coplanar_seams(surfaces, args.crease_degrees)):
            surface.edge_sigmas = np.where(seam, args.seam_sigmas, args.boundary_sigmas)
            seam_edges += int(seam.sum())
        print(f'Coplanar seam edges: {seam_edges:,}', flush=True)
    areas = np.concatenate([s.component_areas for s in surfaces])
    perimeters = np.concatenate([s.perimeter for s in surfaces])
    nominal_spacing = args.base_spacing or math.sqrt(areas.sum() / args.budget)
    # Boundary/crease/UV detail gets additional weight, capped so microscopic
    # disconnected facets cannot absorb an unbounded fraction of the budget.
    detail = np.minimum(args.detail_weight * nominal_spacing * perimeters, 2 * areas)
    importance_areas=[]
    feature_cache=TextureCache(load_texture)
    texture_score_sum=curvature_score_sum=0.
    for i,(surface,group) in enumerate(zip(surfaces,groups.values())):
        feature=feature_cache.get(group['description']['image'])
        importance=prepare_importance(surface,feature,nominal_spacing,args.texture_weight,args.curvature_weight,args.detail_scale)
        narrow=np.clip(1-2*surface.component_areas/np.maximum(surface.perimeter,1e-30)/(3*nominal_spacing),0,1)
        importance+=args.detail_scale*.5*args.curvature_weight*narrow[surface.labels]
        surface.importance=importance
        importance_areas.append(np.bincount(surface.labels,weights=surface.areas*importance,minlength=len(surface.component_areas)))
        texture_score_sum+=float(np.sum(surface.areas*surface.texture_scores))
        curvature_score_sum+=float(np.sum(surface.areas*surface.curvature_scores))
        if (i+1)%500==0:print(f'World-space features {i+1}/{len(groups)}',flush=True)
    weights=np.concatenate(importance_areas)+detail
    if args.base_spacing is not None:
        args.budget=min(args.max_budget,max(len(areas),int(round(weights.sum()/args.base_spacing**2))+len(areas)))
    allocations = allocate(weights,args.budget,minimum=True)
    original_allocations=allocate(areas+detail,args.budget,minimum=True)
    report = {
        'source': str(args.input), 'source_saved': False, 'status': 'analyzed',
        'method': 'Connected material/UV charts; stratified area + boundary allocation; local tangent kNN covariance; '
                  'seam-aware boundary limits; hidden coincident layer removal',
        'implementation': 'surface-adaptive-v2',
        'input_sha256':input_sha256,'source_hashes':source_hashes,
        'profile_sha256':digest(args.calibration_profile),
        'seed': args.seed, 'budget': args.budget, 'mesh_objects': len(objects),
        'triangles': sum(o['triangles'] for o in objects), 'groups': len(groups),
        'charts': len(areas), 'skipped_degenerate_faces': sum(s.skipped for s in surfaces),
        'surface_area_world_units_squared': float(areas.sum()), 'nominal_spacing_world_units': nominal_spacing,
        'detail_weight': args.detail_weight, 'crease_degrees': args.crease_degrees,
        'overlap_sigma_spacing_ratio': args.overlap, 'boundary_sigmas': args.boundary_sigmas,
        'seam_sigmas': args.seam_sigmas, 'coplanar_seam_edges': seam_edges,
        'boundary_rule': 'Tangent sigma perpendicular to chart edge <= edge distance / boundary_sigmas (seam_sigmas on coplanar seams); Gaussian tails and screen filter remain noncompact',
        'quaternion_convention': 'unit wxyz, local-to-world rotation, columns tangent/bitangent/normal',
        'scale_units': 'World units, log standard deviation in PLY', 'source_bounds': bounds,
        'alpha': 'Opaque reference; fixed global Gaussian opacity; original Principled Alpha ignored',
        'opacity':args.opacity,
        'calibration_profile':args.calibration,
        'color': 'SH0 sRGB display color; image bilinear interpolation in scene-linear RGB; no illumination or higher SH',
        'allocation_quantiles': np.quantile(allocations, [0, .5, .9, .99, 1]).tolist(),
        'zero_budget_charts': int((allocations == 0).sum()),
        'load_extract_chart_seconds': time.perf_counter() - start,
        'adaptive':True,'texture_weight':args.texture_weight,'curvature_weight':args.curvature_weight,'detail_scale':args.detail_scale,
        'budget_mode':'target-spacing' if args.base_spacing is not None else 'fixed-count',
        'base_spacing':args.base_spacing,'automatic_budget_cap':args.max_budget,
        'area_weighted_texture_score':texture_score_sum/float(areas.sum()),
        'area_weighted_curvature_score':curvature_score_sum/float(areas.sum()),
    }
    report_path = args.output_dir / 'initialization_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if args.analyze_only:
        return
    counts_per_group = np.split(allocations, np.cumsum([len(s.component_areas) for s in surfaces])[:-1])
    old_counts=np.split(original_allocations,np.cumsum([len(s.component_areas) for s in surfaces])[:-1])
    budget_changes=[dict(object=g['object_name'],material=g['description']['name'],baseline=int(o.sum()),adaptive=int(c.sum()))
                    for g,o,c in zip(groups.values(),old_counts,counts_per_group)]
    report['budget_largest_gains']=sorted(budget_changes,key=lambda r:r['adaptive']-r['baseline'],reverse=True)[:20]
    report['budget_largest_losses']=sorted(budget_changes,key=lambda r:r['adaptive']-r['baseline'])[:20]
    tester = None
    if args.visibility:
        from mesh_to_gs.visibility import VisibilityTester
        tick = time.perf_counter()
        tester = VisibilityTester.for_scene(np.concatenate([s.triangles for s in surfaces]), bounds, args.scene_type,
                                            spacing=args.visibility_grid)
        print(f'Visibility ({tester.mode} scene): {len(tester.viewpoints):,} free-space viewpoints; '
              f'{time.perf_counter()-tick:.1f}s', flush=True)
    body = args.output_dir / 'model.ply.body'
    done = capped = boundary_samples = sampled = removed = 0
    min_scale, max_scale = np.full(3, np.inf), np.zeros(3)
    max_quat_error = max_normal_error = 0.
    cache_key = None
    texture = None
    metadata = SurfaceMetadata(args.output_dir/'surface_metadata', surfaces, groups, args.budget) if args.surface_metadata else None
    with body.open('wb') as stream:
        for group_index, (group, surface, counts) in enumerate(zip(groups.values(), surfaces, counts_per_group)):
            description = group['description']
            image = description['image']
            feature=feature_cache.get(image)
            key = image.as_pointer() if image is not None else None
            if key != cache_key:
                texture = feature.image if feature is not None else (linear_texture(load_texture(image)) if image is not None else None)
                cache_key = key
            if metadata is not None:
                metadata.write_group(group_index, surface, loop_triangles[group_index])
            batches = list(group_samples(surface, counts, args, group_index,feature))
            if tester is not None and batches:
                # One ray-casting launch per group; rows are only removed, never altered.
                drops = np.split(coincident_hidden(tester, *(np.concatenate([b[k] for b in batches]) for k in (0, 1, 3, 4)),
                                                   args.visibility_max_gap, args.visibility_reach), np.cumsum([len(b[0]) for b in batches])[:-1])
            for i, (p,n,uv,scales,quats,nc,nb,faces) in enumerate(batches):
                rows = np.arange(sampled, sampled + len(p))
                sampled += len(p)
                capped += nc
                boundary_samples += nb
                if tester is not None:
                    keep = ~drops[i]
                    removed += int(drops[i].sum())
                    p, n, scales, quats, rows = p[keep], n[keep], scales[keep], quats[keep], rows[keep]
                    uv = None if uv is None else uv[keep]
                    faces = None if faces is None else tuple(f[keep] for f in faces)
                count = len(p)
                if not count:
                    continue
                if metadata is not None:
                    metadata.write_samples(group_index, done, *faces, rows)
                rgb = texture_colors(texture, uv, linear=True) if texture is not None else np.asarray(description['color'])
                values = np.empty((count, len(ATTRIBUTES)), dtype='<f4')
                values[:, :3], values[:, 3:6] = p, n
                values[:, 6:9] = (rgb - .5) / .28209479177387814
                values[:, 9] = math.log(args.opacity / (1-args.opacity))
                values[:, 10:13], values[:, 13:17] = np.log(scales), quats
                if not np.isfinite(values).all():
                    raise ValueError(f'Non-finite output: {group["object_name"]}')
                max_quat_error = max(max_quat_error, float(np.max(np.abs(np.linalg.norm(quats, axis=1) - 1))))
                # wxyz third column is the world normal.
                w, x, y, z = quats.T
                qnormal = np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))
                max_normal_error = max(max_normal_error, float(np.linalg.norm(qnormal - n, axis=1).max()))
                min_scale, max_scale = np.minimum(min_scale, scales.min(0)), np.maximum(max_scale, scales.max(0))
                values.tofile(stream)
                done += count
            # Release topology once its complete output has been streamed.
            surfaces[group_index] = None
            if (group_index + 1) % 50 == 0 or sampled == args.budget:
                print(f'Groups {group_index + 1}/{len(groups)}; {sampled:,}/{args.budget:,} sampled, '
                      f'{removed:,} hidden removed; {time.perf_counter()-start:.1f}s', flush=True)
    if sampled != args.budget or done + removed != sampled:
        raise ValueError(f'Budget mismatch: {sampled} sampled, {done} written, {removed} removed vs {args.budget}')
    model = args.output_dir / 'model.ply'
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex ' + str(done) + '\n'
    header += ''.join('property float ' + a + '\n' for a in ATTRIBUTES) + 'end_header\n'
    with model.open('wb') as stream, body.open('rb') as source:
        stream.write(header.encode('ascii'))
        shutil.copyfileobj(source, stream)
    body.unlink()
    if digest(args.input)!=input_sha256:raise ValueError('Source Blend changed during initialization')
    if metadata is not None:
        report['surface_metadata'] = metadata.finish(model, done)
    if tester is not None:
        report['visibility'] = dict(
            tester.report, max_gap=args.visibility_max_gap, reach_sigmas=args.visibility_reach, sampled=sampled, removed=removed,
            removed_fraction=removed / max(sampled, 1),
            rule='Drop a sample if another surface lies within max_gap along its normal and no ray from its centre '
                 'or +-reach_sigmas tangent footprint reaches any free-space viewpoint (nearest plus a fixed spread); '
                 'kept rows are identical to the unfiltered initialization',
            boundary_samples_note='boundary_samples and boundary_constraint_applications count all generated samples')
    report.update(status='complete', gaussians=done, boundary_samples=boundary_samples,
                  boundary_constraint_applications=capped, max_quaternion_unit_error=max_quat_error,
                  max_rotation_normal_error=max_normal_error, min_scales=min_scale.tolist(),
                  max_scales=max_scale.tolist(), total_seconds=time.perf_counter()-start, model=str(model))
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

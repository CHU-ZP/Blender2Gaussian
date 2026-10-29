"""Initialize opaque BaseColor Gaussians directly from connected .blend surfaces."""

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import bpy
import numpy as np

from mesh_to_gs.surface import (allocate, charts, sample_component, covariance,
                               small_single_face_components, singleton_components, linear_texture, texture_colors)
from validate_blend import extract_scene, load_texture
from mesh_to_gs.adaptive import TextureCache, prepare_importance,narrow_importance


ATTRIBUTES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
              'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def group_samples(surface, counts, args, group_index,texture=None):
    sizes = np.bincount(surface.labels, minlength=len(counts))
    boundary_sizes = np.bincount(surface.labels[surface.boundary_faces],minlength=len(counts))
    single = (counts == 1) & (sizes <= 16) & (boundary_sizes <= 32)
    components = np.flatnonzero(single)
    for start in range(0,len(components),32768):
        p,n,uv,sc,q,nc = singleton_components(surface,components[start:start+32768],args.overlap,args.boundary_sigmas)
        yield p,n,uv,sc,q,nc,0
    fast = (sizes == 1) & (counts >= 2) & (counts <= 3)
    for count in (2, 3):
        components = np.flatnonzero(fast & (counts == count))
        for start in range(0, len(components), 32768):
            p,n,uv,scales,q,nc = small_single_face_components(
                surface, components[start:start+32768], count, args.seed, group_index,
                args.overlap, args.boundary_sigmas)
            yield p,n,uv,scales,q,nc,0
    for component in np.flatnonzero((counts > 0) & ~fast & ~single):
        seed = np.random.SeedSequence([args.seed, group_index, int(component)])
        p,n,uv,segments,spacing,nb = sample_component(surface, component, int(counts[component]), seed,texture)
        scales,q,nc = covariance(p,n,segments,spacing,args.overlap,args.boundary_sigmas,
            max_normal_rotation=args.calibration.get('max_normal_rotation'),
            directional_support=args.calibration.get('directional_support',False))
        yield p,n,uv,scales,q,nc,nb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sk_00918_surface_physical'))
    budget_group=parser.add_mutually_exclusive_group()
    budget_group.add_argument('--budget', type=int)
    budget_group.add_argument('--base-spacing',type=float,help='World-space spacing on low-detail surfaces; infer total count from scene detail')
    parser.add_argument('--max-budget',type=int,default=25_000_000,help='Cap automatic allocation for the laptop GPU')
    parser.add_argument('--seed', type=int, default=3089248367)
    parser.add_argument('--crease-degrees', type=float, default=20)
    parser.add_argument('--detail-weight', type=float, default=.25)
    parser.add_argument('--overlap', type=float, default=.8, help='Tangent sigma / measured local spacing')
    parser.add_argument('--boundary-sigmas', type=float, default=2)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--adaptive',action='store_true',help='Allocate by world-space texture, curvature and narrow-chart detail')
    parser.add_argument('--texture-weight',type=float,default=2.)
    parser.add_argument('--curvature-weight',type=float,default=1.)
    parser.add_argument('--detail-scale',type=float,default=1.)
    parser.add_argument('--opacity',type=float,help='Global Gaussian opacity; overrides the profile')
    parser.add_argument('--calibration-profile',type=Path,help='JSON profile selected on analytic surface patches')
    args = parser.parse_args()
    requested_count,requested_spacing=args.budget,args.base_spacing
    args.calibration=json.loads(args.calibration_profile.read_text()) if args.calibration_profile else {}
    retired={'tau','edge_anisotropy','angular_support'}&args.calibration.keys()
    if retired:parser.error(f'Retired profile keys: {sorted(retired)}; use configs/basecolor_quality.json')
    if args.calibration:
        args.adaptive=True
        for key in ['overlap','texture_weight','curvature_weight','detail_scale','base_spacing','detail_weight']:
            if key=='base_spacing' and (requested_count is not None or requested_spacing is not None):continue
            if key in args.calibration:setattr(args,key,args.calibration[key])
    if args.budget is None:args.budget=10_701_421
    if args.opacity is None:args.opacity=args.calibration.get('opacity',.99)
    if not 0<args.opacity<1:parser.error('Opacity must be between 0 and 1')
    if args.budget <= 0 or args.overlap <= 0 or args.boundary_sigmas <= 0 or min(args.detail_weight,args.texture_weight,args.curvature_weight,args.detail_scale) < 0:
        parser.error('Budget, overlap and boundary sigmas must be positive; detail weight must be nonnegative')
    if args.max_budget<=0 or (args.base_spacing is not None and args.base_spacing<=0):parser.error('Spacing and budget cap must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()), load_ui=False)
    groups, _, _, bounds, objects, _ = extract_scene(surface_groups=True)
    print(f'Extracted {len(objects)} objects, {len(groups)} object/material groups', flush=True)
    surfaces = []
    for i, group in enumerate(groups.values()):
        triangles = np.concatenate(group.pop('triangles'))
        ids = np.concatenate(group.pop('vertex_ids'))
        uv = np.concatenate(group.pop('uvs')) if group['description']['image'] is not None else None
        s = charts(triangles, ids, uv, args.crease_degrees)
        surfaces.append(s)
        if (i + 1) % 250 == 0:
            print(f'Charts {i + 1}/{len(groups)}', flush=True)
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
        if args.adaptive:
            feature=feature_cache.get(group['description']['image'])
            importance=prepare_importance(surface,feature,nominal_spacing,args.texture_weight,args.curvature_weight,args.detail_scale,
                                          args.calibration.get('curvature_power',1))
            narrow=narrow_importance(surface,nominal_spacing,args.calibration.get('thin_support',False))
            importance+=args.detail_scale*args.curvature_weight*narrow[surface.labels]
            surface.importance=importance
            importance_areas.append(np.bincount(surface.labels,weights=surface.areas*importance,minlength=len(surface.component_areas)))
            texture_score_sum+=float(np.sum(surface.areas*surface.texture_scores))
            curvature_score_sum+=float(np.sum(surface.areas*surface.curvature_scores))
        else:importance_areas.append(surface.component_areas)
        if args.adaptive and (i+1)%500==0:print(f'World-space features {i+1}/{len(groups)}',flush=True)
    weights=np.concatenate(importance_areas)+detail
    if args.base_spacing is not None:
        args.budget=min(args.max_budget,max(len(areas),int(round(weights.sum()/args.base_spacing**2))+len(areas)))
    allocations = allocate(weights,args.budget,minimum=True)
    original_allocations=allocate(areas+detail,args.budget,minimum=True)
    report = {
        'source': str(args.input), 'source_saved': False, 'status': 'analyzed',
        'method': 'Connected material/UV charts; stratified area + boundary allocation; local tangent kNN covariance',
        'seed': args.seed, 'budget': args.budget, 'mesh_objects': len(objects),
        'triangles': sum(o['triangles'] for o in objects), 'groups': len(groups),
        'charts': len(areas), 'skipped_degenerate_faces': sum(s.skipped for s in surfaces),
        'surface_area_world_units_squared': float(areas.sum()), 'nominal_spacing_world_units': nominal_spacing,
        'detail_weight': args.detail_weight, 'crease_degrees': args.crease_degrees,
        'overlap_sigma_spacing_ratio': args.overlap, 'boundary_sigmas': args.boundary_sigmas,
        'boundary_rule': 'Tangent sigma perpendicular to chart edge <= edge distance / boundary_sigmas; Gaussian tails and screen filter remain noncompact',
        'quaternion_convention': 'unit wxyz, local-to-world rotation, columns tangent/bitangent/normal',
        'scale_units': 'World units, log standard deviation in PLY', 'source_bounds': bounds,
        'alpha': 'Opaque target; fixed global Gaussian opacity; original Principled Alpha ignored',
        'opacity':args.opacity,
        'color': 'SH0 sRGB display color; image bilinear interpolation in scene-linear RGB; no illumination or higher SH',
        'allocation_quantiles': np.quantile(allocations, [0, .5, .9, .99, 1]).tolist(),
        'zero_budget_charts': int((allocations == 0).sum()),
        'load_extract_chart_seconds': time.perf_counter() - start,
        'adaptive':args.adaptive,'texture_weight':args.texture_weight,'curvature_weight':args.curvature_weight,'detail_scale':args.detail_scale,
        'budget_mode':'target-spacing' if args.base_spacing is not None else 'fixed-count',
        'base_spacing':args.base_spacing,'automatic_budget_cap':args.max_budget,
        'area_weighted_texture_score':texture_score_sum/float(areas.sum()),
        'area_weighted_curvature_score':curvature_score_sum/float(areas.sum()),
        'calibration_profile':args.calibration,
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
    body = args.output_dir / 'model.ply.body'
    done = capped = boundary_samples = 0
    min_scale, max_scale = np.full(3, np.inf), np.zeros(3)
    max_quat_error = max_normal_error = 0.
    opacity_min,opacity_max=1.,0.
    cache_key = None
    texture = None
    with body.open('wb') as stream:
        for group_index, (group, surface, counts) in enumerate(zip(groups.values(), surfaces, counts_per_group)):
            description = group['description']
            image = description['image']
            feature=feature_cache.get(image) if args.adaptive else None
            key = image.as_pointer() if image is not None else None
            if key != cache_key:
                texture = feature.image if feature is not None else (linear_texture(load_texture(image)) if image is not None else None)
                cache_key = key
            for p,n,uv,scales,quats,nc,nb in group_samples(surface, counts, args, group_index,feature):
                count = len(p)
                rgb = texture_colors(texture, uv, linear=True) if texture is not None else np.asarray(description['color'])
                values = np.empty((count, len(ATTRIBUTES)), dtype='<f4')
                values[:, :3], values[:, 3:6] = p, n
                values[:, 6:9] = (rgb - .5) / .28209479177387814
                opacity=np.full(count,args.opacity,np.float32)
                values[:,9]=np.log(opacity/(1-opacity))
                opacity_min=min(opacity_min,float(opacity.min()));opacity_max=max(opacity_max,float(opacity.max()))
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
                capped += nc
                boundary_samples += nb
            # Release topology once its complete output has been streamed.
            surfaces[group_index] = None
            if (group_index + 1) % 50 == 0 or done == args.budget:
                print(f'Groups {group_index + 1}/{len(groups)}; {done:,}/{args.budget:,} Gaussians; {time.perf_counter()-start:.1f}s', flush=True)
    if done != args.budget:
        raise ValueError(f'Budget mismatch: {done} vs {args.budget}')
    model = args.output_dir / 'model.ply'
    header = 'ply\nformat binary_little_endian 1.0\nelement vertex ' + str(done) + '\n'
    header += ''.join('property float ' + a + '\n' for a in ATTRIBUTES) + 'end_header\n'
    with model.open('wb') as stream, body.open('rb') as source:
        stream.write(header.encode('ascii'))
        shutil.copyfileobj(source, stream)
    body.unlink()
    report.update(status='complete', gaussians=done, boundary_samples=boundary_samples,
                  boundary_constraint_applications=capped, max_quaternion_unit_error=max_quat_error,
                  max_rotation_normal_error=max_normal_error, min_scales=min_scale.tolist(),
                  max_scales=max_scale.tolist(), total_seconds=time.perf_counter()-start, model=str(model))
    report.update(opacity_range=[opacity_min,opacity_max])
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

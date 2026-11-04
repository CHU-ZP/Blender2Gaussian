"""Selected joint Mesh/render refinement: prioritized views, moving centres and interleaved splits.

Defaults match views72_priority1500; normals remain fixed with SH0 colour.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch.nn import functional as F

if __package__:
    from .render_gaussians import configure_cuda
else:
    from render_gaussians import configure_cuda
from mesh_to_gs.color import linear_to_srgb, srgb_to_linear
from mesh_to_gs.joint_refinement import (DynamicNeighbours, MovingGaussians, ScreenGradientSplits,
                                         quaternion_matrix, render_loss)
from mesh_to_gs.mesh_supervision import digest, model_inputs, prepare_queries, verify_empty_queries
from mesh_to_gs.mesh_targets import MeshTargets
from mesh_to_gs.view_sampling import ViewPriorities, detail_score, training_order

C0 = .28209479177387814


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--training-cameras', type=Path, help='Fixed training poses; otherwise generate geometry-accepted views')
    p.add_argument('--query-neighbours', type=int, default=64, help='Candidates used when regenerating the original Mesh query cache')
    p.add_argument('--reference-cache', type=Path, help='Existing refine directory with training cameras and EXRs')
    p.add_argument('--query-cache', type=Path, help='Verified Mesh query cache for this exact initializer')
    p.add_argument('--query-count', type=int, default=262144, help='New-cache samples per class')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--views', type=int, default=24)
    p.add_argument('--priority-uniform-fraction', type=float, default=.5)
    p.add_argument('--priority-ema', type=float, default=.9)
    p.add_argument('--sampling-reference-steps', type=int, default=0,
                   help='Mesh minibatch RNG preparation length used by the selected model')
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--dssim-weight', type=float, default=.2)
    p.add_argument('--mesh-weight', type=float, default=.1)
    p.add_argument('--mesh-colour-weight', type=float, default=.01)
    p.add_argument('--centre-weight', type=float, default=1.)
    p.add_argument('--uv-weight', type=float, default=.01)
    p.add_argument('--empty-weight', type=float, default=1.)
    p.add_argument('--tolerance', type=float, default=1e-4)
    p.add_argument('--mesh-pool', type=int, default=16384, help='Training subset per class from verified queries')
    p.add_argument('--mesh-batch', type=int, default=2048, help='Each of surface and empty minibatch sizes')
    p.add_argument('--centre-batch', type=int, default=4096)
    p.add_argument('--neighbours', type=int, default=32)
    p.add_argument('--refresh-every', type=int, default=50)
    p.add_argument('--position-lr', type=float, default=1.6e-4)
    p.add_argument('--scale-lr', type=float, default=.003)
    p.add_argument('--rotation-lr', type=float, default=.001)
    p.add_argument('--colour-lr', type=float, default=.003)
    p.add_argument('--opacity-lr', type=float, default=.01)
    p.add_argument('--scale-factor', type=float, default=2.)
    p.add_argument('--max-displacement', type=float, default=4., help='Numerical movement bound in initial major sigmas')
    p.add_argument('--split-start', type=int, default=100)
    p.add_argument('--split-stop', type=int, default=400)
    p.add_argument('--split-every', type=int, default=100)
    p.add_argument('--split-fraction', type=float, default=.02, help='Maximum parent fraction per event, relative to initial count')
    p.add_argument('--max-growth', type=float, default=.1, help='Maximum total count growth relative to initial count')
    p.add_argument('--uncapped-splits', action='store_true', help='Split every eligible parent; automatic for coverage inputs')
    p.add_argument('--retain-all-surfaces', action='store_true', help='Supervise every retained source face; automatic for coverage inputs')
    p.add_argument('--surface-query-mode', choices=('density', 'area'), default='density')
    p.add_argument('--split-threshold', type=float, default=2e-4, help='Average normalized projected-centre gradient')
    quality = json.loads((Path(__file__).resolve().parents[1]/'configs/joint_quality.json').read_text())
    p.set_defaults(**quality['refinement'])
    args = p.parse_args(argv)
    if args.reference_cache and args.training_cameras:
        p.error('Choose a reference cache or a training-camera file')
    if min(args.steps, args.width, args.height, args.views, args.mesh_pool, args.mesh_batch,
           args.centre_batch, args.neighbours, args.refresh_every, args.split_every) < 1:
        p.error('Counts and intervals must be positive')
    if args.query_count < 16384 or args.scale_factor < 1 or args.max_displacement <= 0:
        p.error('At least 16384 new queries, scale factor >= 1 and positive displacement')
    if not 0 <= args.dssim_weight <= 1 or min(args.mesh_weight, args.mesh_colour_weight, args.centre_weight,
                                           args.uv_weight, args.split_threshold) < 0:
        p.error('Nonnegative weights and split threshold; DSSIM weight in [0,1]')
    if min(args.empty_weight, args.tolerance, args.position_lr, args.scale_lr, args.rotation_lr,
           args.colour_lr, args.opacity_lr) <= 0 or not 0 <= args.split_fraction <= 1 or not 0 <= args.max_growth <= 1:
        p.error('Positive rates/tolerance, split fraction and growth in [0,1]')
    if args.split_start < 1 or args.split_stop < args.split_start:
        p.error('Split window must start at a positive step and end after its start')
    if not 0 <= args.priority_uniform_fraction <= 1 or not 0 <= args.priority_ema < 1 or args.sampling_reference_steps < 0:
        p.error('Uniform fraction in [0,1], EMA in [0,1), nonnegative reference steps')
    return args


def references(args, report, origin):
    from mesh_to_gs.mesh_reference import read_exr, render_references
    if args.reference_cache:
        cached_report = json.loads((args.reference_cache/'refinement_report.json').read_text())
        if Path(cached_report['blend']).resolve() != Path(report['source']).resolve():
            raise ValueError('Reference cache uses a different Blender scene')
        camera_data = json.loads((args.reference_cache/'training_cameras.json').read_text())
        cameras = camera_data['cameras'][:args.views]
        paths = [args.reference_cache/'training_references'/f'{c["name"]}.exr' for c in cameras]
    else:
        from mesh_to_gs.cameras import training_cameras
        from mesh_to_gs.visibility import VisibilityTester
        if args.training_cameras:
            camera_data = json.loads(args.training_cameras.read_text())
            if (camera_data['width'], camera_data['height']) != (args.width, args.height):
                raise ValueError('Training cameras must match the requested resolution')
            cameras = camera_data['cameras'][:args.views]
        else:
            points = PlyData.read(args.model_dir/'model.ply')['vertex'].data
            xyz = np.column_stack([points[k] for k in 'xyz']).astype(np.float64)
            _, metadata, _ = model_inputs(args.model_dir)
            tester = VisibilityTester.for_scene(np.load(metadata/'triangles.npy'),
                                               report['source_bounds'], report['visibility']['mode'])
            settings = json.loads((Path(__file__).resolve().parents[1]/'configs/joint_quality.json').read_text())['training_cameras']
            base_count = min(settings['base_views'], args.views)
            cameras = training_cameras(tester, xyz, base_count, settings['base_seed'], args.width, args.height)
            if args.views > base_count:
                extra = training_cameras(tester, xyz, args.views-base_count, settings['extra_seed'], args.width, args.height)
                for i, camera in enumerate(extra):
                    camera['name'] = f'extra_{i:03d}'
                cameras += extra
            del tester, xyz, points
        paths = [path for _, path, *_ in render_references(Path(report['source']), cameras,
                 args.width, args.height, args.output_dir/'training_references')]
        camera_data = dict(width=args.width, height=args.height, cameras=cameras, convention='OpenCV')
        import bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
    if len(cameras) != args.views:
        raise ValueError('Reference cache has fewer views than requested')
    camera_data['cameras'] = cameras
    (args.output_dir/'training_cameras.json').write_text(json.dumps(camera_data, indent=2)+'\n')
    targets, matrices, intrinsics, details = [], [], [], []
    for camera, path in zip(cameras, paths):
        rgb, alpha, _ = read_exr(path)
        image = torch.from_numpy((rgb+1-alpha[..., None]).astype(np.float32)).permute(2, 0, 1)[None]
        if image.shape[-2:] != (args.height, args.width):
            image = F.interpolate(image, (args.height, args.width), mode='area')
        target = linear_to_srgb(image[0].permute(1, 2, 0)).clamp(0, 1)
        targets.append(target.cuda())
        mask = torch.from_numpy((alpha >= .99).astype(np.float32))[None, None]
        if mask.shape[-2:] != (args.height, args.width):
            mask = F.interpolate(mask, (args.height, args.width), mode='nearest')
        details.append(detail_score(target.numpy(), mask[0, 0].numpy() >= .99))
        view = np.asarray(camera['view_matrix'], np.float64)
        view[:3, 3] += view[:3, :3]@origin
        matrices.append(torch.as_tensor(view, dtype=torch.float32, device='cuda')[None])
        K = np.asarray(camera['K'], np.float64)
        K[0] *= args.width/camera_data['width']
        K[1] *= args.height/camera_data['height']
        intrinsics.append(torch.as_tensor(K, dtype=torch.float32, device='cuda')[None])
    return cameras, targets, matrices, intrinsics, details


def render_training_view(model, camera, matrix, intrinsics, args, rasterization):
    q, scales = model.support()
    image, alpha, info = rasterization(model.position, q, scales, model.opacity_logits.sigmoid(), model.rgb,
            matrix, intrinsics, args.width, args.height, near_plane=camera['near'],
            far_plane=camera['far'], packed=True, eps2d=.05, rasterize_mode='classic')
    return linear_to_srgb((image[0]+1-alpha[0]).clamp(0, 1)).clamp(0, 1), info




def query_pool(args, report, origin):
    cache = args.query_cache or args.output_dir/'mesh_queries'
    if not (cache/'queries.json').exists():
        prepare_queries(args.model_dir, cache, args.query_count, args.query_neighbours, args.tolerance, args.seed,
                        args.retain_all_surfaces, args.surface_query_mode)
    saved = json.loads((cache/'queries.json').read_text())
    config = saved['configuration']
    if config['model_sha256'] != digest(args.model_dir/'model.ply') or \
            config['source_sha256'] != report['input_sha256'] or config['tolerance'] != args.tolerance:
        raise ValueError('Query cache belongs to a different initializer/scene/tolerance')
    if 'empty_label_verification' not in saved:
        saved = verify_empty_queries(args.model_dir, cache)
    points = np.load(cache/'points.npy', mmap_mode='r')
    colours = np.load(cache/'colour.npy', mmap_mode='r')
    empty = np.load(cache/'valid_empty.npy')
    n = saved['surface_queries']
    validation = min(2048, n//16, len(empty)//16)
    count = min(args.mesh_pool, n-validation, len(empty)-validation)
    rng = np.random.default_rng(args.seed)
    foreground = rng.choice(n-validation, count, replace=False)
    background = n+empty[rng.choice(len(empty)-validation, count, replace=False)]
    val_foreground = np.arange(n-validation, n)
    val_background = n+empty[-validation:]
    pool = (points[np.r_[foreground, background, val_foreground, val_background]]-origin).astype(np.float32)
    target_colours = np.concatenate((colours[foreground], colours[val_foreground]))
    return cache, saved, pool, target_colours, count, validation


@torch.no_grad()
def geometry_audit(model, mesh, rows=8192):
    ids = torch.linspace(0, len(model.position)-1, min(rows, len(model.position)), device='cuda').long()
    q, s = model.support(ids)
    normal = quaternion_matrix(q.double())[..., 2]
    initial_normal = quaternion_matrix(model.q_ref[ids].double())[..., 2]
    target, target_normal, _ = mesh.closest(model.position[ids], model.faces[ids])
    distance = (model.position[ids]-target).norm(dim=-1).cpu().numpy()
    displacement = (model.position[ids]-model.position_ref[ids]).norm(dim=-1).cpu().numpy()
    def angles(a, b):
        a, b = F.normalize(a.double(), dim=-1), F.normalize(b.double(), dim=-1)
        return torch.rad2deg(torch.atan2(torch.linalg.cross(a, b).norm(dim=-1),
                            (a*b).sum(-1).abs())).cpu().numpy()
    def summary(values):
        return dict(mean=float(values.mean()), maximum=float(values.max()),
                    quantiles=np.quantile(values, [0, .5, .9, .95, .99, 1]).tolist())
    return dict(rows=len(ids), centre_mesh_distance=summary(distance), centre_displacement=summary(displacement),
                normal_change_degrees=summary(angles(normal, initial_normal)),
                mesh_normal_error_degrees=summary(angles(normal, target_normal)),
                normal_convention='Unoriented plane normal, abs dot; relative to inherited initializer frame')


@torch.no_grad()
def save_model(model, data, origin, directory):
    roots = model.roots.cpu().numpy()
    out = np.array(data[roots], copy=True)
    xyz = model.position.cpu().numpy().astype(np.float64)+origin
    q, scales = model.support()
    normal = quaternion_matrix(q)[..., 2].cpu().numpy()
    q, scales = q.cpu().numpy(), scales.cpu().numpy()
    colour = (linear_to_srgb(model.rgb).clamp(0, 1).cpu().numpy()-.5)/C0
    for i, key in enumerate('xyz'):
        out[key] = xyz[:, i]
    for i, key in enumerate(('nx', 'ny', 'nz')):
        out[key] = normal[:, i]
    for i in range(3):
        out[f'scale_{i}'] = np.log(scales[:, i])
        out[f'f_dc_{i}'] = colour[:, i]
    for i in range(4):
        out[f'rot_{i}'] = q[:, i]
    out['opacity'] = model.opacity_logits.cpu().numpy()
    if not all(np.isfinite(out[k]).all() for k in out.dtype.names) or not (scales > 0).all():
        raise ValueError('Nonfinite output or nonpositive scales')
    path = directory/'model.ply'
    PlyData([PlyElement.describe(out, 'vertex')]).write(path)
    np.save(directory/'lineage.npy', roots)
    return dict(model=str(path), model_sha256=digest(path), gaussians=len(out),
                moved_rows=int(np.count_nonzero(np.linalg.norm(xyz-
                    np.column_stack([data[k][roots] for k in 'xyz']), axis=1) > 1e-6)))


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit('Use a new output directory; existing models are never overwritten')
    configure_cuda()
    from gsplat import rasterization
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    report, metadata, coverage_input = model_inputs(args.model_dir)
    args.uncapped_splits = bool(args.uncapped_splits or coverage_input)
    args.retain_all_surfaces = bool(args.retain_all_surfaces or coverage_input)
    origin = np.asarray(report['source_bounds'], np.float64).mean(0)
    tensor = lambda x: torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device='cuda')
    data = PlyData.read(args.model_dir/'model.ply', mmap='r')['vertex'].data
    positions = np.column_stack([data[k] for k in 'xyz']).astype(np.float64)-origin
    model = MovingGaussians(tensor(positions), tensor(np.column_stack([data[f'scale_{i}'] for i in range(3)])),
             tensor(np.column_stack([data[f'rot_{i}'] for i in range(4)])),
             srgb_to_linear((tensor(np.column_stack([data[f'f_dc_{i}'] for i in range(3)]))*C0+.5).clamp(0, 1)),
             tensor(data['opacity']), torch.as_tensor(np.array(np.load(metadata/'face.npy')),
                    device='cuda'), args.scale_factor, args.max_displacement)
    del positions
    initial_count = len(model.position)
    cameras, targets, matrices, K, details = references(args, report, origin)
    cache, query_report, pool, colours, pool_count, validation_count = query_pool(args, report, origin)
    points, colours = tensor(pool), tensor(colours)
    neighbours = DynamicNeighbours(pool, args.neighbours)
    print('neighbours', json.dumps(neighbours.rebuild(model, 0)), flush=True)
    mesh = MeshTargets(np.load(metadata/'triangles.npy', mmap_mode='r'), origin)
    def validation():
        with torch.no_grad():
            rows = torch.arange(2*pool_count, len(pool), device='cuda')
            return {k: float(v) for k, v in model.mesh_losses(points[rows], neighbours.ids[rows],
                    colours[pool_count:], validation_count, args.tolerance, args.empty_weight).items()}
    initial_validation = validation()
    initial_geometry = geometry_audit(model, mesh)
    rates = dict(position=args.position_lr, log_tangent=args.scale_lr, rotation=args.rotation_lr,
                 rgb=args.colour_lr, opacity_logits=args.opacity_lr)
    optimizer = torch.optim.Adam([dict(params=[p], lr=rates[name], name=name) for name, p in model.named_parameters()
                                  if p.requires_grad], eps=1e-15)
    splitter = ScreenGradientSplits(model)
    order, rng = training_order(args.seed, len(cameras), args.steps, args.sampling_reference_steps)
    view_rng = np.random.default_rng(args.seed+1)
    history, splits = [], []
    maximum_count = None if args.uncapped_splits else initial_count+int(initial_count*args.max_growth)
    initial_view_losses = np.zeros(len(cameras))
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            image, info = render_training_view(model, camera, matrices[index], K[index], args, rasterization)
            initial_view_losses[index] = max(0., float(render_loss(image, targets[index], args.dssim_weight)[0]))
            del image, info
    priorities = ViewPriorities(initial_view_losses, details, args.priority_uniform_fraction, args.priority_ema)
    prepared_seconds = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    print(f'Prepared {initial_count:,} Gaussians, {args.views} views, {pool_count} queries/class in {prepared_seconds:.1f}s', flush=True)
    for step, index in enumerate(order, 1):
        index = int(view_rng.choice(len(cameras), p=priorities.probabilities()))
        if step > 1 and (step-1) % args.refresh_every == 0 and neighbours.step != step-1:
            print('neighbours', json.dumps(neighbours.rebuild(model, step-1)), flush=True)
        optimizer.zero_grad(set_to_none=True)
        image, info = render_training_view(model, cameras[index], matrices[index], K[index], args, rasterization)
        info['means2d'].retain_grad()
        photometric, l1, ssim = render_loss(image, targets[index], args.dssim_weight)
        priorities.update(index, float(photometric.detach()))
        loss = photometric
        terms = {}
        if args.mesh_weight or args.mesh_colour_weight:
            foreground_rows = rng.integers(pool_count, size=args.mesh_batch)
            background_rows = rng.integers(pool_count, size=args.mesh_batch)
            fg = torch.as_tensor(foreground_rows, device='cuda')
            bg = pool_count+torch.as_tensor(background_rows, device='cuda')
            rows = torch.cat((fg, bg))
            terms = model.mesh_losses(points[rows], neighbours.ids[rows], colours[fg], len(fg),
                                      args.tolerance, args.empty_weight)
            loss = loss+args.mesh_weight*terms['geometry']+args.mesh_colour_weight*terms['appearance']
        if args.centre_weight:
            visible = info['gaussian_ids']
            half = args.centre_batch//2
            other = (visible[torch.randint(len(visible), (args.centre_batch-half,), device='cuda')] if len(visible) else
                     torch.randint(len(model.position), (args.centre_batch-half,), device='cuda'))
            selected = torch.cat((torch.randint(len(model.position), (half,), device='cuda'), other))
            closest, mesh_normal, _ = mesh.closest(model.position[selected], model.faces[selected])
            with torch.no_grad():
                model.attachment_point[selected] = closest
                model.attachment_normal[selected] = mesh_normal
            spacing = model.log_ref[selected, :2].exp().max(-1).values.detach().clamp(min=.001)
            centre = ((model.position[selected]-closest).square().sum(-1)/spacing.square()).mean()
            plane = model.attachment_loss()
            loss = loss+args.centre_weight*(centre+plane)
            terms['centre'] = centre
            terms['centre_plane'] = plane
        if args.uv_weight:
            prior = (model.rgb-model.rgb_ref).square().mean()
            loss = loss+args.uv_weight*prior
            terms['uv_prior'] = prior
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at step {step}')
        loss.backward()
        splitter.accumulate(info)
        if step == 1 or step % 25 == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        model.clamp_()
        if args.split_start <= step <= args.split_stop and step % args.split_every == 0:
            limit = None if args.uncapped_splits else max(0, min(int(initial_count*args.split_fraction), maximum_count-len(model.position)))
            parents = splitter.choose(model, limit, args.split_threshold)
            event = dict(step=step, parents=len(parents), before=len(model.position), threshold=args.split_threshold)
            model.split(parents, optimizer)
            event['after'] = len(model.position)
            splits.append(event)
            splitter.reset(model)
            if len(parents):
                print('neighbours', json.dumps(neighbours.rebuild(model, step)), flush=True)
            print('split', json.dumps(event), flush=True)
        if step == 1 or step % 25 == 0 or step == args.steps:
            entry = dict(step=step, loss=float(loss.detach()), render=float(photometric.detach()), l1=float(l1.detach()),
                         camera=cameras[index]['name'],
                         ssim=float(ssim.detach()), gaussians=len(model.position), neighbour_age=step-neighbours.step,
                         terms={k: float(v.detach()) for k, v in terms.items()}, validation=validation(),
                         seconds=time.perf_counter()-tick)
            history.append(entry)
            print(json.dumps(entry), flush=True)
        del image, info, loss, photometric, terms, l1, ssim
    optimization_seconds = time.perf_counter()-tick
    neighbours.rebuild(model, args.steps)
    final_validation = validation()
    audit = geometry_audit(model, mesh)
    saved = save_model(model, data, origin, args.output_dir)
    if coverage_input:
        target = args.output_dir/'surface'
        shutil.copytree(metadata, target)
        roots = model.roots.cpu().numpy()
        for name in ('barycentric', 'boundary', 'sample'):
            np.save(target/f'{name}.npy', np.load(metadata/f'{name}.npy')[roots])
        np.save(target/'face.npy', model.faces.cpu().numpy())
    result = dict(status='complete', method='joint_mesh_render_dynamic_gaussians', **saved,
                  blend=report['source'], source_sha256=report['input_sha256'],
                  coverage_status='invalidated by optimization; independent recertification required' if coverage_input else 'not certified',
                  gaussian_count_limit=maximum_count, split_policy='all threshold-eligible parents' if args.uncapped_splits else 'legacy fraction/growth caps',
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  input_model=str(args.model_dir/'model.ply'), input_sha256=digest(args.model_dir/'model.ply'),
                  origin=origin.tolist(), initial_count=initial_count, splits=splits, neighbours=neighbours.history,
                  neighbour_audit=neighbours.audit(model, pool_count), initial_validation=initial_validation,
                  final_validation=final_validation, initial_geometry=initial_geometry, final_geometry=audit,
                  query_cache=str(cache), query_pool_per_class=pool_count, query_validation_per_class=validation_count,
                  query_verification=query_report['empty_label_verification'], history=history,
                  view_sampling='priority', split_schedule='interleaved',
                  sampling_policy='Online loss/detail priorities',
                  view_statistics=[dict(camera=c['name'], updates=int(priorities.counts[i]),
                      detail=details[i], initial_render_loss=float(priorities.initial_losses[i]),
                      final_loss_ema=float(priorities.losses[i]), probability=float(priorities.probabilities()[i]))
                      for i, c in enumerate(cameras)],
                  preparation_seconds=prepared_seconds, optimization_seconds=optimization_seconds,
                  total_seconds=time.perf_counter()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gpu=torch.cuda.get_device_name(0),
                  render_objective='(1-0.2)*L1 + 0.2*(1-SSIM); full sRGB image after linear white compositing',
                  mesh_objective='surface/empty support + surface colour + dense local-plane and exact sampled centre/Mesh distance + initial UV prior',
                  normal_mode='fixed inherited normal; in-plane rotation',
                  rule='Positive tangent scales; SH0; normal thickness 2% of minor; RGB/opacity trainable; two-child long-axis splits',
                  limitations=['Finite periodically rebuilt neighbours approximate the spatial support field',
                               'SH0 surface parameterization and bounded motion/scales differ from unrestricted upstream 3DGS',
                               'No opacity reset, pruning or cloning; two-child replacement uses screen-gradient parent selection',
                               'Optimization completion does not certify continuous surface coverage'])
    (args.output_dir/'refinement_report.json').write_text(json.dumps(result, indent=2)+'\n')
    root = Path(__file__).resolve().parents[1]
    for name in ('scripts/refine_joint.py', 'scripts/render_gaussians.py', 'mesh_to_gs/joint_refinement.py', 'mesh_to_gs/mesh_targets.py', 'mesh_to_gs/view_sampling.py',
                 'mesh_to_gs/mesh_supervision.py', 'mesh_to_gs/color.py', 'configs/joint_quality.json', 'uv.lock'):
        target = args.output_dir/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    print(json.dumps(saved), flush=True)


if __name__ == '__main__':
    main()

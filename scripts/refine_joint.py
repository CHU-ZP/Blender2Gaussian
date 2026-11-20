"""Surface-bound joint refinement with prioritized training views and interleaved splits."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

if __package__:
    from .render_gaussians import configure_cuda
else:
    from render_gaussians import configure_cuda
from mesh_to_gs.color import (MATERIAL_EXPOSURE_STOPS, aces_fitted,
                              composite_display_rgb, linear_to_srgb, srgb_to_linear)
from mesh_to_gs.joint_refinement import (DynamicNeighbours, MovingGaussians, ScreenGradientSplits,
                                         alpha_supervision_loss, depth_supervision_loss,
                                         quaternion_matrix, render_loss)
from mesh_to_gs.mesh_supervision import digest, model_inputs, prepare_queries, verify_empty_queries
from mesh_to_gs.mesh_targets import MeshTargets
from mesh_to_gs.repair_geometry import frame
from mesh_to_gs.view_sampling import ViewPriorities, detail_score, training_order
from mesh_to_gs.voxel_constraints import load_voxel_bounds, audit_voxel_bounds, voxel_export_positions

C0 = .28209479177387814


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path)
    selected, _ = selector.parse_known_args(argv)
    overrides = {}
    if selected.config:
        try:
            overrides = json.loads(selected.config.read_text())
        except (OSError, ValueError) as error:
            selector.error(f'Cannot read training config: {error}')
        if not isinstance(overrides, dict):
            selector.error('Training config must be a JSON object')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='JSON argument defaults; relative paths in JSON resolve from the repository root')
    p.add_argument('--dry-run', action='store_true', help='Print resolved configuration without preparing queries, rendering or training')
    p.add_argument('--profile', choices=('joint',), default='joint')
    p.add_argument('model_dir', type=Path, nargs='?', help='Initializer directory; required unless supplied by --config')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--training-cameras', type=Path, help='Fixed training poses; otherwise generate geometry-accepted views')
    p.add_argument('--query-neighbours', type=int, default=64, help='Candidates used when regenerating the original Mesh query cache')
    p.add_argument('--reference-cache', type=Path, help='Existing refine directory with training cameras and EXRs')
    p.add_argument('--reference-mode', choices=('basecolor', 'material'), default='basecolor',
                   help='Reference appearance: emission BaseColor or scene material/light transport')
    p.add_argument('--reference-samples', type=int, default=16,
                   help='Cycles samples per training reference')
    p.add_argument('--material-prediction-space', choices=('display_linear', 'scene_linear'),
                   default='scene_linear',
                   help='display_linear fits final PBR appearance without a second exposure/ACES; scene_linear reproduces legacy radiance fitting')
    p.add_argument('--discard-reference-exrs', action='store_true',
                   help='Remove newly rendered EXRs after loading targets; keep RGB previews and provenance')
    p.add_argument('--reference-preview-width', type=int, default=0,
                   help='Maximum width of saved reference PNG previews; 0 saves full resolution')
    p.add_argument('--query-cache', type=Path, help='Verified Mesh query cache for this exact initializer')
    p.add_argument('--query-count', type=int, default=262144, help='New-cache samples per class')
    p.add_argument('--constrain-to-voxels', action='store_true',
                   help='Keep octree centres and split children inside their original leaf voxel')
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
    p.add_argument('--baked-colour-weight', type=float, default=0.,
                   help='Training-view PBR point-colour prior; also initializes observed colours from those views')
    p.add_argument('--max-hue-shift', type=float, default=0.,
                   help='Bound opaque point hue relative to reliable training PBR or source colour, allowing desaturation')
    p.add_argument('--footprint-groups', default='',
                   help='Comma-separated opaque source group IDs with physical patch-edge tangent-footprint limits')
    p.add_argument('--depth-weight', type=float, default=0.)
    p.add_argument('--alpha-weight', type=float, default=0.)
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
    p.add_argument('--sh-degree', type=int, choices=range(4), default=0,
                   help='Trainable SH degree; degree zero keeps the legacy RGB parameterization')
    p.add_argument('--sh-warmup-steps', type=int, default=0,
                   help='Enable higher SH bands progressively over this many steps')
    p.add_argument('--geometry-loss-steps', type=int, default=0,
                   help='Linearly decay centre/depth/alpha auxiliary losses to zero')
    p.add_argument('--geometry-freeze-steps', type=int, default=0,
                   help='Freeze centres, scales and rotation for this many initial updates')
    p.add_argument('--opacity-freeze-steps', type=int, default=0,
                   help='Freeze opacity for this many initial updates')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Export intermediate models and full training-view metrics at this interval')
    p.add_argument('--material-semantics', type=Path,
                   help='Verified source-material metadata and opaque first-hit depth masks')
    p.add_argument('--bind-source-faces', action='store_true',
                   help='Project centres to their original triangles after each geometry update and split')
    p.add_argument('--prune-start', type=int, default=0)
    p.add_argument('--prune-every', type=int, default=0)
    p.add_argument('--prune-opacity-threshold', type=float, default=.005)
    p.add_argument('--prune-min-visibility', type=int, default=1)
    p.add_argument('--opacity-reset-start', type=int, default=0)
    p.add_argument('--opacity-reset-every', type=int, default=0)
    p.add_argument('--opacity-reset-value', type=float, default=.05)
    p.add_argument('--scale-factor', type=float, default=2.)
    p.add_argument('--max-displacement', type=float, default=4., help='Numerical movement bound in initial major sigmas')
    p.add_argument('--warmup-steps', type=int, default=0,
                   help='Delay splitting and ramp centre/scale learning rates during initial stabilization')
    p.add_argument('--warmup-position-factor', type=float, default=.25)
    p.add_argument('--warmup-scale-factor', type=float, default=.5)
    p.add_argument('--split-start', type=int, default=100)
    p.add_argument('--split-stop', type=int, default=400)
    p.add_argument('--split-every', type=int, default=100)
    p.add_argument('--split-fraction', type=float, default=.02, help='Maximum parent fraction per event, relative to initial count')
    p.add_argument('--max-growth', type=float, default=.1, help='Maximum total count growth relative to initial count')
    p.add_argument('--uncapped-splits', action='store_true', help='Split every eligible parent')
    p.add_argument('--retain-all-surfaces', action='store_true', help='Supervise every retained source face')
    p.add_argument('--filter-hidden-surfaces', action='store_true',
                   help='Visibility-filter surface queries while retaining source faces with no initializer samples')
    p.add_argument('--surface-query-mode', choices=('density', 'area'), default='density')
    p.add_argument('--split-threshold', type=float, default=2e-4, help='Average normalized projected-centre gradient')
    p.add_argument('--residual-candidate-count', type=int, default=0,
                   help='Fixed visible surface candidates checked for under-supported Mesh regions')
    p.add_argument('--residual-add-start', type=int, default=0)
    p.add_argument('--residual-add-every', type=int, default=0)
    p.add_argument('--residual-max-add-per-event', type=int, default=0)
    p.add_argument('--residual-support-threshold', type=float, default=.35)
    p.set_defaults(**json.loads((root/'configs/pbr_v10_area_surface_bound.json').read_text()))
    actions = {action.dest: action for action in p._actions
               if action.dest not in ('help', 'config', 'dry_run', 'output_dir')}
    for key, value in overrides.items():
        if key not in actions:
            p.error(f'Unknown or unsupported training config key: {key}')
        action = actions[key]
        if action.type is Path:
            if value is not None and not isinstance(value, str):
                p.error(f'{key} must be a path string or null')
            if value is not None:
                value = Path(value)
                value = value if value.is_absolute() else root/value
        elif isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                p.error(f'{key} must be a JSON boolean')
        elif action.type in (int, float):
            allowed = (int,) if action.type is int else (int, float)
            if isinstance(value, bool) or not isinstance(value, allowed) or not math.isfinite(value):
                p.error(f'{key} must be a finite {action.type.__name__}')
            value = action.type(value)
        elif not isinstance(value, str):
            p.error(f'{key} must be a string')
        if action.choices is not None and value not in action.choices:
            p.error(f'Invalid {key}: {value!r}; choose from {list(action.choices)}')
        overrides[key] = value
    p.set_defaults(**overrides)
    args = p.parse_args(argv)
    if args.model_dir is None:
        p.error('An initializer model_dir is required, either positionally or in --config')
    if args.reference_cache and args.training_cameras:
        p.error('Choose a reference cache or a training-camera file')
    if args.material_semantics and (args.residual_candidate_count or args.constrain_to_voxels):
        p.error('Material semantics currently support source-face models without residual additions')
    if args.footprint_groups:
        if (not args.bind_source_faces or args.residual_candidate_count or
                any(not v.isdigit() for v in args.footprint_groups.split(','))):
            p.error('Footprint limits require original-face binding, valid source group IDs and no residual additions')
    if (args.baked_colour_weight or args.max_hue_shift) and (not args.material_semantics or not args.depth_weight or
                                    args.reference_mode != 'material' or args.residual_candidate_count or
                                    args.material_prediction_space != 'display_linear'):
        p.error('Baked PBR colour prior requires verified material/depth metadata and no residual additions')
    if args.reference_samples < 1 or args.reference_preview_width < 0:
        p.error('Reference samples must be positive and preview width nonnegative')
    if not math.isfinite(args.max_hue_shift) or (args.max_hue_shift and args.sh_degree):
        p.error('Hue constraint requires a finite nonnegative tolerance and SH0')
    if min(args.steps, args.width, args.height, args.views, args.mesh_pool, args.mesh_batch,
           args.centre_batch, args.neighbours, args.refresh_every, args.split_every) < 1:
        p.error('Counts and intervals must be positive')
    if args.query_count < 16384 or args.scale_factor < 1 or args.max_displacement <= 0:
        p.error('At least 16384 new queries, scale factor >= 1 and positive displacement')
    if not 0 <= args.dssim_weight <= 1 or min(args.mesh_weight, args.mesh_colour_weight, args.centre_weight,
                                           args.uv_weight, args.split_threshold, args.depth_weight,
                                           args.alpha_weight, args.baked_colour_weight, args.max_hue_shift) < 0:
        p.error('Nonnegative weights and split threshold; DSSIM weight in [0,1]')
    if min(args.empty_weight, args.tolerance, args.position_lr, args.scale_lr, args.rotation_lr,
           args.colour_lr, args.opacity_lr) <= 0 or not 0 <= args.split_fraction <= 1 or not 0 <= args.max_growth <= 1:
        p.error('Positive rates/tolerance, split fraction and growth in [0,1]')
    if args.split_start < 1 or args.split_stop < args.split_start:
        p.error('Split window must start at a positive step and end after its start')
    if args.warmup_steps < 0 or not 0 < args.warmup_position_factor <= 1 or not 0 < args.warmup_scale_factor <= 1:
        p.error('Warmup steps must be nonnegative and warmup learning-rate factors in (0,1]')
    if min(args.residual_candidate_count, args.residual_add_start, args.residual_add_every,
           args.residual_max_add_per_event) < 0 or not 0 <= args.residual_support_threshold <= 1:
        p.error('Residual-sampling controls must be nonnegative and support threshold in [0,1]')
    if args.residual_candidate_count and min(args.residual_add_start, args.residual_add_every,
                                             args.residual_max_add_per_event) < 1:
        p.error('Residual candidates require a positive start step, interval and per-event budget')
    if not 0 <= args.priority_uniform_fraction <= 1 or not 0 <= args.priority_ema < 1 or args.sampling_reference_steps < 0:
        p.error('Uniform fraction in [0,1], EMA in [0,1), nonnegative reference steps')
    if min(args.sh_warmup_steps, args.geometry_loss_steps, args.geometry_freeze_steps,
           args.opacity_freeze_steps, args.checkpoint_every) < 0:
        p.error('SH and auxiliary-loss schedules must be nonnegative')
    if min(args.prune_start, args.prune_every, args.prune_min_visibility,
           args.opacity_reset_start, args.opacity_reset_every) < 0:
        p.error('Pruning and opacity-reset controls must be nonnegative')
    if not 0 < args.prune_opacity_threshold < 1 or not 0 < args.opacity_reset_value < 1:
        p.error('Opacity thresholds and reset values must lie in (0,1)')
    return args


def references(args, report, origin):
    from mesh_to_gs.mesh_reference import (MATERIAL_DISPLAY_TRANSFORM, MATERIAL_EXPOSURE_STOPS,
                                           MATERIAL_LIGHTING_PROFILE, read_exr, render_references)
    if args.reference_cache:
        cached_report = json.loads((args.reference_cache/'refinement_report.json').read_text())
        if Path(cached_report['blend']).resolve() != Path(report['source']).resolve():
            raise ValueError('Reference cache uses a different Blender scene')
        cached_mode = cached_report.get('reference_mode')
        if cached_mode is not None and cached_mode != args.reference_mode:
            raise ValueError(f'Reference cache mode is {cached_mode!r}, requested {args.reference_mode!r}')
        camera_data = json.loads((args.reference_cache/'training_cameras.json').read_text())
        cameras = camera_data['cameras'][:args.views]
        reference_items = ((camera, args.reference_cache/'training_references'/f'{camera["name"]}.exr', False)
                           for camera in cameras)
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
            _, metadata = model_inputs(args.model_dir)
            tester = VisibilityTester.for_scene(np.load(metadata/'triangles.npy'),
                                               report['source_bounds'], report['visibility']['mode'])
            base_count = min(24, args.views)
            cameras = training_cameras(tester, xyz, base_count, 20261101, args.width, args.height)
            if args.views > base_count:
                extra = training_cameras(tester, xyz, args.views-base_count, 2026110301, args.width, args.height)
                for i, camera in enumerate(extra):
                    camera['name'] = f'extra_{i:03d}'
                cameras += extra
            del tester, xyz, points
        reference_items = ((camera, path, True) for camera, path, *_ in render_references(
            Path(report['source']), cameras, args.width, args.height,
            args.output_dir/'training_references', samples=args.reference_samples,
            material=args.reference_mode == 'material'))
        camera_data = dict(width=args.width, height=args.height, cameras=cameras, convention='OpenCV')
    if len(cameras) != args.views:
        raise ValueError('Reference cache has fewer views than requested')
    camera_data['cameras'] = cameras
    (args.output_dir/'training_cameras.json').write_text(json.dumps(camera_data, indent=2)+'\n')
    depth_manifest = None
    if args.material_semantics:
        depth_manifest = json.loads((args.material_semantics/'manifest.json').read_text())
        if (depth_manifest['source_sha256'] != report['input_sha256'] or
                depth_manifest['camera_sha256'] != digest(args.output_dir/'training_cameras.json')):
            raise ValueError('Material/depth metadata uses a different source or camera set')
    targets, target_depths, depth_masks, target_alphas = [], [], [], []
    matrices, intrinsics, details = [], [], []
    reference_stats = []
    if args.depth_weight:
        from scipy.ndimage import binary_dilation, binary_erosion
    rendered_references = args.reference_cache is None
    if args.discard_reference_exrs:
        # A reused reference cache keeps its EXRs in the cache directory, but
        # still writes small per-run previews here. Ensure the destination
        # exists for both freshly rendered and cached references.
        (args.output_dir/'training_references').mkdir(parents=True, exist_ok=True)
    for camera, path, temporary in reference_items:
        rgb, alpha, depth = read_exr(path)
        if args.reference_mode == 'material':
            radiance = np.divide(rgb, alpha[..., None], out=np.zeros_like(rgb),
                                 where=alpha[..., None] > 1e-6)
            rgb = aces_fitted(radiance * (2. ** MATERIAL_EXPOSURE_STOPS)) * alpha[..., None]
        image = torch.from_numpy((rgb+1-alpha[..., None]).astype(np.float32)).permute(2, 0, 1)[None]
        if image.shape[-2:] != (args.height, args.width):
            image = F.interpolate(image, (args.height, args.width), mode='area')
        target = linear_to_srgb(image[0].permute(1, 2, 0)).clamp(0, 1)
        targets.append(target.cuda())
        if args.discard_reference_exrs:
            from PIL import Image
            pixels = (target.numpy()*255).round().astype(np.uint8)
            preview = Image.fromarray(pixels)
            if args.reference_preview_width and preview.width > args.reference_preview_width:
                preview.thumbnail((args.reference_preview_width, args.reference_preview_width), Image.Resampling.LANCZOS)
            preview.save(args.output_dir/'training_references'/f'{camera["name"]}.png')
            foreground = np.asarray(alpha) >= .99
            clipped_foreground = (float(np.all(pixels == 255, axis=-1)[foreground].mean())
                                  if foreground.any() else 1.)
            reference_stats.append(dict(camera=camera['name'], mean_srgb=float(target.mean()),
                                        clipped_fraction=float((pixels == 255).mean()),
                                        foreground_fraction=float(foreground.mean()),
                                        clipped_foreground_fraction=clipped_foreground,
                                        preview_width=preview.width, preview_height=preview.height))
        if args.alpha_weight:
            alpha_target = torch.from_numpy(np.asarray(alpha, np.float32))[None, None]
            if alpha_target.shape[-2:] != (args.height, args.width):
                alpha_target = F.interpolate(alpha_target, (args.height, args.width), mode='bilinear', align_corners=False)
            target_alphas.append(alpha_target[0, 0].clamp(0, 1).cuda())
        if args.depth_weight:
            depth = np.asarray(depth, np.float32)
            valid = np.isfinite(depth) & (depth > camera['near']) & (depth < camera['far'])
            edge = np.zeros(depth.shape, bool)
            for axis in (0, 1):
                a = [slice(None), slice(None)]; b = [slice(None), slice(None)]
                a[axis], b[axis] = slice(None, -1), slice(1, None)
                a, b = tuple(a), tuple(b)
                jump = np.abs(depth[a]-depth[b]) > np.maximum(.05, .02*np.minimum(depth[a], depth[b]))
                jump = (jump & valid[a] & valid[b]) | (valid[a] != valid[b])
                edge[a] |= jump; edge[b] |= jump
            interior = binary_erosion(valid, iterations=3) & ~binary_dilation(edge, iterations=3)
            if depth_manifest:
                mask_path = args.material_semantics/'opaque_depth_masks'/f'{camera["name"]}.npy'
                saved_mask = depth_manifest['depth_masks'][camera['name']]
                if digest(mask_path) != saved_mask['sha256'] or digest(path) != saved_mask['reference_exr_sha256']:
                    raise ValueError('Opaque depth mask or corresponding GT changed')
                interior &= np.load(mask_path)
            depth_tensor = torch.from_numpy(np.where(valid, depth, 0).astype(np.float32))[None, None]
            mask_tensor = torch.from_numpy(interior.astype(np.float32))[None, None]
            if depth_tensor.shape[-2:] != (args.height, args.width):
                depth_tensor = F.interpolate(depth_tensor, (args.height, args.width), mode='nearest')
                mask_tensor = F.interpolate(mask_tensor, (args.height, args.width), mode='nearest')
            target_depths.append(depth_tensor[0, 0].cuda())
            depth_masks.append(mask_tensor[0, 0].bool().cuda())
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
        if temporary and args.discard_reference_exrs:
            # Each multilayer EXR is only an interchange file. Free it as soon
            # as its RGB/depth/alpha targets have been transferred to tensors,
            # so a 216-view material run needs only one reference file of
            # temporary disk space at a time.
            path.unlink()
        del rgb, alpha, depth, image, target, mask
    if rendered_references:
        import bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
    (args.output_dir/'training_reference_report.json').write_text(json.dumps(dict(
        reference_mode=args.reference_mode,
        reference_lighting=MATERIAL_LIGHTING_PROFILE if args.reference_mode == 'material' else None,
        display_transform=MATERIAL_DISPLAY_TRANSFORM if args.reference_mode == 'material' else 'standard',
        material_exposure_stops=MATERIAL_EXPOSURE_STOPS if args.reference_mode == 'material' else 0.,
        samples=args.reference_samples, source_sha256=report['input_sha256'],
        exrs_retained=(args.reference_cache is not None or not args.discard_reference_exrs),
        reference_cache=None if args.reference_cache is None else str(args.reference_cache),
        views=reference_stats), indent=2)+'\n')
    return cameras, targets, target_depths, depth_masks, target_alphas, matrices, intrinsics, details


def active_sh_degree(args, step):
    if not args.sh_degree:
        return 0
    if not args.sh_warmup_steps:
        return args.sh_degree
    # Start from DC, then add one band at a time.  The last band is enabled at
    # the end of the requested warmup interval.
    return min(args.sh_degree, max(0, int(step*args.sh_degree/args.sh_warmup_steps)))


def prediction_display_transform(args):
    return ('aces_fitted' if args.reference_mode == 'material' and
            args.material_prediction_space == 'scene_linear' else 'standard')


def set_update_stage(model, args, step):
    """No gradients or Adam updates for frozen parameters, including momentum."""
    for name in ('position', 'log_tangent', 'rotation'):
        getattr(model, name).requires_grad_(step > args.geometry_freeze_steps)
    model.opacity_logits.requires_grad_(step > args.opacity_freeze_steps)


def render_training_view(model, camera, matrix, intrinsics, args, rasterization, step=0):
    q, scales = model.support()
    colours = model.rgb
    if model.sh_degree:
        from gsplat.cuda._wrapper import spherical_harmonics
        degree = active_sh_degree(args, step)
        camera_position = torch.linalg.inv(matrix[0])[:3, 3]
        directions = model.position-camera_position
        # gsplat evaluates SH in sRGB and adds 0.5.  Decode before the
        # physically correct linear compositing used by the legacy path.
        evaluated = spherical_harmonics(degree, directions, model.sh_coefficients())
        colours = srgb_to_linear((evaluated+.5).clamp(0, 1))
    image, alpha, info = rasterization(model.position, q, scales, model.opacity_logits.sigmoid(), colours,
            matrix, intrinsics, args.width, args.height, near_plane=camera['near'],
            far_plane=camera['far'], packed=True, eps2d=.05, rasterize_mode='classic',
            render_mode='RGB+ED' if args.depth_weight else 'RGB')
    if args.depth_weight:
        info['expected_depth'] = image[0, ..., 3]
        image = image[..., :3]
    image = composite_display_rgb(image[0], alpha[0],
                display_transform=prediction_display_transform(args),
                exposure_stops=MATERIAL_EXPOSURE_STOPS)
    return image, alpha[0, ..., 0], info




def query_pool(args, report, origin):
    cache = args.query_cache or args.output_dir/'mesh_queries'
    if not (cache/'queries.json').exists():
        prepare_queries(args.model_dir, cache, args.query_count, args.query_neighbours, args.tolerance, args.seed,
                        args.retain_all_surfaces, args.surface_query_mode, args.filter_hidden_surfaces)
    saved = json.loads((cache/'queries.json').read_text())
    config = saved['configuration']
    if config['model_sha256'] != digest(args.model_dir/'model.ply') or \
            config['source_sha256'] != report['input_sha256'] or config['tolerance'] != args.tolerance:
        raise ValueError('Query cache belongs to a different initializer/scene/tolerance')
    if (saved['surface_queries'] != args.query_count or config['k'] != args.query_neighbours or
            config['seed'] != args.seed or bool(config.get('retain_all_surfaces', False)) != args.retain_all_surfaces or
            config.get('surface_query_mode', 'density') != args.surface_query_mode or
            bool(config.get('filter_hidden_surface', False)) != args.filter_hidden_surfaces):
        raise ValueError('Query cache uses a different count, seed, retention policy or surface distribution')
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
    residual = None
    if args.residual_candidate_count:
        candidate_count = min(args.residual_candidate_count, n)
        all_faces = np.load(cache/'surface_face.npy', mmap_mode='r')
        all_bary = np.load(cache/'surface_barycentric.npy', mmap_mode='r')
        boundary_quota = candidate_count//2
        candidate_rows = np.empty(candidate_count, dtype=np.int64)
        boundary_count = 0
        if boundary_quota:
            triangles = np.load(model_inputs(args.model_dir)[1]/'triangles.npy', mmap_mode='r')
            area_spacing = json.loads((args.model_dir/'area_sampling_report.json').read_text())['mean_area_spacing_world_units']
            faces = np.asarray(all_faces, dtype=np.int64)
            bary = np.asarray(all_bary, dtype=np.float64)
            tri = np.asarray(triangles[faces], dtype=np.float64)
            side = np.linalg.norm(tri[:, [1, 2, 0]]-tri[:, [2, 0, 1]], axis=-1)
            twice_area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
            altitude = twice_area[:, None]/np.maximum(side, 1e-30)
            boundary_rows = np.flatnonzero((bary*altitude).min(axis=1) < 1.5*area_spacing)
            boundary_count = min(boundary_quota, len(boundary_rows))
            if boundary_count:
                candidate_rows[:boundary_count] = rng.choice(boundary_rows, boundary_count, replace=False)
        remaining = candidate_count-boundary_count
        if remaining:
            available = np.ones(n, dtype=bool)
            if boundary_count:
                available[candidate_rows[:boundary_count]] = False
            rows = np.flatnonzero(available)
            candidate_rows[boundary_count:] = rng.choice(rows, remaining, replace=False)
        candidate_points = (points[candidate_rows]-origin).astype(np.float32)
        candidate_faces = np.asarray(all_faces[candidate_rows], dtype=np.int32)
        candidate_bary = np.asarray(all_bary[candidate_rows], dtype=np.float64)
        candidate_colours = np.asarray(colours[candidate_rows], dtype=np.float32)
        candidate_start = len(pool)
        pool = np.concatenate((pool, candidate_points), axis=0)
        residual = dict(start=candidate_start, rows=candidate_count, faces=candidate_faces,
                        barycentric=candidate_bary, colours=candidate_colours,
                        boundary_candidates=boundary_count,
                        used=np.zeros(candidate_count, dtype=bool))
    return cache, saved, pool, target_colours, count, validation, residual


@torch.no_grad()
def geometry_audit(model, mesh, rows=8192):
    count = len(model.position)
    sample_count = min(rows, count)
    # Float32 can round the last valid index up to count above 2**24 rows.
    ids = torch.arange(sample_count, device=model.position.device) * (count-1) // max(sample_count-1, 1)
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
    if model.sh_degree:
        extra = [(f'f_rest_{i}', '<f4') for i in range(3*((model.sh_degree+1)**2-1))]
        out = np.empty(len(roots), dtype=np.dtype(data.dtype.descr + extra))
        for name in data.dtype.names:
            out[name] = data[name][roots]
    else:
        out = np.array(data[roots], copy=True)
    xyz = model.position.cpu().numpy().astype(np.float64)+origin
    voxel_export = None
    if model.voxel_lower is not None:
        xyz, voxel_export = voxel_export_positions(model, xyz, origin)
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
    if model.sh_degree:
        rest = model.sh_rest.detach().cpu().numpy().transpose(0, 2, 1).reshape(len(out), -1)
        for i in range(rest.shape[1]):
            out[f'f_rest_{i}'] = rest[:, i]
    for i in range(4):
        out[f'rot_{i}'] = q[:, i]
    out['opacity'] = model.opacity_logits.cpu().numpy()
    if not all(np.isfinite(out[k]).all() for k in out.dtype.names) or not (scales > 0).all():
        raise ValueError('Nonfinite output or nonpositive scales')
    path = directory/'model.ply'
    PlyData([PlyElement.describe(out, 'vertex')]).write(path)
    np.save(directory/'lineage.npy', roots)
    result = dict(model=str(path), model_sha256=digest(path), gaussians=len(out),
                  moved_rows=int(np.count_nonzero(np.linalg.norm(xyz-
                      np.column_stack([data[k][roots] for k in 'xyz']), axis=1) > 1e-6)))
    if voxel_export is not None:
        result['voxel_export'] = voxel_export
    return result


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}, indent=2))
        return
    from mesh_to_gs.mesh_reference import MATERIAL_DISPLAY_TRANSFORM, MATERIAL_LIGHTING_PROFILE
    if args.output_dir.exists():
        raise SystemExit('Use a new output directory; existing models are never overwritten')
    configure_cuda()
    from gsplat import rasterization
    args.output_dir.mkdir(parents=True)
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    report, metadata = model_inputs(args.model_dir)
    origin = np.asarray(report['source_bounds'], np.float64).mean(0)
    tensor = lambda x: torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device='cuda')
    data = PlyData.read(args.model_dir/'model.ply', mmap='r')['vertex'].data
    if any(name.startswith('f_rest_') for name in data.dtype.names):
        raise ValueError('The initializer must contain SH0; resume from a trained SH model is not supported')
    positions = np.column_stack([data[k] for k in 'xyz']).astype(np.float64)-origin
    voxel_bounds = voxel_record = None
    if args.constrain_to_voxels:
        lower, upper, voxel_record = load_voxel_bounds(args.model_dir, report, origin, positions)
        voxel_bounds = (tensor(lower), tensor(upper))
        del lower, upper
    model = MovingGaussians(tensor(positions), tensor(np.column_stack([data[f'scale_{i}'] for i in range(3)])),
             tensor(np.column_stack([data[f'rot_{i}'] for i in range(4)])),
             srgb_to_linear((tensor(np.column_stack([data[f'f_dc_{i}'] for i in range(3)]))*C0+.5).clamp(0, 1)),
             tensor(data['opacity']), torch.as_tensor(np.array(np.load(metadata/'face.npy')),
                    device='cuda'), args.scale_factor, args.max_displacement,
             voxel_bounds=voxel_bounds, sh_degree=args.sh_degree,
             fixed_source_bounds=args.bind_source_faces)
    del positions
    del voxel_bounds
    material_record = None
    if args.material_semantics:
        from mesh_to_gs.material_semantics import apply_material_semantics
        material_record = apply_material_semantics(model, args.material_semantics,
                            args.model_dir/'model.ply', report['input_sha256'])
        (args.output_dir/'material_adaptation.json').write_text(json.dumps(material_record, indent=2)+'\n')
    if voxel_record is not None:
        voxel_record['initial_projection'] = model.voxel_initial_projection
        voxel_record['initial_audit'] = audit_voxel_bounds(model)
        shutil.copyfile(args.model_dir/'leaf_cells.npy', args.output_dir/'voxel_cells.npy')
        (args.output_dir/'voxel_constraints.json').write_text(json.dumps(voxel_record, indent=2)+'\n')
    footprints = footprint_record = None
    if args.footprint_groups:
        from mesh_to_gs.footprint_constraints import SourcePatchFootprints
        groups = json.loads((metadata/'groups.json').read_text())
        footprints = SourcePatchFootprints(model, np.load(metadata/'triangles.npy', mmap_mode='r'),
                     groups, list(map(int, args.footprint_groups.split(','))), origin=origin)
        footprints.project_(model)
        footprints.save(args.output_dir)
        footprint_record = footprints.record
        (args.output_dir/'footprint_adaptation.json').write_text(json.dumps(footprint_record, indent=2)+'\n')
    initial_count = len(model.position)
    cameras, targets, target_depths, depth_masks, target_alphas, matrices, K, details = references(args, report, origin)
    baked_record = None
    if args.baked_colour_weight or args.max_hue_shift:
        from mesh_to_gs.appearance_prior import bake_training_appearance
        baked_record = bake_training_appearance(model, targets, target_depths, depth_masks, matrices, K, cameras)
        baked_record.update(input_sha256=digest(args.model_dir/'model.ply'),
                            source_sha256=report['input_sha256'],
                            training_camera_sha256=digest(args.output_dir/'training_cameras.json'))
        (args.output_dir/'appearance_adaptation.json').write_text(json.dumps(baked_record, indent=2)+'\n')
        print('baked_appearance', json.dumps(baked_record), flush=True)
    if args.max_hue_shift:
        model.max_hue_shift = args.max_hue_shift
        hue_reference = torch.where((model.baked_weight >= .5)[:, None], model.baked_rgb, model.rgb_ref)
        np.save(args.output_dir/'hue_reference.npy', hue_reference.cpu().numpy())
        baked_record.update(hue_constraint=dict(tolerance=args.max_hue_shift,
            reference='Training PBR where confidence >=0.5, source BaseColor otherwise; transparent rows excluded',
            allows='RGB mean changes and desaturation toward neutral; mean reduced if display gamut requires it',
            reference_sha256=digest(args.output_dir/'hue_reference.npy')))
        (args.output_dir/'appearance_adaptation.json').write_text(json.dumps(baked_record, indent=2)+'\n')
        del hue_reference
        model.clamp_()
        print('hue_constraint', json.dumps(baked_record['hue_constraint']), flush=True)
    mesh_enabled = bool(args.mesh_weight or args.mesh_colour_weight or args.residual_candidate_count)
    if mesh_enabled:
        cache, query_report, pool, colours, pool_count, validation_count, residual = query_pool(args, report, origin)
        points, colours = tensor(pool), tensor(colours)
        neighbours = DynamicNeighbours(pool, args.neighbours)
        print('neighbours', json.dumps(neighbours.rebuild(model, 0)), flush=True)
    else:
        cache = query_report = neighbours = residual = None
        pool = colours = points = None
        pool_count = validation_count = 0
    mesh = MeshTargets(np.load(metadata/'triangles.npy', mmap_mode='r'), origin)
    def validation():
        if neighbours is None:
            return {}
        with torch.no_grad():
            rows = torch.arange(2*pool_count, 2*pool_count+2*validation_count, device='cuda')
            return {k: float(v) for k, v in model.mesh_losses(points[rows], neighbours.ids[rows],
                    colours[pool_count:], validation_count, args.tolerance, args.empty_weight).items()}
    initial_validation = validation()
    initial_geometry = geometry_audit(model, mesh)
    rates = dict(position=args.position_lr, log_tangent=args.scale_lr, rotation=args.rotation_lr,
                 rgb=args.colour_lr, opacity_logits=args.opacity_lr, sh_rest=args.colour_lr)
    optimizer = torch.optim.Adam([dict(params=[p], lr=rates[name], name=name) for name, p in model.named_parameters()
                                  if p.requires_grad], eps=1e-15)
    base_lrs = {group['name']: group['lr'] for group in optimizer.param_groups}
    splitter = ScreenGradientSplits(model)
    order, rng = training_order(args.seed, len(cameras), args.steps, args.sampling_reference_steps)
    view_rng = np.random.default_rng(args.seed+1)
    history, splits = [], []
    maximum_count = None if args.uncapped_splits else initial_count+int(initial_count*args.max_growth)
    initial_view_losses = np.zeros(len(cameras))
    triangles_world = np.load(metadata/'triangles.npy', mmap_mode='r')
    visibility = torch.zeros(len(model.position), dtype=torch.int32, device='cuda')
    opacity_resets = []
    prune_events = []
    residual_events = []
    initial_render_metrics = []
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            image, alpha, info = render_training_view(model, camera, matrices[index], K[index], args, rasterization, step=0)
            objective, mae, ssim = render_loss(image, targets[index], args.dssim_weight)
            initial_view_losses[index] = max(0., float(objective))
            mse = (image-targets[index]).square().mean().clamp(min=1e-12)
            initial_render_metrics.append(dict(camera=camera['name'], loss=float(objective),
                mae=float(mae), ssim=float(ssim), psnr_db=float(-10*torch.log10(mse))))
            del image, info
    initial_render_summary = {key: float(np.mean([row[key] for row in initial_render_metrics]))
                              for key in ('loss', 'mae', 'ssim', 'psnr_db')}
    (args.output_dir/'initial_render_metrics.json').write_text(json.dumps(dict(
        summary=initial_render_summary, views=initial_render_metrics,
        prediction_display_transform=prediction_display_transform(args)), indent=2)+'\n')
    print('initial_render', json.dumps(initial_render_summary), flush=True)
    priorities = ViewPriorities(initial_view_losses, details, args.priority_uniform_fraction, args.priority_ema)
    prepared_seconds = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    print(f'Prepared {initial_count:,} Gaussians, {args.views} views, {pool_count} queries/class in {prepared_seconds:.1f}s', flush=True)
    for step, index in enumerate(order, 1):
        set_update_stage(model, args, step)
        index = int(view_rng.choice(len(cameras), p=priorities.probabilities()))
        if neighbours is not None and step > 1 and (step-1) % args.refresh_every == 0 and neighbours.step != step-1:
            print('neighbours', json.dumps(neighbours.rebuild(model, step-1)), flush=True)
        optimizer.zero_grad(set_to_none=True)
        image, alpha, info = render_training_view(model, cameras[index], matrices[index], K[index], args, rasterization, step=step)
        if info['means2d'].requires_grad:
            info['means2d'].retain_grad()
        if len(info['gaussian_ids']):
            visibility.index_add_(0, info['gaussian_ids'], torch.ones_like(info['gaussian_ids'], dtype=torch.int32))
        photometric, l1, ssim = render_loss(image, targets[index], args.dssim_weight)
        priorities.update(index, float(photometric.detach()))
        loss = photometric
        terms = {}
        auxiliary_factor = (max(0., 1.-step/args.geometry_loss_steps)
                            if args.geometry_loss_steps else 1.)
        if args.alpha_weight and auxiliary_factor:
            terms['alpha'] = alpha_supervision_loss(alpha, target_alphas[index])
            loss = loss + args.alpha_weight*auxiliary_factor*terms['alpha']
        if args.depth_weight and auxiliary_factor:
            terms['depth'] = depth_supervision_loss(info['expected_depth'], target_depths[index], depth_masks[index])
            loss = loss + args.depth_weight*auxiliary_factor*terms['depth']
        if args.mesh_weight or args.mesh_colour_weight:
            foreground_rows = rng.integers(pool_count, size=args.mesh_batch)
            background_rows = rng.integers(pool_count, size=args.mesh_batch)
            fg = torch.as_tensor(foreground_rows, device='cuda')
            bg = pool_count+torch.as_tensor(background_rows, device='cuda')
            rows = torch.cat((fg, bg))
            mesh_terms = model.mesh_losses(points[rows], neighbours.ids[rows], colours[fg], len(fg),
                                           args.tolerance, args.empty_weight)
            terms.update(mesh_terms)
            loss = loss+args.mesh_weight*mesh_terms['geometry']+args.mesh_colour_weight*mesh_terms['appearance']
        if args.centre_weight and auxiliary_factor:
            visible = info['gaussian_ids']
            half = args.centre_batch//2
            other = (visible[torch.randint(len(visible), (args.centre_batch-half,), device='cuda')] if len(visible) else
                     torch.randint(len(model.position), (args.centre_batch-half,), device='cuda'))
            selected = torch.cat((torch.randint(len(model.position), (half,), device='cuda'), other))
            closest, mesh_normal, _ = (mesh.closest_owning if args.bind_source_faces else mesh.closest)(
                model.position[selected], model.faces[selected])
            with torch.no_grad():
                model.attachment_point[selected] = closest
                model.attachment_normal[selected] = mesh_normal
            spacing = model.log_ref[selected, :2].exp().max(-1).values.detach().clamp(min=.001)
            centre = ((model.position[selected]-closest).square().sum(-1)/spacing.square()).mean()
            plane = model.attachment_loss()
            loss = loss+args.centre_weight*auxiliary_factor*(centre+plane)
            terms['centre'] = centre
            terms['centre_plane'] = plane
        if args.uv_weight:
            prior = (model.rgb-model.rgb_ref).square().mean()
            loss = loss+args.uv_weight*prior
            terms['uv_prior'] = prior
        if args.baked_colour_weight:
            from mesh_to_gs.appearance_prior import appearance_prior_loss
            terms['baked_colour'] = appearance_prior_loss(model)
            loss = loss+args.baked_colour_weight*terms['baked_colour']
        if args.warmup_steps:
            warm = min(1., step/args.warmup_steps)
            for group in optimizer.param_groups:
                if group['name'] in ('position', 'log_tangent'):
                    factor = args.warmup_position_factor if group['name'] == 'position' else args.warmup_scale_factor
                    group['lr'] = base_lrs[group['name']] * (factor + (1-factor)*warm)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at step {step}')
        loss.backward()
        if info['means2d'].requires_grad:
            splitter.accumulate(info)
        if step == 1 or step % 25 == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        model.clamp_()
        if args.bind_source_faces and step > args.geometry_freeze_steps:
            closest, _, _ = mesh.closest_owning(model.position, model.faces)
            with torch.no_grad():
                model.position.copy_(closest)
        if footprints is not None:
            footprints.project_(model)
        if (args.opacity_reset_every and step >= args.opacity_reset_start and
                (step-args.opacity_reset_start) % args.opacity_reset_every == 0):
            model.reset_opacity(args.opacity_reset_value)
            state = optimizer.state.get(model.opacity_logits, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key].zero_()
            opacity_resets.append(dict(step=step, value=args.opacity_reset_value,
                                       gaussians=len(model.position)))
            print('opacity_reset', json.dumps(opacity_resets[-1]), flush=True)
        if (args.prune_every and step >= args.prune_start and
                (step-args.prune_start) % args.prune_every == 0):
            low_opacity = model.opacity_logits.sigmoid() < args.prune_opacity_threshold
            remove = low_opacity & (visibility >= args.prune_min_visibility) & ~model.frozen
            before = len(model.position)
            removed = model.prune(remove, optimizer)
            if removed:
                visibility = visibility[~remove]
                neighbours = (None if neighbours is None else neighbours)
                splitter = ScreenGradientSplits(model)
                if neighbours is not None:
                    neighbours.rebuild(model, step)
            prune_events.append(dict(step=step, removed=removed, before=before,
                                     after=len(model.position), threshold=args.prune_opacity_threshold))
            print('prune', json.dumps(prune_events[-1]), flush=True)
        if (args.split_start <= step <= args.split_stop and
                step > max(args.warmup_steps, args.geometry_freeze_steps, args.opacity_freeze_steps) and
                step % args.split_every == 0):
            limit = None if args.uncapped_splits else max(0, min(int(initial_count*args.split_fraction), maximum_count-len(model.position)))
            parents = splitter.choose(model, limit, args.split_threshold)
            event = dict(step=step, parents=len(parents), before=len(model.position), threshold=args.split_threshold)
            model.split(parents, optimizer)
            if args.bind_source_faces and len(parents):
                closest, _, _ = mesh.closest_owning(model.position, model.faces)
                with torch.no_grad():
                    model.position.copy_(closest)
            if footprints is not None:
                footprints.project_(model)
            event['after'] = len(model.position)
            splits.append(event)
            # Child rows start with no visibility history.  The next renders
            # repopulate the counters before a pruning event.
            visibility = torch.zeros(len(model.position), dtype=torch.int32, device='cuda')
            splitter.reset(model)
            if len(parents) and neighbours is not None:
                print('neighbours', json.dumps(neighbours.rebuild(model, step)), flush=True)
            print('split', json.dumps(event), flush=True)
        if (residual is not None and step >= args.residual_add_start and
                step % args.residual_add_every == 0 and
                (maximum_count is None or len(model.position) < maximum_count)):
            with torch.no_grad():
                candidate_rows = np.flatnonzero(~residual['used'])
                if len(candidate_rows):
                    ids = torch.as_tensor(residual['start'] + candidate_rows, device='cuda')
                    support_ids = neighbours.ids[ids]
                    support = model.occupancy(points[ids], support_ids, args.tolerance).cpu().numpy()
                    chosen = candidate_rows[support < args.residual_support_threshold]
                    remaining = args.residual_max_add_per_event if maximum_count is None else min(
                        args.residual_max_add_per_event, maximum_count-len(model.position))
                    chosen = chosen[:remaining]
                    if len(chosen):
                        faces = residual['faces'][chosen]
                        selected_triangles = np.asarray(triangles_world[faces], dtype=np.float64)
                        normals = np.cross(selected_triangles[:, 1]-selected_triangles[:, 0],
                                           selected_triangles[:, 2]-selected_triangles[:, 0])
                        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-20)
                        quats = Rotation.from_matrix(frame(normals)).as_quat(scalar_first=True)
                        tangent = float(np.median(np.exp(model.log_ref[:, :2].detach().cpu().numpy())))
                        log_scales = np.tile(np.log([tangent, tangent, .02*tangent]), (len(chosen), 1))
                        nearest = neighbours.ids[torch.as_tensor(residual['start'] + chosen, device='cuda'), 0]
                        roots = model.roots[nearest].to(torch.int64)
                        start_row, added = model.add(points[torch.as_tensor(residual['start'] + chosen)],
                                                     torch.as_tensor(log_scales, device='cuda', dtype=torch.float32),
                                                     torch.as_tensor(quats, device='cuda', dtype=torch.float32),
                                                     torch.as_tensor(residual['colours'][chosen], device='cuda'),
                                                     torch.full((len(chosen),), math.log(.99/.01), device='cuda'),
                                                     torch.as_tensor(faces, device='cuda'), roots, optimizer)
                        residual['used'][chosen] = True
                        neighbours.rebuild(model, step)
                        residual_events.append(dict(step=step, candidates=int(len(candidate_rows)),
                                                    selected=int(added), start=int(start_row),
                                                    support_threshold=args.residual_support_threshold))
                        splitter = ScreenGradientSplits(model)
                        print('residual_add', json.dumps(residual_events[-1]), flush=True)
        if step == 1 or step % 25 == 0 or step == args.steps:
            entry = dict(step=step, loss=float(loss.detach()), render=float(photometric.detach()), l1=float(l1.detach()),
                         camera=cameras[index]['name'],
                         ssim=float(ssim.detach()), gaussians=len(model.position),
                         neighbour_age=None if neighbours is None else step-neighbours.step,
                         terms={k: float(v.detach()) for k, v in terms.items()}, validation=validation(),
                         seconds=time.perf_counter()-tick)
            history.append(entry)
            print(json.dumps(entry), flush=True)
        del image, info, loss, photometric, terms, l1, ssim
        if args.checkpoint_every and (step % args.checkpoint_every == 0 or step == args.steps):
            checkpoint_dir = args.output_dir/'checkpoints'/f'step{step:05d}'
            checkpoint_dir.mkdir(parents=True)
            checkpoint = save_model(model, data, origin, checkpoint_dir)
            checkpoint.update(step=step, appearance=dict(color_domain=(args.material_prediction_space
                              if args.reference_mode == 'material' else 'display_linear')))
            (checkpoint_dir/'refinement_report.json').write_text(json.dumps(checkpoint, indent=2)+'\n')
            metrics = []
            with torch.no_grad():
                for i, camera in enumerate(cameras):
                    prediction, alpha, info = render_training_view(model, camera, matrices[i], K[i], args, rasterization, step)
                    objective, mae, similarity = render_loss(prediction, targets[i], args.dssim_weight)
                    row = dict(camera=camera['name'], loss=float(objective), mae=float(mae),
                               ssim=float(similarity), psnr_db=float(-10*torch.log10(
                               (prediction-targets[i]).square().mean().clamp(min=1e-12))))
                    if args.depth_weight:
                        mask = depth_masks[i]
                        error = (info['expected_depth']-target_depths[i]).abs()
                        threshold = torch.maximum(torch.full_like(error, .03), .01*target_depths[i])
                        row.update(depth_mae=float(error[mask].mean()), depth_consistent_coverage=float(
                            ((error <= threshold) & (alpha >= .95))[mask].float().mean()))
                    metrics.append(row)
                    del prediction, alpha, info
            summary = {key: float(np.mean([row[key] for row in metrics]))
                       for key in metrics[0] if key != 'camera'}
            (checkpoint_dir/'training_metrics.json').write_text(json.dumps(dict(step=step,
                    summary=summary, views=metrics), indent=2)+'\n')
            print('checkpoint', json.dumps(dict(step=step, **summary)), flush=True)
    optimization_seconds = time.perf_counter()-tick
    if neighbours is not None:
        neighbours.rebuild(model, args.steps)
    final_validation = validation()
    audit = geometry_audit(model, mesh)
    if voxel_record is not None:
        voxel_record['final_audit'] = audit_voxel_bounds(model)
        ids = model.roots.cpu().numpy()//voxel_record['samples_per_initial_voxel']
        np.save(args.output_dir/'voxel_ids.npy', ids.astype(np.int32))
        (args.output_dir/'voxel_constraints.json').write_text(json.dumps(voxel_record, indent=2)+'\n')
    saved = save_model(model, data, origin, args.output_dir)
    result = dict(status='complete',
                  method='joint_mesh_render_dynamic_gaussians', **saved,
                  blend=report['source'], source_sha256=report['input_sha256'],
                  coverage_status='not certified',
                  gaussian_count_limit=maximum_count, split_policy='all threshold-eligible parents' if args.uncapped_splits else 'legacy fraction/growth caps',
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  initializer_method=report.get('method'),
                  input_model=str(args.model_dir/'model.ply'), input_sha256=digest(args.model_dir/'model.ply'),
                  origin=origin.tolist(), initial_count=initial_count, splits=splits,
                  neighbours=[] if neighbours is None else neighbours.history,
                  neighbour_audit=None if neighbours is None else neighbours.audit(model, pool_count),
                  initial_validation=initial_validation,
                  initial_render=initial_render_summary,
                  final_validation=final_validation, initial_geometry=initial_geometry, final_geometry=audit,
                  query_cache=str(cache), query_pool_per_class=pool_count, query_validation_per_class=validation_count,
                  query_verification=None if query_report is None else query_report['empty_label_verification'], history=history,
                  residual_surface_additions=residual_events,
                  opacity_resets=opacity_resets, pruning=prune_events,
                  appearance=dict(sh_degree=args.sh_degree, sh_schedule='progressive' if args.sh_warmup_steps else 'full',
                                  reference_mode=args.reference_mode,
                                  color_domain=(args.material_prediction_space
                                                if args.reference_mode == 'material' else 'display_linear')),
                  reference_lighting=(MATERIAL_LIGHTING_PROFILE
                                      if args.reference_mode == 'material' else None),
                  render_display_transform=prediction_display_transform(args),
                  render_exposure_stops=(MATERIAL_EXPOSURE_STOPS
                                         if prediction_display_transform(args) == 'aces_fitted' else 0.),
                  reference_display_transform=(MATERIAL_DISPLAY_TRANSFORM
                                               if args.reference_mode == 'material' else 'standard'),
                  reference_exposure_stops=(MATERIAL_EXPOSURE_STOPS
                                            if args.reference_mode == 'material' else 0.),
                  view_sampling='priority', split_schedule='interleaved',
                  sampling_policy='Online loss/detail priorities',
                  view_statistics=[dict(camera=c['name'], updates=int(priorities.counts[i]),
                      detail=details[i], initial_render_loss=float(priorities.initial_losses[i]),
                      final_loss_ema=float(priorities.losses[i]), probability=float(priorities.probabilities()[i]))
                      for i, c in enumerate(cameras)],
                  preparation_seconds=prepared_seconds, optimization_seconds=optimization_seconds,
                  total_seconds=time.perf_counter()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  gpu=torch.cuda.get_device_name(0),
                  render_objective='(1-0.2)*L1 + 0.2*(1-SSIM); full material-rendered sRGB image after linear white compositing',
                  mesh_objective=('Original-triangle centre projection; optional opaque image-depth stabilization'
                                  if args.bind_source_faces else
                                  'Only explicitly nonzero Mesh support, centre/depth/alpha and BaseColor prior weights apply'),
                  update_stages=dict(geometry_freeze_steps=args.geometry_freeze_steps,
                                     opacity_freeze_steps=args.opacity_freeze_steps),
                  mesh_objective_configuration=dict(surface_query_mode=args.surface_query_mode,
                      mesh_weight=args.mesh_weight, empty_weight=args.empty_weight,
                      surface_basecolor_weight=args.mesh_colour_weight,
                      centre_weight=args.centre_weight, initial_rgb_prior_weight=args.uv_weight,
                      geometry_loss_steps=args.geometry_loss_steps,
                      color_prior_domain='source linear BaseColor; not PBR radiance'),
                  image_geometry_objective=dict(depth_weight=args.depth_weight, alpha_weight=args.alpha_weight,
                      depth_mask=('Verified opaque geometric first hit and matching Cycles Z; excludes edges'
                                  if args.material_semantics else
                                  'Valid Cycles Z interior; excludes depth/silhouette edges; transparency not disambiguated'),
                      alpha_loss='Class-balanced binary cross entropy'),
                  normal_mode='fixed inherited normal; in-plane rotation',
                  voxel_constraint=voxel_record,
                  material_adaptation=material_record,
                  footprint_adaptation=footprint_record,
                  appearance_adaptation=baked_record,
                  source_face_binding=args.bind_source_faces,
                  displacement_radius_reference=('Initial ancestor sigma; inherited unchanged through splits'
                                                 if args.bind_source_faces else 'Current split scale reference'),
                  scale_ceiling_reference=('Initial ancestor tangent sigma; inherited unchanged through splits'
                                           if args.bind_source_faces else 'Current split scale reference'),
                  rule=f'Positive tangent scales; SH degree {args.sh_degree}; normal thickness 2% of minor; stage-controlled geometry/opacity updates; only enabled split/reset/prune events apply',
                  limitations=['Finite periodically rebuilt neighbours approximate the spatial support field',
                               'Surface parameterization and bounded motion/scales differ from unrestricted upstream 3DGS',
                               'Pruning protects unseen/frozen rows but is still heuristic; two-child replacement uses screen-gradient parent selection',
                               'Optimization completion does not certify continuous surface coverage'])
    (args.output_dir/'refinement_report.json').write_text(json.dumps(result, indent=2)+'\n')
    root = Path(__file__).resolve().parents[1]
    if args.config:
        shutil.copyfile(args.config, args.output_dir/'training_config.json')
    for name in ('scripts/refine_joint.py', 'scripts/render_gaussians.py', 'mesh_to_gs/area_sampling.py',
                 'mesh_to_gs/joint_refinement.py', 'mesh_to_gs/mesh_targets.py', 'mesh_to_gs/material_semantics.py',
                 'scripts/prepare_material_geometry.py', 'mesh_to_gs/appearance_prior.py', 'mesh_to_gs/footprint_constraints.py', 'mesh_to_gs/view_sampling.py',
                 'mesh_to_gs/mesh_supervision.py', 'mesh_to_gs/color.py', 'mesh_to_gs/voxel_constraints.py',
                 'mesh_to_gs/repair_geometry.py', 'mesh_to_gs/mesh_reference.py',
                 'configs/pbr_v10_area_surface_bound.json', 'configs/material_lighting_v10.json', 'uv.lock'):
        target = args.output_dir/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    print(json.dumps(saved), flush=True)


if __name__ == '__main__':
    main()

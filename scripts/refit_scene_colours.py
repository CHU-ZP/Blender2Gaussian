"""Select opaque splats from repeated training residuals and validate a colour-only update."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData
import torch

from mesh_to_gs.color import composite_display_rgb, MATERIAL_EXPOSURE_STOPS, srgb_to_linear
from mesh_to_gs.mesh_supervision import digest
if __package__:
    from .render_gaussians import centered_view_matrix, configure_cuda
else:
    from render_gaussians import centered_view_matrix, configure_cuda

C0 = .28209479177387814


def accept_refit(base, candidate):
    return (base['local_pixels'] > 0 and candidate['local_pixels'] == base['local_pixels']
            and candidate['local_mae'] < base['local_mae']
            and candidate['mae'] < base['mae'] and candidate['mse'] <= base['mse'])


class Scene:
    def __init__(self, root):
        configure_cuda()
        from gsplat import rasterization
        self.rasterize = rasterization
        self.references = root/'references'
        self.material = root/'material'
        path = root/'base/model.ply'
        rows = PlyData.read(path, mmap='r')['vertex'].data
        def columns(prefix, count):
            return torch.from_numpy(np.column_stack([rows[f'{prefix}{i}'] for i in range(count)])).cuda()
        xyz = np.column_stack([rows[k] for k in 'xyz']).astype(np.float64)
        self.origin = (xyz.min(0) + xyz.max(0))/2
        self.means = torch.from_numpy((xyz-self.origin).astype(np.float32)).cuda()
        self.quats = torch.nn.functional.normalize(columns('rot_', 4), dim=-1)
        self.scales = columns('scale_', 3).exp()
        self.opacity = torch.from_numpy(np.array(rows['opacity'])).cuda().sigmoid()
        self.srgb = (columns('f_dc_', 3)*C0+.5).clamp(0, 1)
        self.linear = srgb_to_linear(self.srgb)
        roots = np.load(root/'base/lineage.npy', mmap_mode='r')
        self.opaque = torch.from_numpy(np.load(self.material/'row_alpha.npy')[roots] >= .99).cuda()
        self.frames = {}
        for split in ('training', 'validation'):
            data = json.loads((self.references/f'{split}_cameras.json').read_text())
            self.width, self.height = data['width'], data['height']
            self.frames[split] = data['cameras']
        self.targets = {}

    def reference(self, camera, need_depth=False):
        from mesh_to_gs.mesh_reference import read_exr
        name = camera['name']
        if not need_depth and name in self.targets:
            return self.targets[name].cuda(), None
        rgb, alpha, depth = read_exr(self.references/'training_references'/f'{name}.exr')
        target = composite_display_rgb(rgb, alpha[..., None], display_transform='aces_fitted',
                                       exposure_stops=MATERIAL_EXPOSURE_STOPS)
        if not np.isfinite(target).all():
            raise ValueError(f'Nonfinite reference {name}')
        self.targets[name] = torch.from_numpy(target)
        return self.targets[name].cuda(), torch.from_numpy(depth).cuda() if need_depth else None

    def render(self, camera, colours, depth=False):
        view = centered_view_matrix(camera['view_matrix'], self.origin)
        rgb, alpha, info = self.rasterize(self.means, self.quats, self.scales, self.opacity, colours,
              torch.tensor(view, dtype=torch.float32, device='cuda')[None],
              torch.tensor(camera['K'], dtype=torch.float32, device='cuda')[None],
              self.width, self.height, near_plane=camera['near'], far_plane=camera['far'],
              eps2d=.05, packed=True, rasterize_mode='classic', render_mode='RGB+ED' if depth else 'RGB')
        image = composite_display_rgb(rgb[0, ..., :3], alpha[0])
        return image, rgb[0, ..., 3:], alpha[0, ..., 0], info

    @torch.no_grad()
    def select(self):
        votes = torch.zeros(len(self.means), dtype=torch.int32, device='cuda')
        for camera in self.frames['training']:
            target, z = self.reference(camera, need_depth=True)
            image, expected, alpha, info = self.render(camera, self.linear, depth=True)
            interior = torch.from_numpy(np.load(self.material/'opaque_depth_masks'/f'{camera["name"]}.npy')).cuda()
            tolerance = (.01*z).clamp(min=.03)
            reliable = interior & (alpha >= .95) & ((expected[..., 0]-z).abs() <= tolerance)
            error = (image-target).abs().mean(-1)
            if not reliable.any():
                continue
            # Ignore sub-quantization residuals and prioritize the worst quartile per view.
            threshold = torch.quantile(error[reliable], .75).clamp(min=4/255)
            xy = info['means2d'].floor().long()
            x, y = xy.unbind(-1)
            inside = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
            ids = info['gaussian_ids'][inside].long()
            x, y = x[inside], y[inside]
            good = reliable[y,x] & (error[y,x] >= threshold) & self.opaque[ids]
            good &= (info['depths'][inside]-z[y,x]).abs() <= tolerance[y,x]
            votes[ids[good]] += 1
        return torch.where(votes >= 2)[0]

    @torch.no_grad()
    def scores(self, colours, marker):
        total_error = total_squared = local_error = 0.
        pixels = local_pixels = 0
        marked = torch.cat((colours, marker), dim=1)
        for camera in self.frames['validation']:
            target, _ = self.reference(camera)
            image, weight, _, _ = self.render(camera, marked)
            error = (image-target).abs()
            local = weight[..., 0] >= .05
            pixels += error.numel()//3
            local_pixels += int(local.sum())
            total_error += float(error.double().sum())
            total_squared += float(error.double().square().sum())
            local_error += float(error[local].double().sum())
        return dict(mae=total_error/(3*pixels), mse=total_squared/(3*pixels), local_pixels=local_pixels,
                    local_mae=local_error/(3*local_pixels) if local_pixels else None)


def refit(root, output, passes):
    scene = Scene(root)
    selected = scene.select()
    marker = torch.zeros_like(scene.opacity[:, None])
    marker[selected] = 1
    base = scene.scores(scene.linear, marker)
    before = root/'base/model.ply'
    report = dict(model='base/model.ply', reason='no repeated residuals', selected_rows=len(selected),
                  base_validation=base, candidate_validation=None, passes=passes)
    if len(selected) and not base['local_pixels']:
        report['reason'] = 'no validation coverage of selected splats'
    if len(selected) and base['local_pixels']:
        values = torch.nn.Parameter(scene.srgb[selected].clone())
        optimizer = torch.optim.Adam([values], lr=.004)
        marked = torch.cat((scene.linear, marker), dim=1)
        rng = np.random.default_rng(json.loads((root/'training_config.json').read_text())['seed'])
        frames = scene.frames['training']
        for _ in range(passes):
            for index in rng.permutation(len(frames)):
                camera = frames[index]
                target, _ = scene.reference(camera)
                colours = marked.index_copy(0, selected, torch.cat((srgb_to_linear(values), marker[selected]), 1))
                image, weights, _, _ = scene.render(camera, colours)
                mask = weights[..., 0].detach() >= .05
                if not mask.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = (image[mask]-target[mask]).abs().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite colour fit')
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    values.clamp_(0, 1)
        # Score exactly the float32 DC coefficients which will be written to PLY.
        dc = (values.detach()-.5)/C0
        fitted = scene.linear.index_copy(0, selected, srgb_to_linear((dc*C0+.5).clamp(0, 1)))
        candidate = scene.scores(fitted, marker)
        report.update(candidate_validation=candidate, reason='validation did not improve')
        if accept_refit(base, candidate):
            path = output/'model.ply'
            shutil.copyfile(before, path)
            with path.open('r+b') as stream:
                rows = PlyData.read(stream, mmap='r+')['vertex'].data
                ids, coeff = selected.cpu().numpy(), dc.cpu().numpy()
                for i in range(3):
                    rows[f'f_dc_{i}'][ids] = coeff[:, i]
                rows.flush()
            report.update(model=str(path.relative_to(root)), reason='validation accepted colour refit')
    report['model_sha256'] = digest(root/report['model'])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--passes', type=int, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = refit(args.root, args.output, args.passes)
    (args.output/'decision.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

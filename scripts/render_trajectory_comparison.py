"""Render synchronized GT Mesh and Gaussian frames, then encode a comparison video."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mesh_to_gs.mesh_supervision import digest
from mesh_to_gs.trajectories import keyframe_indices


def run_render(command, log, root):
    print('Rendering; log:', log, flush=True)
    with log.open('w') as stream:
        subprocess.run(command, check=True, cwd=root, stdout=stream, stderr=subprocess.STDOUT)


def preview(model_dir, folder, source, data, samples, root):
    """Check twelve paired views before spending time on the full trajectory."""
    probe = folder/'preview_probe'
    probe.mkdir()
    count = len(data['cameras'])
    indices = (np.arange(12)*count//12 if data['closed'] else np.linspace(0, count-1, 12).round().astype(int))
    selected = [data['cameras'][i] for i in indices]
    camera_file = probe/'cameras.json'
    camera_file.write_text(json.dumps(dict(data, cameras=selected), indent=2)+'\n')
    run_render([sys.executable, str(root/'scripts/render_gaussians.py'), str(model_dir/'model.ply'),
                '--cameras', str(camera_file), '--output-dir', str(probe/'renders'), '--no-contact-sheet'],
               probe/'gaussian_render.log', root)
    run_render([sys.executable, str(root/'scripts/render_geometry_reference.py'), '--blend', str(source),
                '--cameras', str(camera_file), '--output-dir', str(probe/'mesh'),
                '--samples', str(samples), '--rgb-only'], probe/'mesh_render.log', root)
    tile_width = 960
    tile_height = round((data['height']+32)*tile_width/(2*data['width']))
    sheet = Image.new('RGB', (2*tile_width, 6*tile_height), 'white')
    for tile, (index, camera) in enumerate(zip(indices, selected)):
        name = camera['name']+'.png'
        image = comparison_frame(probe/'mesh'/name, probe/'renders'/name, data['width'], data['height'],
                                 f'{index/data["fps"]:.2f} s')
        sheet.paste(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS),
                    ((tile % 2)*tile_width, (tile//2)*tile_height))
    sheet.save(probe/'preview.jpg', quality=95)
    print('Saved paired preview:', probe/'preview.jpg', flush=True)


def comparison_frame(gt_path, gaussian_path, width, height, caption):
    """Keep both renders at their original pixel resolution, with labels above."""
    frame = Image.new('RGB', (2*width, height+32), '#17212b')
    font = ImageFont.load_default(size=20)
    draw = ImageDraw.Draw(frame)
    draw.text((12, 5), 'GT Mesh | '+caption, font=font, fill='white')
    draw.text((width+12, 5), 'Gaussian | '+caption, font=font, fill='white')
    for path, x in ((gt_path, 0), (gaussian_path, width)):
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f'Unexpected image size: {path}: {image.size}')
            frame.paste(image.convert('RGB'), (x, 32))
    return frame


def compose(folder, data, sampled, ffmpeg):
    names = [camera['name'] for camera in data['cameras']]
    gt = json.loads((folder/'mesh/reference_report.json').read_text())
    gaussians = json.loads((folder/'renders/render_report.json').read_text())
    if gt['source_sha256'] != sampled['source_sha256'] or gt['cameras_sha256'] != sampled['cameras_sha256']:
        raise ValueError('GT reference source or cameras differ from the sampled trajectory')
    if gaussians['model_sha256'] != sampled['model_sha256']:
        raise ValueError('Gaussian renders belong to a different model')
    gaussian_video = json.loads((folder/'video_report.json').read_text())
    if gaussian_video['cameras_sha256'] != sampled['cameras_sha256']:
        raise ValueError('Gaussian renders use different cameras')
    for report in (gt, gaussians):
        if [view['camera'] for view in report['views']] != names:
            raise ValueError('The Mesh and Gaussian frame sequences must match exactly')
        if (report['width'], report['height']) != (data['width'], data['height']):
            raise ValueError('The Mesh and Gaussian image dimensions must match')
    max_error = max(v['projection_error_pixels'] for v in gt['views'])
    if max_error > .001:
        raise ValueError('Mesh camera projection mismatch')
    width, height, fps = data['width'], data['height'], data['fps']
    output_size = (2*width, height+32)
    video = folder/'comparison.mp4'
    partial = folder/'comparison.partial.mp4'
    if video.exists() or partial.exists():
        raise ValueError('Comparison video already exists')
    selected = keyframe_indices(len(names), data['closed']).tolist()
    thumb_width = 960
    thumb_height = round(output_size[1]*thumb_width/output_size[0])
    sheet = Image.new('RGB', (2*thumb_width, 3*thumb_height), 'white')
    process = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{output_size[0]}x{output_size[1]}',
        '-framerate', str(fps), '-i', 'pipe:0', '-frames:v', str(len(names)), '-an',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-threads', '4', '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE)
    try:
        for index, name in enumerate(names):
            frame = comparison_frame(folder/'mesh'/f'{name}.png', folder/'renders'/f'{name}.png',
                                     width, height, f'{index/fps:.2f} s')
            process.stdin.write(frame.tobytes())
            if index in selected:
                for tile, selected_index in enumerate(selected):
                    if selected_index == index:
                        sheet.paste(frame.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS),
                                    ((tile % 2)*thumb_width, (tile//2)*thumb_height))
            if index == len(names)//2:
                frame.save(folder/'comparison_poster.jpg', quality=95)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('FFmpeg failed to encode the comparison video')
    except BaseException:
        process.kill()
        process.wait()
        raise
    partial.rename(video)
    sheet.save(folder/'comparison_preview.jpg', quality=95)
    report = dict(status='complete', frames=len(names), fps=fps,
        duration_seconds=len(names)/fps, width=output_size[0], height=output_size[1],
        panel_width=width, panel_height=height, label_height=32, layout=['GT Mesh', 'Gaussian'],
        model_sha256=sampled['model_sha256'], source_sha256=sampled['source_sha256'],
        cameras_sha256=sampled['cameras_sha256'], video_sha256=digest(video),
        maximum_projection_error_pixels=max_error, mesh_samples=gt['samples'],
        mesh_device=gt['device'], color='Unlit opaque BaseColor; linear white compositing; sRGB output',
        codec='H.264 / yuv420p', crf=18, keyframe_indices=selected,
        source_blend_saved=False, existing_model_retrained=False)
    (folder/'comparison_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def gallery(folder, scene, data):
    title = html.escape(scene)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · GT Mesh / Gaussian</title><style>body{{font:16px system-ui;margin:24px auto;max-width:1600px;padding:0 16px;background:#f7f8fa;color:#17212b}}video,img{{width:100%;border-radius:6px}}input{{width:100%}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.map{{max-width:1200px}}a{{color:#1264bf}}</style>
<h1>{title} 轨迹对比</h1><p>{data['duration_seconds']:g} 秒 · {data['fps']} fps · 每侧 {data['width']}×{data['height']} · 左 GT Mesh，右 Gaussian</p>
<video controls preload="metadata" poster="comparison_poster.jpg" src="comparison.mp4"></video>
<p><a href="comparison.mp4">下载对比视频</a> · <a href="trajectory.mp4">高斯视频</a> · <a href="cameras.json">相机轨迹</a></p>
<h2>同步逐帧对比</h2><input id="frame" type="range" min="0" max="{len(data['cameras'])-1}" value="0"><p id="time">0.00 s</p>
<div class="pair"><div><p>GT Mesh</p><a id="gt_link" href="mesh/frame_000000.png"><img id="gt" src="mesh/frame_000000.png"></a></div>
<div><p>Gaussian</p><a id="gs_link" href="renders/frame_000000.png"><img id="gs" src="renders/frame_000000.png"></a></div></div>
<h2>关键帧</h2><img src="comparison_preview.jpg"><h2>移动路线</h2><img class="map" src="path.png">
<script>const slider=document.getElementById('frame');slider.oninput=()=>{{const n=Number(slider.value);const name='frame_'+String(n).padStart(6,'0')+'.png';for(const [id,dir] of [['gt','mesh'],['gs','renders']]){{document.getElementById(id).src=dir+'/'+name;document.getElementById(id+'_link').href=dir+'/'+name;}}document.getElementById('time').textContent=(n/{data['fps']}).toFixed(2)+' s';}};</script></html>'''
    (folder/'gallery.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('trajectory_dir', type=Path)
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--compose-only', action='store_true', help='Use completed, matching frame reports')
    parser.add_argument('--preview-only', action='store_true', help='Render twelve paired views for a composition check')
    parser.add_argument('--gpu', type=int, help='CUDA device index inherited by both renderers')
    args = parser.parse_args()
    ffmpeg = str(args.ffmpeg.resolve()) if args.ffmpeg else shutil.which('ffmpeg')
    if not ffmpeg or not Path(ffmpeg).is_file() or args.samples < 1:
        parser.error('FFmpeg and positive Mesh samples are required')
    if args.preview_only and args.compose_only:
        parser.error('Choose either preview-only or compose-only')
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_dir, folder = args.model_dir.resolve(), args.trajectory_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    camera_file = folder/'cameras.json'
    data = json.loads(camera_file.read_text())
    sampled = json.loads((folder/'trajectory_report.json').read_text())
    if digest(model_dir/'model.ply') != sampled['model_sha256'] or digest(camera_file) != sampled['cameras_sha256']:
        parser.error('Model or cameras differ from the audited trajectory')
    source = Path(sampled['source_blend'])
    if digest(source) != sampled['source_sha256']:
        parser.error('Source Mesh differs from the audited trajectory')
    if args.preview_only:
        preview(model_dir, folder, source, data, args.samples, root)
        return
    if not args.compose_only:
        if (folder/'mesh').exists() or (folder/'renders').exists():
            parser.error('Use a new trajectory directory, or --compose-only for complete renders')
        run_render([sys.executable, str(root/'scripts/render_trajectory.py'), str(model_dir),
                    str(folder), '--ffmpeg', ffmpeg], folder/'gaussian_render.log', root)
        run_render([sys.executable, str(root/'scripts/render_geometry_reference.py'),
                    '--blend', str(source), '--cameras', str(camera_file), '--output-dir', str(folder/'mesh'),
                    '--samples', str(args.samples), '--rgb-only'], folder/'mesh_render.log', root)
    report = compose(folder, data, sampled, ffmpeg)
    gallery(folder, model_dir.name, data)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print('Saved comparison:', folder/'comparison.mp4', flush=True)


if __name__ == '__main__':
    main()

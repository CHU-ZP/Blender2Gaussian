"""Render sampled cameras with the retained model and encode an H.264 trajectory video."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

from mesh_to_gs.mesh_supervision import digest
from mesh_to_gs.trajectories import keyframe_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('trajectory_dir', type=Path)
    parser.add_argument('--ffmpeg', type=Path, help='FFmpeg executable with the libx264 encoder')
    args = parser.parse_args()
    ffmpeg = str(args.ffmpeg) if args.ffmpeg else shutil.which('ffmpeg')
    if not ffmpeg or not Path(ffmpeg).is_file():
        parser.error('FFmpeg is required; provide --ffmpeg /path/to/ffmpeg')
    root = Path(__file__).resolve().parents[1]
    camera_file = args.trajectory_dir/'cameras.json'
    data = json.loads(camera_file.read_text())
    sampled = json.loads((args.trajectory_dir/'trajectory_report.json').read_text())
    model = args.model_dir/'model.ply'
    if digest(model) != sampled['model_sha256']:
        parser.error('Sampled trajectory belongs to a different retained model')
    if digest(camera_file) != sampled['cameras_sha256']:
        parser.error('Camera file differs from the audited trajectory')
    names = [c['name'] for c in data['cameras']]
    if names != [f'frame_{i:06d}' for i in range(len(names))] or data['width'] % 2 or data['height'] % 2:
        parser.error('Contiguous frame names and even video dimensions are required')
    renders = args.trajectory_dir/'renders'
    if renders.exists():
        parser.error('Existing trajectory renders are never overwritten')
    subprocess.run([sys.executable, str(root/'scripts/render_gaussians.py'), str(model),
                    '--cameras', str(camera_file), '--output-dir', str(renders), '--no-contact-sheet'], check=True, cwd=root)
    video = args.trajectory_dir/'trajectory.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                    '-framerate', str(data['fps']), '-start_number', '0', '-i', str(renders/'frame_%06d.png'),
                    '-frames:v', str(len(names)), '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart', str(video)], check=True)
    frames = keyframe_indices(len(names), data['closed'])
    sheet = Image.new('RGB', (3*480, 2*(270+28)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        with Image.open(renders/f'{names[frame]}.png') as image:
            sheet.paste(image.convert('RGB').resize((480, 270), Image.Resampling.LANCZOS), ((i % 3)*480, (i//3)*298))
        draw.text(((i % 3)*480+8, (i//3)*298+276), f'{i+1} | {frame/data["fps"]:.2f} s', fill='black')
    sheet.save(args.trajectory_dir/'preview.jpg', quality=93)
    shutil.copyfile(renders/f'{names[len(names)//2]}.png', args.trajectory_dir/'poster.png')
    title = html.escape(args.model_dir.name)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} 轨迹渲染</title><style>body{{font:16px system-ui;margin:24px auto;max-width:1280px;padding:0 16px;background:#f7f8fa;color:#17212b}}video,img{{width:100%;border-radius:8px}}input{{width:100%}}.map{{max-width:850px}}a{{color:#1264bf}}</style>
<h1>{title} 轨迹渲染</h1><p>{data['duration_seconds']:g} 秒 · {data['fps']} fps · {data['width']}×{data['height']} · 原 best 模型</p>
<video controls preload="metadata" poster="poster.png" src="trajectory.mp4"></video>
<p><a href="trajectory.mp4">下载视频</a> · <a href="cameras.json">相机轨迹</a></p><h2>逐帧查看</h2>
<input id="frame" type="range" min="0" max="{len(names)-1}" value="0"><p id="time">0.00 s</p><img id="image" src="renders/frame_000000.png">
<h2>关键帧</h2><img src="preview.jpg"><h2>移动路线</h2><img class="map" src="path.png">
<script>const slider=document.getElementById('frame');slider.oninput=()=>{{const n=Number(slider.value);document.getElementById('image').src='renders/frame_'+String(n).padStart(6,'0')+'.png';document.getElementById('time').textContent=(n/{data['fps']}).toFixed(2)+' s';}};</script></html>'''
    (args.trajectory_dir/'gallery.html').write_text(page)
    report = json.loads((renders/'render_report.json').read_text())
    assert report['model_sha256'] == sampled['model_sha256'] and len(report['views']) == len(names)
    video_report = dict(status='complete', model_sha256=report['model_sha256'], cameras_sha256=digest(camera_file),
                        video_sha256=digest(video), frames=len(names), fps=data['fps'], width=data['width'], height=data['height'],
                        duration_seconds=data['duration_seconds'], codec='H.264 / yuv420p', crf=18,
                        encoder=subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
                        renderer='gsplat; original full model; linear white compositing; classic; eps2d=.05')
    (args.trajectory_dir/'video_report.json').write_text(json.dumps(video_report, indent=2)+'\n')
    print('Saved trajectory video:', video, flush=True)


if __name__ == '__main__':
    main()

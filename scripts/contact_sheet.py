"""Side-by-side contact sheet: Mesh reference and Gaussian renders, one row per camera."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', type=Path, required=True)
    parser.add_argument('--column', nargs=2, action='append', metavar=('LABEL', 'DIRECTORY'), required=True,
                        help='Image directory with <camera>.png; repeat for each column (first: reference)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()
    cameras = json.loads(args.cameras.read_text())
    height = round(args.width * cameras['height'] / cameras['width'])
    label = 28
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    sheet = Image.new('RGB', (args.width * len(args.column), (height + label) * len(cameras['cameras'])), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, camera in enumerate(cameras['cameras']):
        for col, (name, directory) in enumerate(args.column):
            with Image.open(Path(directory) / f"{camera['name']}.png") as image:
                sheet.paste(image.convert('RGB').resize((args.width, height), Image.Resampling.LANCZOS),
                            (col * args.width, row * (height + label)))
            draw.text((col * args.width + 8, row * (height + label) + height + 5), f"{camera['name']} | {name}",
                      font=font, fill='black')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == '__main__':
    main()

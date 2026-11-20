"""Build the current area-sampled Mesh-to-Gaussian initializer."""
import argparse
import json
from pathlib import Path

from mesh_to_gs.area_sampling import initialize_area_sampled


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    allocation = parser.add_mutually_exclusive_group()
    allocation.add_argument('--count', type=int)
    allocation.add_argument('--spacing', type=float, help='Target area spacing in world units')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sigma-factor', type=float)
    parser.add_argument('--opacity', type=float)
    parser.add_argument('--chunk-size', type=int)
    config = Path(__file__).resolve().parents[1]/'configs/area_sampling_initialization.json'
    defaults = json.loads(config.read_text())
    default_spacing = defaults.pop('spacing')
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.count is None and args.spacing is None:
        args.spacing = default_spacing
    return args


def main():
    args = parse_args()
    report = initialize_area_sampled(args.input, args.output_dir, total=args.count, spacing=args.spacing,
        seed=args.seed, sigma_factor=args.sigma_factor, opacity=args.opacity,
        chunk_size=args.chunk_size)
    print(report['timings_seconds'], flush=True)


if __name__ == '__main__':
    main()

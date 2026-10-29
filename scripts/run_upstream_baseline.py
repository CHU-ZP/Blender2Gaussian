"""Build untouched upstream algorithm sources, then adapt a .blend to its inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

from render_gaussians import configure_cuda


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="45f867e")
    parser.add_argument("--blend", type=Path, default=Path("data/sk_00918.blend"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sk_00918_upstream_baseline"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, build = output / "source", output / "build"
    commit = subprocess.check_output(["git", "rev-parse", args.commit], cwd=root, text=True).strip()
    if not source.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(source), commit], cwd=root, check=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == commit
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "csrc", "mesh_to_gs", "mesh2gs.py", "mesh2gs_python.py"],
        cwd=root, text=True).splitlines()
    fingerprints = {}
    for name in paths:
        expected = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        actual = (source / name).read_bytes()
        assert actual == expected, name
        fingerprints[name] = {"sha256": hashlib.sha256(actual).hexdigest(),
                              "current_workspace_matches_upstream": (root / name).read_bytes() == expected}
    print(f"Verified untouched baseline {commit}: {len(paths)} algorithm files", flush=True)
    configure_cuda()
    import torch
    from torch.utils.cpp_extension import load

    build.mkdir(exist_ok=True)
    tick = time.perf_counter()
    extension = load(name="_C", sources=[str(source / "csrc/bindings.cpp"), str(source / "csrc/mesh_to_gs_kernel.cu")],
                     extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], build_directory=str(build), verbose=True)
    sys.modules["_C"] = extension
    sys.path.insert(0, str(source))
    import mesh_to_gs

    assert Path(mesh_to_gs.__file__).resolve().is_relative_to(source)
    provenance = {
        "commit": commit, "algorithm_sources": fingerprints, "source_worktree": str(source),
        "source_worktree_clean": True, "extension": str(extension.__file__),
        "extension_sha256": digest(Path(extension.__file__)), "build_seconds": time.perf_counter() - tick,
        "build_adaptation": "PyTorch JIT chooses the ABI matching torch; upstream algorithm sources are untouched",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python_converter": str(mesh_to_gs.__file__), "input_blend": str(args.blend.resolve()),
        "input_sha256": digest(args.blend.resolve()), "native_convert_signature": extension.convert.__doc__,
        "input_adaptation": "Direct bpy geometry, material batches and Base Color; no GLB intermediate",
        "coordinate_restoration": "upstream-cli: restore centers only, retaining the original CLI scale behavior",
        "kernel_fixes_applied": False, "diagnostic_scale_cap_applied": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    sys.argv = ["validate_blend.py", str(args.blend.resolve()), "--output-dir", str(output),
                "--coordinate-restoration", "upstream-cli"]
    runpy.run_path(str(root / "scripts/validate_blend.py"), run_name="__main__")


if __name__ == "__main__":
    main()

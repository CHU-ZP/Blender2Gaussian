"""Validate the existing CUDA converter against meshes and Base Color in a .blend scene."""

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import resource
import shutil
import time

import bpy
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

from mesh_to_gs import mesh_to_gs_cuda


def material_description(material):
    if material is None:
        return {"name": "<no material>", "color": [0.5, 0.5, 0.5], "image": None,
                "alpha": 1.0, "alpha_linked": False}
    nodes = material.node_tree.nodes if material.node_tree else []
    outputs = [n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    if len(outputs) != 1 or len(outputs[0].inputs["Surface"].links) != 1:
        raise ValueError(f"Unsupported surface output: {material.name}")
    bsdf = outputs[0].inputs["Surface"].links[0].from_node
    if bsdf.type != "BSDF_PRINCIPLED":
        raise ValueError(f"Surface is not a direct Principled BSDF: {material.name}")
    base = bsdf.inputs["Base Color"]
    image = None
    if base.is_linked:
        node = base.links[0].from_node
        if node.type != "TEX_IMAGE" or node.image is None or node.inputs["Vector"].is_linked:
            raise ValueError(f"Unsupported Base Color graph: {material.name}")
        if node.extension != "REPEAT" or node.interpolation != "Linear":
            raise ValueError(f"Unsupported texture filtering/wrapping: {material.name}")
        image = node.image
    # Constants in Principled sockets are scene-linear; image bytes follow the
    # same RGB texture convention as the existing GLB conversion entry point.
    linear = np.asarray(base.default_value[:3], dtype=np.float64)
    color = np.where(linear <= 0.0031308, linear * 12.92,
                     1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)
    return {"name": material.name, "color": np.clip(color, 0, 1).tolist(), "image": image,
            "alpha": float(bsdf.inputs["Alpha"].default_value),
            "alpha_linked": bool(bsdf.inputs["Alpha"].is_linked)}


def load_texture(image):
    if image.packed_file:
        source = BytesIO(image.packed_file.data)
    else:
        source = bpy.path.abspath(image.filepath, library=image.library)
    with Image.open(source) as bitmap:
        return np.asarray(bitmap.convert("RGB"), dtype=np.uint8)


def extract_scene(surface_groups=False):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if any(instance.is_instance for instance in depsgraph.object_instances):
        raise ValueError("This validation adapter does not expand collection/geometry instances")
    groups = {}
    vertex_sets = []
    objects = []
    modifiers = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            if not mesh.vertices or not mesh.polygons:
                continue
            mesh.calc_loop_triangles()
            vertices = np.empty((len(mesh.vertices), 3), np.float64)
            mesh.vertices.foreach_get("co", vertices.ravel())
            matrix = np.array(evaluated.matrix_world, dtype=np.float64)
            world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            vertex_sets.append(world)
            triangles = np.empty((len(mesh.loop_triangles), 3), np.int32)
            mesh.loop_triangles.foreach_get("vertices", triangles.ravel())
            material_indices = np.empty(len(triangles), np.int32)
            mesh.loop_triangles.foreach_get("material_index", material_indices)
            loops = np.empty_like(triangles)
            mesh.loop_triangles.foreach_get("loops", loops.ravel())
            uv = None
            if mesh.uv_layers.active:
                uv = np.empty((len(mesh.loops), 2), np.float32)
                mesh.uv_layers.active.data.foreach_get("uv", uv.ravel())
            for index in np.unique(material_indices):
                material = obj.material_slots[index].material if index < len(obj.material_slots) else None
                material_key = material.as_pointer() if material else 0
                key = (len(objects), material_key) if surface_groups else material_key
                if key not in groups:
                    groups[key] = {"description": material_description(material), "triangles": [], "uvs": []}
                    if surface_groups:
                        groups[key].update(object_name=obj.name, object_index=len(objects), vertex_ids=[])
                group = groups[key]
                selected = material_indices == index
                group["triangles"].append(world[triangles[selected]])
                if surface_groups:
                    group["vertex_ids"].append(triangles[selected])
                if group["description"]["image"] is not None:
                    if uv is None:
                        raise ValueError(f"Missing UV map on textured object: {obj.name}")
                    group["uvs"].append(uv[loops[selected]])
            objects.append({"name": obj.name, "vertices": len(world), "triangles": len(triangles)})
            modifiers += len(obj.modifiers)
        finally:
            evaluated.to_mesh_clear()
    if not vertex_sets:
        raise ValueError("No render-enabled mesh geometry found")
    vertex_count = sum(len(v) for v in vertex_sets)
    center = sum((v.sum(axis=0) for v in vertex_sets), start=np.zeros(3)) / vertex_count
    radius = max(float(np.linalg.norm(v - center, axis=1).max()) for v in vertex_sets)
    if radius <= 1e-6:
        raise ValueError("Scene extent is too small")
    bounds = [np.min([v.min(0) for v in vertex_sets], axis=0).tolist(),
              np.max([v.max(0) for v in vertex_sets], axis=0).tolist()]
    return groups, center, radius, bounds, objects, modifiers


def batches(triangles, resolution, density, face_limit, gaussian_limit):
    normalized = np.asarray(triangles, dtype=np.float32)
    areas = np.linalg.norm(np.cross(normalized[:, 1] - normalized[:, 0],
                                    normalized[:, 2] - normalized[:, 0]), axis=1) * 0.5
    counts = np.ceil(density * areas.astype(np.float64) * resolution ** 2).astype(np.int64)
    counts[areas < 1e-12] = 0
    cumulative = np.concatenate(([0], np.cumsum(counts)))
    start = 0
    while start < len(triangles):
        stop = min(start + face_limit, len(triangles))
        stop = min(stop, int(np.searchsorted(cumulative, cumulative[start] + gaussian_limit, side="right")) - 1)
        if stop <= start:
            raise ValueError("One triangle exceeds the Gaussian batch budget; reduce --resolution")
        yield start, stop, int(cumulative[stop] - cumulative[start]), int(np.count_nonzero(counts[start:stop] == 0))
        start = stop


def save_preview(points, colors, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), layout="constrained")
    for axis, (a, b, title) in zip(axes, [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        axis.scatter(points[:, a], points[:, b], c=np.clip(colors, 0, 1), s=0.2, linewidths=0, rasterized=True)
        axis.set(title=title, xlabel="XYZ"[a], ylabel="XYZ"[b], aspect="equal")
    fig.suptitle("sk_00918: sampled Gaussian centers and Base Color (point preview)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sk_00918_validation"))
    parser.add_argument("--resolution", "-R", type=int, default=512)
    parser.add_argument("--density", "-k", type=float, default=2.0)
    parser.add_argument("--faces-per-batch", type=int, default=100_000)
    parser.add_argument("--gaussians-per-batch", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, help="Optional native RNG seed for matched comparisons; sampling method is unchanged")
    parser.add_argument("--coordinate-restoration", choices=("positions-and-scales", "upstream-cli"),
                        default="positions-and-scales",
                        help="upstream-cli reproduces mesh2gs.py: restore centers but leave scales normalized")
    args = parser.parse_args()
    if min(args.resolution, args.density, args.faces_per_batch, args.gaussians_per_batch) <= 0:
        parser.error("Resolution, density and batch budgets must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()), load_ui=False)
    loaded = time.perf_counter()
    groups, center, radius, bounds, objects, modifiers = extract_scene()
    extracted = time.perf_counter()
    report = {
        "source": str(args.input), "status": "running", "resolution": args.resolution, "density": args.density,
        "coordinate_restoration": args.coordinate_restoration,
        "seed": args.seed,
        "blender": bpy.app.version_string, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "mesh_objects": len(objects), "vertices": sum(o["vertices"] for o in objects),
        "triangles": sum(o["triangles"] for o in objects), "modifiers": modifiers,
        "source_bounds": bounds, "normalization_center": center.tolist(), "normalization_radius": radius,
        "materials": [], "gaussians": 0, "batches": 0, "skipped_faces_estimate": 0,
        "non_unit_quaternions": 0, "quaternion_unit_tolerance": 1e-3,
        "load_seconds": loaded - started, "extract_seconds": extracted - loaded,
        "scope": "All render-enabled mesh objects in the active scene, evaluated geometry, world transforms, face-corner UVs and Base Color; no GLB intermediate",
        "limitations": ["Existing converter uses opacity 0.99; Principled Alpha is recorded but not transferred",
                        "Roughness, metallic, normal maps, lighting and higher SH are not transferred",
                        "CUDA uses unit-normal local-to-world wxyz quaternions and independent thread RNG subsequences",
                        ("World restoration applies the normalization radius to both positions and Gaussian scales"
                         if args.coordinate_restoration == "positions-and-scales" else
                         "Reproduces upstream CLI: only centers are restored; scales remain normalized")],
    }
    print(f"Loaded {len(objects)} meshes, {report['triangles']:,} triangles, {len(groups)} materials", flush=True)
    print(f"GPU: {report['gpu']}; global radius: {radius:.6f}; R={args.resolution}", flush=True)
    attributes = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    body_path = args.output_dir / "model.ply.body"
    model_path = args.output_dir / "model.ply"
    rng = np.random.default_rng(20261028)
    sample_keys = np.empty(0)
    sample_points = np.empty((0, 3), np.float32)
    sample_colors = np.empty((0, 3), np.float32)
    conversion_seconds = 0.0
    point_bounds = [np.full(3, np.inf), np.full(3, -np.inf)]
    quaternion_norm_range = [np.inf, -np.inf]
    torch.cuda.reset_peak_memory_stats()
    with body_path.open("wb") as stream:
        for group_number, group in enumerate(groups.values(), 1):
            description = group["description"]
            triangles = np.concatenate(group.pop("triangles"))
            triangles -= center
            triangles /= radius
            uvs = np.concatenate(group.pop("uvs")) if description["image"] is not None else None
            texture = load_texture(description["image"]) if description["image"] is not None else None
            material_report = {"name": description["name"], "triangles": len(triangles), "gaussians": 0,
                               "textured": texture is not None, "image": description["image"].name if texture is not None else None,
                               "source_alpha": description["alpha"], "source_alpha_linked": description["alpha_linked"]}
            for start, stop, predicted, skipped in batches(triangles, args.resolution, args.density,
                                                           args.faces_per_batch, args.gaussians_per_batch):
                report["skipped_faces_estimate"] += skipped
                if predicted == 0:
                    continue
                vertices = np.asarray(triangles[start:stop].reshape(-1, 3), dtype=np.float32)
                faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
                kwargs = {"uvs": np.ascontiguousarray(uvs[start:stop].reshape(-1, 2)), "texture": texture} if texture is not None else {
                    "vertex_colors": np.broadcast_to(np.asarray(description["color"], np.float32), vertices.shape).copy()}
                if args.seed is not None:
                    kwargs["seed"] = args.seed
                tick = time.perf_counter()
                model = mesh_to_gs_cuda(vertices, faces, R=args.resolution, k=args.density, **kwargs)
                conversion_seconds += time.perf_counter() - tick
                count = len(model.get_xyz)
                values = np.zeros((count, len(attributes)), dtype="<f4")
                values[:, :3] = model.get_xyz.detach().cpu().numpy().astype(np.float64) * radius + center
                values[:, 6:9] = model._features_dc.detach().cpu().numpy().reshape(count, 3)
                values[:, 9] = model._opacity.detach().cpu().numpy()
                scale_offset = math.log(radius) if args.coordinate_restoration == "positions-and-scales" else 0.0
                values[:, 10:13] = model._scaling.detach().cpu().numpy() + scale_offset
                values[:, 13:17] = model._rotation.detach().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite output in material {description['name']}")
                norms = np.linalg.norm(values[:, 13:17], axis=1)
                if np.any(norms <= 1e-8):
                    raise ValueError(f"Zero quaternion in material {description['name']}")
                report["non_unit_quaternions"] += int(np.count_nonzero(np.abs(norms - 1) > report["quaternion_unit_tolerance"]))
                quaternion_norm_range = [min(quaternion_norm_range[0], float(norms.min())),
                                         max(quaternion_norm_range[1], float(norms.max()))]
                point_bounds[0] = np.minimum(point_bounds[0], values[:, :3].min(0))
                point_bounds[1] = np.maximum(point_bounds[1], values[:, :3].max(0))
                values.tofile(stream)
                keys = rng.random(count)
                keep = min(count, 120_000)
                selected = np.argpartition(keys, keep - 1)[:keep]
                sample_keys = np.concatenate((sample_keys, keys[selected]))
                sample_points = np.concatenate((sample_points, values[selected, :3]))
                sample_colors = np.concatenate((sample_colors, values[selected, 6:9] * 0.28209479177387814 + 0.5))
                if len(sample_keys) > 120_000:
                    selected = np.argpartition(sample_keys, 119_999)[:120_000]
                    sample_keys, sample_points, sample_colors = sample_keys[selected], sample_points[selected], sample_colors[selected]
                report["gaussians"] += count
                report["batches"] += 1
                material_report["gaussians"] += count
                del model, values
            report["materials"].append(material_report)
            if group_number % 20 == 0 or group_number == len(groups):
                print(f"Materials {group_number}/{len(groups)}; Gaussians {report['gaussians']:,}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
        if not report["gaussians"]:
            raise ValueError("No Gaussians produced")
    header = "ply\nformat binary_little_endian 1.0\nelement vertex " + str(report["gaussians"]) + "\n"
    header += "".join(f"property float {name}\n" for name in attributes) + "end_header\n"
    temporary_ply = model_path.with_suffix(".ply.tmp")
    with temporary_ply.open("wb") as output, body_path.open("rb") as stream:
        output.write(header.encode("ascii"))
        shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
    temporary_ply.replace(model_path)
    body_path.unlink()
    ply = PlyData.read(model_path, mmap="r")
    if len(ply["vertex"]) != report["gaussians"]:
        raise ValueError("PLY vertex count mismatch")
    if model_path.stat().st_size != len(header.encode("ascii")) + report["gaussians"] * 68:
        raise ValueError("PLY byte size mismatch")
    for start in range(0, report["gaussians"], 100_000):
        part = ply["vertex"].data[start:start + 100_000]
        for attribute in attributes:
            if not np.isfinite(part[attribute]).all():
                raise ValueError(f"Non-finite PLY field {attribute}")
    if np.any(point_bounds[0] < np.asarray(bounds[0]) - 1e-3) or np.any(point_bounds[1] > np.asarray(bounds[1]) + 1e-3):
        raise ValueError("Gaussian centers fall outside source bounds")
    save_preview(sample_points, sample_colors, args.output_dir / "preview.png")
    report.update(output=str(model_path), output_bytes=model_path.stat().st_size,
                  gaussian_bounds=[v.tolist() for v in point_bounds], quaternion_norm_range=quaternion_norm_range,
                  conversion_seconds=conversion_seconds, total_seconds=time.perf_counter() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  textured_materials=sum(m["textured"] for m in report["materials"]),
                  alpha_materials=sum(m["source_alpha_linked"] or m["source_alpha"] < 1 for m in report["materials"]),
                  objects=objects)
    report["numeric_data_path_checks"] = "passed"
    report["quaternion_unit_check"] = "passed" if report["non_unit_quaternions"] == 0 else "failed"
    report["status"] = "passed_with_limitations" if report["non_unit_quaternions"] or report["alpha_materials"] else "passed"
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "mesh_objects", "triangles", "gaussians", "textured_materials", "alpha_materials", "non_unit_quaternions", "output", "total_seconds", "peak_cuda_allocated_bytes"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

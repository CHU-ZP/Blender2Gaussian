"""Measure converter geometry errors using analytic triangles and the .blend mesh."""

import argparse
import json
from pathlib import Path

import bpy
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from mesh_to_gs import mesh_to_gs_cuda
from validate_blend import extract_scene


def describe(values):
    return dict(zip(("min", "p50", "p90", "p99", "max"),
                    map(float, np.percentile(values, [0, 50, 90, 99, 100]))))


def synthetic_checks():
    triangle = np.array([[0, 0, 0], [0.01, 0, 0], [0.003, 0.008, 0]], dtype=np.float32)
    rotation = Rotation.from_euler("xyz", [0.3, 0.6, 1.1]).as_matrix()
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    checks = []
    for label, shrink in [("ordinary_triangle", 1.0), ("same_shape_1000_times_smaller", 0.001)]:
        vertices = np.asarray(triangle @ rotation.T * shrink, dtype=np.float32)
        model = mesh_to_gs_cuda(vertices, faces, R=512, k=2)
        raw_q = model._rotation.detach().cpu().numpy()[0]
        local_to_world = Rotation.from_quat(raw_q, scalar_first=True).as_matrix()
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]).astype(np.float64)
        area = np.linalg.norm(normal) * 0.5
        normal /= np.linalg.norm(normal)
        points = model.get_xyz.detach().cpu().numpy()
        tangent_basis = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
        uv = np.linalg.lstsq(tangent_basis, (points - vertices[0]).T, rcond=None)[0].T
        barycentric = np.column_stack((1 - uv.sum(1), uv))
        longest_edge = max(np.linalg.norm(vertices[1] - vertices[0]), np.linalg.norm(vertices[2] - vertices[1]),
                           np.linalg.norm(vertices[0] - vertices[2]))
        entry = {"case": label, "area": float(area), "longest_edge": float(longest_edge),
                 "expected_continuous_count": float(2 * area * 512 ** 2), "actual_count": len(points),
                 "scales": model.get_scaling.detach().cpu().numpy()[0].tolist(),
                 "raw_quaternion_norm": float(np.linalg.norm(raw_q)),
                 "normal_axis_error_degrees": float(np.degrees(np.arccos(np.clip(abs(local_to_world[:, 2] @ normal), 0, 1)))),
                 "normal_axis_error_after_conjugation_degrees": float(np.degrees(np.arccos(np.clip(abs(local_to_world[2] @ normal), 0, 1)))),
                 "min_sample_barycentric": float(barycentric.min()),
                 "max_sample_distance_to_plane": float(np.abs((points - vertices[0]) @ normal).max())}
        checks.append(entry)
        assert barycentric.min() > -1e-5
        assert entry["max_sample_distance_to_plane"] < 1e-7
    # Uniform barycentric sampling has mean (1/3,1/3,1/3).
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    model = mesh_to_gs_cuda(vertices, faces, R=256, k=2)
    points = model.get_xyz.detach().cpu().numpy()
    barycentric = np.column_stack((1 - points[:, :2].sum(1), points[:, :2]))
    _, repetitions = np.unique(points, axis=0, return_counts=True)
    sampling = {"samples": len(points), "unique_centers": len(repetitions),
                "max_identical_center_repetitions": int(repetitions.max()),
                "unique_center_fraction": float(len(repetitions) / len(points)),
                "center_2_equals_center_256": bool(np.array_equal(points[2], points[256])),
                "min_barycentric": float(barycentric.min()),
                "mean_barycentric": barycentric.mean(0).tolist(), "expected_mean": [1 / 3] * 3}
    assert barycentric.min() > -1e-6
    return {"triangles": checks, "barycentric_sampling": sampling}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, default=Path("data/sk_00918.blend"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sk_00918_validation/diagnosis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    synthetic = synthetic_checks()
    print("Analytic geometry checks:", json.dumps(synthetic, indent=2), flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()), load_ui=False)
    groups, center, radius, _, _, _ = extract_scene()
    totals = {"triangles": 0, "skipped_triangles": 0, "predicted_gaussians": 0,
              "triangles_with_zeroed_normal": 0, "predicted_gaussians_with_zeroed_normal": 0,
              "one_gaussian_triangles": 0, "gaussians_sigma_long_over_0_5m": 0,
              "gaussians_long_sigma_over_longest_edge": 0}
    examples = []
    continuous_counts, actual_counts, face_areas, long_scales, longest_edges = [], [], [], [], []
    for group in groups.values():
        triangles = np.asarray((np.concatenate(group["triangles"]) - center) / radius, dtype=np.float32)
        edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 1],
                          triangles[:, 0] - triangles[:, 2]), axis=1)
        L = np.linalg.norm(edges, axis=2).max(1)
        cross_length = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        area = 0.5 * cross_length
        mu = 2 * area * 512 ** 2
        count = np.ceil(mu).astype(np.int64)
        count[area < 1e-12] = 0
        valid = count > 0
        zero_normal = valid & (cross_length <= 0)
        inv_sqrt_count = 1 / np.sqrt(np.maximum(count, 1))
        support1 = .5 * L * inv_sqrt_count
        sigma = np.minimum(np.maximum(support1 / np.sqrt(np.pi), 1 / 2048),
                           np.minimum(1 / 8, support1)) * radius
        totals["triangles"] += len(triangles)
        totals["skipped_triangles"] += int((~valid).sum())
        totals["predicted_gaussians"] += int(count.sum())
        totals["triangles_with_zeroed_normal"] += int(zero_normal.sum())
        totals["predicted_gaussians_with_zeroed_normal"] += int(count[zero_normal].sum())
        totals["one_gaussian_triangles"] += int((count == 1).sum())
        totals["gaussians_sigma_long_over_0_5m"] += int(count[sigma > .5].sum())
        totals["gaussians_long_sigma_over_longest_edge"] += int(count[sigma > L * radius].sum())
        selected = np.flatnonzero(valid & (sigma > L * radius))
        selected = selected[np.argsort((sigma / np.maximum(L * radius, 1e-20))[selected])[-3:]]
        for i in selected:
            examples.append({"material": group["description"]["name"], "world_area": float(area[i] * radius ** 2),
                             "world_longest_edge": float(L[i] * radius), "sigma_long_world": float(sigma[i]),
                             "sigma_to_longest_edge": float(sigma[i] / (L[i] * radius)), "count": int(count[i]),
                             "continuous_count": float(mu[i]), "normal_zeroed": bool(zero_normal[i])})
        continuous_counts.append(mu[valid])
        actual_counts.append(count[valid])
        face_areas.append(area[valid] * radius ** 2)
        long_scales.append(sigma[valid])
        longest_edges.append(L[valid] * radius)
    mu, count = np.concatenate(continuous_counts), np.concatenate(actual_counts)
    world_areas, sigmas, lengths = map(np.concatenate, (face_areas, long_scales, longest_edges))
    loops, remainder = count // 256, count % 256
    unique_upper_bound = np.where(loops > 0, 2 * loops + np.maximum(254, remainder), count)
    report = {"synthetic": synthetic, "scene": totals, "normalization_radius": radius,
              "scale_model": "Actual face count; minimum scale bounded by face dimensions / sqrt(count); relative normal thickness",
              "blender_unit_system": bpy.context.scene.unit_settings.system,
              "blender_unit_scale_length": bpy.context.scene.unit_settings.scale_length,
              "reconstruction_note": "Per-face counts/scales reconstructed in NumPy float32; FMA and ceil may differ slightly from CUDA",
              "one_gaussian_fraction_of_valid_faces": float((count == 1).mean()),
              "allocation_inflation_due_to_ceil": float(count.sum() / mu.sum()),
              "legacy_rng_overlap_unique_centers_upper_bound": int(unique_upper_bound.sum()),
              "current_sampling": "Independent thread subsequences and distinct triangular strata; see synthetic unique-center count",
              "largest_face_gaussian_count": int(count.max()),
              "per_face_count_inflation_quantiles": describe(count / mu),
              "one_gaussian_count_inflation_quantiles": describe((count / mu)[count == 1]),
              "world_area_quantiles": describe(world_areas), "sigma_over_edge_quantiles": describe(sigmas / lengths),
              "most_oversized_examples": sorted(examples, key=lambda entry: entry["sigma_to_longest_edge"], reverse=True)[:10]}
    path = args.output_dir / "geometry_checks.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("Scene geometry checks:", json.dumps({key: value for key, value in report.items() if key != "synthetic"}, ensure_ascii=False, indent=2), flush=True)
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()

"""Export four interior viewpoints for the sk_00918 scene in OpenCV convention."""

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector
import numpy as np


def look_at(eye, target):
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = eye
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()), load_ui=False)
    scene = bpy.context.scene
    original = scene.camera
    if original is None or original.data.type != "PERSP":
        raise ValueError("Expected a perspective camera in the source scene")
    opencv_from_blender = np.diag([1, -1, -1, 1])
    original_c2w = np.array(original.matrix_world) @ opencv_from_blender
    original_fx = args.width * original.data.lens / original.data.sensor_width
    views = [("01_original_camera", "Original interior camera", original_c2w, original_fx)]
    focal = args.width / (2 * math.tan(math.radians(80) / 2))
    for name, label, eye, target in [
        ("02_lounge_corner", "Lounge towards the main room", [121.0, 137.7, 1.6], [112.5, 132.0, 1.4]),
        ("03_opposite_corner", "Opposite corner across the room", [104.0, 137.5, 1.6], [116.0, 131.2, 1.4]),
        ("04_seating_area", "Main room towards the seating area", [121.0, 129.3, 1.6], [113.8, 135.8, 1.25]),
    ]:
        views.append((name, label, look_at(eye, target), focal))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {"source_blend": str(args.input), "convention": "OpenCV: camera +X right, +Y down, +Z forward",
              "width": args.width, "height": args.height, "cameras": []}
    for name, label, c2w, fx in views:
        close_surfaces = []
        origin = Vector(c2w[:3, 3])
        for direction in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, Vector(direction), distance=0.2)
            if hit:
                close_surfaces.append({"object": obj.name, "distance": float((location - origin).length)})
        entry = {"name": name, "label": label, "camera_to_world": c2w.tolist(),
                 "view_matrix": np.linalg.inv(c2w).tolist(),
                 "K": [[fx, 0, args.width / 2], [0, fx, args.height / 2], [0, 0, 1]],
                 "near": 0.05, "far": 250.0, "surfaces_within_20cm": close_surfaces}
        result["cameras"].append(entry)
        print(f"Camera {name}: eye={c2w[:3, 3].round(3).tolist()}, nearby surfaces={len(close_surfaces)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

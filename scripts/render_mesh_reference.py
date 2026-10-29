"""Render the Blender mesh at the saved Gaussian cameras, with comparison sheets."""

import argparse
import json
from pathlib import Path
import time

import bpy
from mathutils import Matrix
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def configure_device(scene):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            selected = []
            for device in prefs.devices:
                device.use = device.type == backend
                if device.use:
                    selected.append(device.name)
            if selected:
                scene.cycles.device = "GPU"
                return {"backend": backend, "devices": selected}
        except (TypeError, RuntimeError):
            continue
    scene.cycles.device = "CPU"
    return {"backend": "CPU", "devices": ["CPU"]}


def configure_camera(scene, camera, entry, width, height):
    K = np.array(entry["K"], dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera.matrix_world = Matrix((np.array(entry["camera_to_world"]) @ np.diag([1, -1, -1, 1])).tolist())
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = fx / fy
    data = camera.data
    data.type, data.sensor_fit, data.sensor_width = "PERSP", "HORIZONTAL", 36
    data.lens = fx * data.sensor_width / width
    data.shift_x = (width / 2 - cx) / width
    data.shift_y = (cy - height / 2) * (fx / fy) / width
    data.clip_start, data.clip_end = entry["near"], entry["far"]
    data.dof.use_dof = False
    bpy.context.view_layer.update()
    # Check the Blender projection against the saved OpenCV intrinsics.
    projection = np.array(camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(),
                          x=width, y=height, scale_x=1, scale_y=fx / fy))
    cv = np.array([[0, 0, 3, 1], [.5, .3, 3, 1], [-.6, -.4, 4, 1]], dtype=float)
    clip = cv @ np.diag([1, -1, -1, 1]) @ projection.T
    ndc = clip[:, :2] / clip[:, 3:4]
    pixels = np.column_stack(((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2))
    expected = cv[:, :3] @ K.T
    expected = expected[:, :2] / expected[:, 2:3]
    error = float(np.max(np.abs(pixels - expected)))
    if error > .001:
        raise ValueError(f"Camera projection mismatch: {error} pixels")
    return error


def base_color_materials():
    changes = []
    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        tree = material.node_tree
        outputs = [node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output]
        if len(outputs) != 1 or len(outputs[0].inputs["Surface"].links) != 1:
            raise ValueError(f"Unsupported output: {material.name}")
        output = outputs[0]
        original = output.inputs["Surface"].links[0].from_socket
        bsdf = original.node
        if bsdf.type != "BSDF_PRINCIPLED":
            raise ValueError(f"Expected Principled BSDF: {material.name}")
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
        emission.inputs["Strength"].default_value = 1
        if bsdf.inputs["Base Color"].is_linked:
            tree.links.new(bsdf.inputs["Base Color"].links[0].from_socket, emission.inputs["Color"])
        tree.links.new(emission.outputs[0], output.inputs["Surface"])
        changes.append((tree, output, original, emission))
    return changes


def add_preview_lights(scene):
    lights = []
    for index, location in enumerate([
        (105.5, 136.2, 3.1), (111.4, 136.2, 3.1), (117.8, 136.2, 3.1),
        (108.5, 129.5, 3.1), (115.0, 129.5, 3.1), (122.0, 131.8, 3.1),
    ]):
        data = bpy.data.lights.new(f"Mesh preview area {index + 1}", "AREA")
        data.energy, data.shape, data.size = 500, "DISK", 3
        light = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(light)
        light.location = location  # Default -Z emission points down into the room.
        lights.append({"name": data.name, "type": "AREA", "power_watts": data.energy,
                       "diameter": data.size, "location": list(location), "direction": [0, 0, -1]})
    return lights


def make_sheets(output, cameras, gaussian_dir):
    thumb = (800, 450)
    row_height = 486
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for mode in ("base_color", "principled"):
        sheet = Image.new("RGB", (1600, row_height * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for i, camera in enumerate(cameras):
            x, y = (i % 2) * 800, (i // 2) * row_height
            with Image.open(output / mode / (camera["name"] + ".png")) as src:
                sheet.paste(src.convert("RGB").resize(thumb, Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 10, y + 458), f"{camera['name']} | Mesh {mode}", font=font, fill="black")
        sheet.save(output / (mode + "_contact_sheet.png"))
    comparison = Image.new("RGB", (1600, row_height * len(cameras)), "white")
    draw = ImageDraw.Draw(comparison)
    for row, camera in enumerate(cameras):
        for col, (path, label) in enumerate([
            (output / "base_color", "Mesh Base Color (opaque)"),
            (gaussian_dir, "Gaussian (scale fixes)"),
        ]):
            with Image.open(path / (camera["name"] + ".png")) as src:
                comparison.paste(src.convert("RGB").resize(thumb, Image.Resampling.LANCZOS), (col * 800, row * row_height))
            draw.text((col * 800 + 10, row * row_height + 458), f"{camera['name']} | {label}", font=font, fill="black")
    comparison.save(output / "mesh_vs_gaussian.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, default=Path("data/sk_00918.blend"))
    parser.add_argument("--cameras", type=Path, default=Path("outputs/sk_00918_validation/renders/cameras.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sk_00918_mesh_reference"))
    parser.add_argument("--gaussian-dir", type=Path, default=Path("outputs/sk_00918_scale_fixes/renders"))
    parser.add_argument("--base-samples", type=int, default=16)
    parser.add_argument("--material-samples", type=int, default=64)
    args = parser.parse_args()
    for mode in ("base_color", "principled"):
        (args.output_dir / mode).mkdir(parents=True, exist_ok=True)
    cameras = json.loads(args.cameras.read_text())
    width, height = cameras["width"], cameras["height"]
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()), load_ui=False)
    scene = bpy.context.scene
    source_settings = {"engine": scene.render.engine, "view_transform": scene.view_settings.view_transform,
                       "light_objects": [obj.name for obj in scene.objects if obj.type == "LIGHT"],
                       "world_background": [{"color": list(node.inputs["Color"].default_value),
                                             "strength": node.inputs["Strength"].default_value}
                                            for node in scene.world.node_tree.nodes if node.type == "BACKGROUND"]}
    scene.render.engine = "CYCLES"
    device = configure_device(scene)
    print("Cycles device:", json.dumps(device), flush=True)
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, height, 100
    scene.render.use_compositing = scene.render.use_sequencer = False
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_persistent_data = True
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = 20261029
    scene.cycles.max_bounces, scene.cycles.transparent_max_bounces = 8, 16
    scene.view_settings.look = "None"
    scene.view_settings.exposure, scene.view_settings.gamma = 0, 1
    world = bpy.data.worlds.new("Mesh reference neutral white world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1
    scene.world = world
    data = bpy.data.cameras.new("Gaussian comparison camera")
    camera = bpy.data.objects.new("Gaussian comparison camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    changes = base_color_materials()
    results = []
    preview_lights = []
    for mode in ("base_color", "principled"):
        if mode == "principled":
            for tree, output, original, emission in changes:
                tree.links.new(original, output.inputs["Surface"])
                tree.nodes.remove(emission)
            preview_lights = add_preview_lights(scene)
        scene.view_settings.view_transform = "Standard" if mode == "base_color" else "AgX"
        scene.cycles.samples = args.base_samples if mode == "base_color" else args.material_samples
        scene.cycles.use_denoising = mode == "principled"
        for entry in cameras["cameras"]:
            error = configure_camera(scene, camera, entry, width, height)
            path = (args.output_dir / mode / (entry["name"] + ".png")).resolve()
            scene.render.filepath = str(path)
            print(f"Rendering Mesh {mode}: {entry['name']} ({scene.cycles.samples} samples)", flush=True)
            tick = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
                if image.size != (width, height) or pixels.std() < 1:
                    raise ValueError(f"Empty or unexpected image: {path}")
            results.append({"mode": mode, "camera": entry["name"], "image": str(path),
                            "seconds": time.perf_counter() - tick, "intrinsics_max_error_pixels": error})
            print(json.dumps(results[-1]), flush=True)
    make_sheets(args.output_dir, cameras["cameras"], args.gaussian_dir)
    report = {"source_blend": str(args.blend), "blender": bpy.app.version_string,
              "source_settings": source_settings, "device": device, "engine": "CYCLES",
              "width": width, "height": height, "cameras_file": str(args.cameras),
              "mesh_objects": sum(obj.type == "MESH" and not obj.hide_render for obj in scene.objects),
              "base_color": {"samples": args.base_samples, "view_transform": "Standard", "denoising": False,
                             "appearance": "Base Color via unit emission; opaque, matching Gaussian alpha limitation; no shading"},
              "principled": {"samples": args.material_samples, "view_transform": "AgX", "denoising": True,
                             "appearance": "Original Principled materials including Alpha; neutral uniform white environment, strength 1; six added interior area lights",
                             "added_preview_lights": preview_lights},
              "source_blend_saved": False, "views": results}
    (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved eight Mesh renders and comparison sheets to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()

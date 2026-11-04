"""Read evaluated world-space Mesh geometry and supported Principled BaseColor."""
from io import BytesIO

import bpy
import numpy as np
from PIL import Image


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


def extract_scene():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if any(instance.is_instance for instance in depsgraph.object_instances):
        raise ValueError("This validation adapter does not expand collection/geometry instances")
    groups = {}
    object_bounds = []
    objects = []
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
            object_bounds.append((world.min(0),world.max(0)))
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
                key = (len(objects), material_key)
                if key not in groups:
                    groups[key] = {"description": material_description(material), "triangles": [], "uvs": []}
                    groups[key].update(object_name=obj.name, object_index=len(objects), vertex_ids=[],
                                       loop_triangles=[])
                group = groups[key]
                selected = material_indices == index
                group["triangles"].append(world[triangles[selected]])
                group["vertex_ids"].append(triangles[selected])
                group["loop_triangles"].append(np.flatnonzero(selected).astype(np.int32))
                if group["description"]["image"] is not None:
                    if uv is None:
                        raise ValueError(f"Missing UV map on textured object: {obj.name}")
                    group["uvs"].append(uv[loops[selected]])
            objects.append({"name": obj.name, "vertices": len(world), "triangles": len(triangles)})
        finally:
            evaluated.to_mesh_clear()
    if not object_bounds:
        raise ValueError("No render-enabled mesh geometry found")
    bounds = [np.min([b[0] for b in object_bounds],axis=0).tolist(),
              np.max([b[1] for b in object_bounds],axis=0).tolist()]
    return groups, bounds, objects


"""Convert an assembled glTF scene into a .blend the Blender surface initializer accepts.

The initializer supports a direct Principled BSDF with a constant or a single
Repeat/Linear image Base Color. Vertex-colour Base Color inputs (unsupported)
are replaced by their mean vertex colour, a constant; every replacement is
listed in the JSON report. Geometry and placements are not modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    replaced = []
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED' or not node.inputs['Base Color'].is_linked:
                continue
            source = node.inputs['Base Color'].links[0].from_node
            if source.type != 'VERTEX_COLOR':
                continue
            colours = []
            for obj in bpy.data.objects:
                if obj.type == 'MESH' and any(s.material == material for s in obj.material_slots):
                    layer = obj.data.color_attributes.get(source.layer_name) or obj.data.color_attributes.active_color
                    if layer is not None:
                        values = np.empty(len(layer.data) * 4, np.float32)
                        layer.data.foreach_get('color', values)
                        colours.append(values.reshape(-1, 4)[:, :3])
            mean = np.concatenate(colours).mean(0) if colours else np.full(3, .5)
            material.node_tree.links.remove(node.inputs['Base Color'].links[0])
            node.inputs['Base Color'].default_value = (*mean.tolist(), 1.)
            replaced.append(dict(material=material.name, layer=source.layer_name, scene_linear_rgb=mean.tolist()))
    for image in bpy.data.images:
        if image.source == 'FILE' and not image.packed_file:
            image.pack()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()), compress=False)
    with args.input.open('rb') as stream:
        source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = dict(source=str(args.input), source_sha256=source_sha, output=str(args.output),
                  mesh_objects=sum(o.type == 'MESH' for o in bpy.data.objects), replaced_vertex_colour=replaced,
                  note='glTF import is Y-up to Blender Z-up; units are the source units (not metrically verified)')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()

"""World-space Blender triangles, UVs and source textures used by area sampling."""
import json
from collections import OrderedDict

import numpy as np

from .surface import linear_texture


class ColourTextures:
    def __init__(self, loader, max_bytes=512*1024**2):
        self.loader, self.max_bytes = loader, max_bytes
        self.items, self.nbytes = OrderedDict(), 0

    def get(self, image):
        key = image.as_pointer()
        if key in self.items:
            value = self.items.pop(key)
        else:
            value = linear_texture(self.loader(image))
            self.nbytes += value.nbytes
        self.items[key] = value
        while self.nbytes > self.max_bytes and len(self.items) > 1:
            self.nbytes -= self.items.popitem(last=False)[1].nbytes
        return value


def extract_geometry(source, directory):
    import bpy
    from .blend import extract_scene
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
    raw, bounds, objects = extract_scene()
    directory.mkdir()
    chunks, uv_chunks, loop_chunks, groups, images = [], [], [], [], []
    offset = skipped = 0
    for index, group in enumerate(raw.values()):
        tri = np.concatenate(group.pop('triangles'))
        cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        area2 = np.hypot(np.hypot(cross[:, 0], cross[:, 1]), cross[:, 2])
        valid = np.isfinite(tri).all((1, 2)) & np.isfinite(area2) & (area2 > 0)
        skipped += int((~valid).sum())
        tri = tri[valid]
        desc = group['description']
        uv = np.concatenate(group.pop('uvs'))[valid] if group['uvs'] else np.zeros((len(tri), 3, 2), np.float32)
        chunks.append(tri); uv_chunks.append(uv)
        loop_chunks.append(np.concatenate(group.pop('loop_triangles'))[valid])
        groups.append(dict(object=group['object_name'], object_index=group['object_index'],
            material=desc['name'], color=desc['color'], image=None if desc['image'] is None else desc['image'].name,
            has_uv=desc['image'] is not None, alpha=desc['alpha'], alpha_linked=desc['alpha_linked'],
            faces=[offset, offset+len(tri)], charts=[index, index+1]))
        images.append(desc['image'])
        offset += len(tri)
    triangles, uv = np.concatenate(chunks), np.concatenate(uv_chunks)
    np.save(directory/'triangles.npy', triangles)
    np.save(directory/'uv.npy', uv)
    np.save(directory/'loop_triangle.npy', np.concatenate(loop_chunks))
    np.save(directory/'chart.npy', np.repeat(np.arange(len(groups), dtype=np.int32), [len(c) for c in chunks]))
    (directory/'groups.json').write_text(json.dumps(groups, ensure_ascii=False, indent=1)+'\n')
    return triangles, uv, groups, images, dict(source_bounds=bounds, mesh_objects=len(objects),
        triangles=sum(o['triangles'] for o in objects), retained_triangles=len(triangles),
        skipped_degenerate_faces=skipped, groups=len(groups))

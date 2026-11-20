"""Cache source alpha and training-only opaque first-hit depth masks.

The input PLY and original attachments remain unchanged. Transparent first
hits never supply depth targets, even if Cycles composites an opaque alpha.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from mesh_to_gs.mesh_supervision import digest, model_inputs

@wp.kernel
def _first_hits(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3),
                directions: wp.array(dtype=wp.vec3), camera_z: wp.array(dtype=float),
                alpha: wp.array(dtype=float), near: float, far: float,
                depth: wp.array(dtype=float), hit_face: wp.array(dtype=wp.int32),
                barycentric: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    query = wp.mesh_query_ray(mesh, origins[i], directions[i], far/camera_z[i])
    if query.result:
        z = query.t*camera_z[i]
        depth[i] = z
        if z > near:
            hit_face[i] = query.face
            barycentric[i] = wp.vec2(query.u, query.v)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('Use a new metadata directory')
    args.output_dir.mkdir(parents=True)
    tick = time.perf_counter()
    report, metadata = model_inputs(args.model_dir)
    if digest(report['source']) != report['input_sha256']:
        raise ValueError('Source scene changed')
    import bpy
    from mesh_to_gs.blend import material_description
    from mesh_to_gs.mesh_reference import read_exr
    from scipy.ndimage import binary_dilation, binary_erosion
    if __package__:
        from .compare_renders import contours
    else:
        from compare_renders import contours
    bpy.ops.wm.open_mainfile(filepath=report['source'], load_ui=False)
    groups = json.loads((metadata/'groups.json').read_text())
    triangles = np.load(metadata/'triangles.npy', mmap_mode='r')
    face_group = np.asarray(np.load(metadata/'chart.npy'), np.int32)
    descriptions, group_alpha, alpha_textures = {}, [], {}
    for group in groups:
        name = group['material']
        if name not in descriptions:
            material = None if name == '<no material>' else bpy.data.materials[name]
            desc = material_description(material)
            if desc['alpha_linked']:
                from io import BytesIO
                from PIL import Image
                nodes = [node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED']
                socket = nodes[0].inputs['Alpha'] if len(nodes) == 1 else None
                link = socket.links[0] if socket is not None and len(socket.links) == 1 else None
                if link is None or link.from_node.type != 'TEX_IMAGE' or link.from_socket.name != 'Alpha':
                    raise ValueError(f'Unsupported linked alpha graph: {name}')
                image = link.from_node.image
                source = BytesIO(image.packed_file.data) if image.packed_file else bpy.path.abspath(image.filepath, library=image.library)
                with Image.open(source) as bitmap:
                    extrema = bitmap.convert('RGBA').getchannel('A').getextrema()
                    if extrema[0] != extrema[1]:
                        node = link.from_node
                        if node.extension != 'REPEAT' or node.interpolation != 'Linear' or node.inputs['Vector'].is_linked:
                            raise ValueError(f'Unsupported alpha texture mapping/filter: {name}')
                        alpha_textures[name] = np.asarray(bitmap.convert('RGBA'))[..., 3].astype(np.float32)/255
                desc['alpha'] = extrema[0]/255 if extrema[0] == extrema[1] else 1.
            descriptions[name] = dict(alpha=desc['alpha'], alpha_linked=desc['alpha_linked'],
                                      linked_alpha_handling=('Bilinear UV texture alpha' if name in alpha_textures else
                                        'Verified constant image alpha') if desc['alpha_linked'] else None)
        group_alpha.append(descriptions[name]['alpha'])
    alpha = np.asarray(group_alpha, np.float32)[face_group]
    faces = np.load(metadata/'face.npy', mmap_mode='r')
    count = np.bincount(face_group[faces], minlength=len(groups))
    tri = np.asarray(triangles)
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1)/2
    group_area = np.bincount(face_group, weights=area, minlength=len(groups))
    density = count/np.maximum(group_area, 1e-30)
    uv = np.load(metadata/'uv.npy', mmap_mode='r')
    from mesh_to_gs.adaptive import bilinear
    def surface_alpha(selected_faces, bary):
        if bary.shape[1] != 3:
            raise ValueError('Expected all three barycentric weights')
        values = np.asarray(alpha[selected_faces]).copy()
        selected_groups = face_group[selected_faces]
        for group_index in np.unique(selected_groups):
            name = groups[int(group_index)]['material']
            if name in alpha_textures:
                selected = np.flatnonzero(selected_groups == group_index)
                coords = np.einsum('ni,nij->nj', bary[selected], uv[selected_faces[selected]])
                values[selected] = bilinear(alpha_textures[name][..., None], coords)[:, 0]
        return values
    row_alpha = np.empty(len(faces), np.float32)
    row_bary = np.load(metadata/'barycentric.npy', mmap_mode='r')
    for start in range(0, len(faces), 262144):
        stop = min(start+262144, len(faces))
        saved_bary = row_bary[start:stop]
        # Both the initializer and Warp store weights for vertices 0 and 1.
        bary = np.column_stack((saved_bary, 1-saved_bary.sum(-1)))
        row_alpha[start:stop] = surface_alpha(faces[start:stop], bary)
    np.save(args.output_dir/'row_alpha.npy', row_alpha)
    for name, value in [('face_alpha.npy', alpha), ('face_group.npy', face_group), ('group_density.npy', density)]:
        np.save(args.output_dir/name, value)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    camera_file = args.references/'training_cameras.json'
    camera_data = json.loads(camera_file.read_text())
    origin = np.asarray(report['source_bounds']).mean(0)
    wp.config.quiet = True
    wp.init()
    flat = (triangles.reshape(-1, 3)-origin).astype(np.float32)
    mesh = wp.Mesh(points=wp.array(flat, dtype=wp.vec3, device='cuda:0'),
                   indices=wp.array(np.arange(len(flat), dtype=np.int32), dtype=wp.int32, device='cuda:0'))
    alpha_gpu = wp.array(alpha, device='cuda:0')
    h, w = camera_data['height'], camera_data['width']
    y, x = np.mgrid[:h, :w]
    pixels = np.stack((x+.5, y+.5, np.ones_like(x)), axis=-1).reshape(-1, 3)
    masks = {}
    (args.output_dir/'opaque_depth_masks').mkdir()
    for i, camera in enumerate(camera_data['cameras']):
        mat = np.asarray(camera['camera_to_world'])
        local = pixels@np.linalg.inv(camera['K']).T
        lengths = np.linalg.norm(local, axis=1)
        directions = ((local/lengths[:, None])@mat[:3, :3].T).astype(np.float32)
        origins = np.broadcast_to(mat[:3, 3]-origin, directions.shape).copy().astype(np.float32)
        geometric = wp.zeros(h*w, device='cuda:0')
        hit_face = wp.full(h*w, -1, dtype=wp.int32, device='cuda:0')
        hit_bary = wp.zeros(h*w, dtype=wp.vec2, device='cuda:0')
        wp.launch(_first_hits, h*w, device='cuda:0', inputs=[mesh.id,
            wp.array(origins, dtype=wp.vec3, device='cuda:0'),
            wp.array(directions, dtype=wp.vec3, device='cuda:0'),
            wp.array((local[:, 2]/lengths).astype(np.float32), device='cuda:0'), alpha_gpu,
            camera['near'], camera['far'], geometric, hit_face, hit_bary])
        geo = geometric.numpy().reshape(h, w)
        ray_faces = hit_face.numpy()
        hit = ray_faces >= 0
        bary2 = hit_bary.numpy()[hit]
        bary = np.column_stack((bary2, 1-bary2.sum(-1)))
        ray_alpha = np.zeros(h*w, np.float32)
        ray_alpha[hit] = surface_alpha(ray_faces[hit], bary)
        hit = (hit & (ray_alpha >= .99)).reshape(h, w)
        exr = args.references/'training_references'/f'{camera["name"]}.exr'
        _, reference_alpha, depth = read_exr(exr)
        valid = hit & (reference_alpha >= .99) & np.isfinite(depth) & (depth > camera['near']) & (depth < camera['far'])
        valid &= np.abs(geo-depth) <= .005
        interior = binary_erosion(valid, iterations=3) & ~binary_dilation(contours(depth, valid), iterations=3)
        path = args.output_dir/'opaque_depth_masks'/f'{camera["name"]}.npy'
        np.save(path, interior)
        masks[camera['name']] = dict(sha256=digest(path), pixels=int(interior.sum()), reference_exr_sha256=digest(exr))
        if i == 0 or (i+1) % 20 == 0 or i+1 == len(camera_data['cameras']):
            print(json.dumps(dict(views=i+1, total=len(camera_data['cameras']), seconds=time.perf_counter()-tick)), flush=True)
    manifest = dict(input_sha256=digest(args.model_dir/'model.ply'), source_sha256=report['input_sha256'],
        camera_sha256=digest(camera_file), material_constants=descriptions,
        files={name:digest(args.output_dir/name) for name in ('face_alpha.npy','face_group.npy','group_density.npy','row_alpha.npy')},
        depth_masks=masks, depth_semantics='Opaque geometric first hit; Cycles camera-Z agrees within 5 mm; eroded 3 px',
        transparent_gaussians=int((row_alpha < .99).sum()), seconds=time.perf_counter()-tick)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('complete', manifest['transparent_gaussians'], manifest['seconds'], flush=True)

if __name__ == '__main__':
    main()

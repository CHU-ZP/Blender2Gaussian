# Local validation scene

Place `sk_00918.blend` in this directory to reproduce the interior validation.
The 402,372,868-byte source asset is kept locally and excluded from Git.

SHA256: `514601b18893a7c18c0759e703898eee0e262e5c5ad9e0c2e97f3a7119e9cba6`

The evaluated scene contains 2,494 mesh objects and 4,936,252 triangles. One
world unit is one meter. Fixed validation cameras are versioned in
[`configs/sk_00918_cameras_test.json`](../configs/sk_00918_cameras_test.json).

See [`ENVIRONMENT.md`](../ENVIRONMENT.md) for initialization and rendering.

## Second scene (optional)

`riliaodian.blend` is converted from the user-accepted restaurant delivery
`/data/public_data/wanhu/for_sences/riliaodian_2026-09-15-095822/scene_delivery_final_001/scene.glb`
with `scripts/convert_gltf_scene.py` (3 vertex-colour materials replaced by
their mean colour, see `riliaodian.json`). Objects only, no walls or floor; the initializer detects this
(`--scene-type auto` selects open-scene visibility viewpoints).
Cameras: `configs/riliaodian_cameras_{train,test,render}.json`.

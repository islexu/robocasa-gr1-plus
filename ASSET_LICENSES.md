# Asset provenance and licenses

**Powered by Poly Haven.** The eight photo textures in the default v0.3 pack are
[Poly Haven](https://polyhaven.com/) originals released under
[CC0](https://polyhaven.com/license). The lockfile contains each asset page, metadata
URL, exact 1K diffuse JPEG URL, byte count, upstream MD5 and independently measured SHA-256.
Files were retrieved and verified on 2026-10-22. Original image bytes are not bundled
in this repository; `gr1-plus assets fetch` prepares them explicitly.

| Split | Asset IDs | Families |
| --- | --- | --- |
| Calibration | `wood_table_001`, `denim_fabric` | Wood, fabric |
| Development | `brick_wall_001`, `concrete_floor_01` | Brick, concrete |
| Confirmation | `marble_01`, `blue_floor_tiles_01`, `blue_plaster_wall`, `fabric_leather_01` | Marble, ceramic, plaster, leather |

The original household primitive-coordinate data (block, carton, can, bottle, jar, mug)
authored for `scene_v3.py` are also released under CC0-1.0. These are simple procedural
geometry models, not Poly Haven/RoboTwin/Objaverse/GSO scanned models. Generator code
retains the repository code license. Their geometric bounds fit the placement checks;
they do not establish photorealism or a validated physical benchmark.

Original RoboCasa-GR1 robot/scene/object assets are downloaded with the official upstream
tools and retain their own licenses. RoboTwin's texture/object library, LIBERO-PRO data,
and SimplerEnv assets are not redistributed or converted here. DiT4DiT source/weights
and Cosmos weights retain their original licenses and are installed separately.

# Test-mesh assets

- `car.obj` — stylized sedan from the [Kenney Car Kit](https://kenney.nl/assets/car-kit)
  (version 3.1), licensed [CC0 / public domain](https://creativecommons.org/publicdomain/zero/1.0/)
  (crediting Kenney is optional; gladly given). Converted for this repository:
  groups merged, faces triangulated, midpoint-subdivided for vertex density.
  Used as a `--target-obj --smooth 0.2` showcase for `shape_fit`.
- `cat.obj` — stretching cat (low-resolution remesh), from [odedstein/meshes](https://github.com/odedstein/meshes/tree/master/objects/cat);
  original mesh by [billyd via Thingiverse](https://www.thingiverse.com/thing:1565405),
  licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used as a
  `--target-obj --smooth 0.2` showcase for `shape_fit`.
- `robot.obj` — toy robot, **original**: procedurally generated for this repository as
  densely tessellated triangle soup (boxes, cylinders, spheres). Same license as the
  repository (Apache-2.0). Used as a `--target-obj --smooth 0.2` showcase for `shape_fit`.

Every asset in this directory carries a clear license: CC0, CC BY 4.0 (attributed
above and in the workspace NOTICE), or the repository's own Apache-2.0.

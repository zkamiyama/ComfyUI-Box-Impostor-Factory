---
name: chat-3d-viewer
description: Build and display a polished, self-contained interactive 3D asset inspector directly in ChatGPT or as a downloadable HTML file. Use for uploaded 3D meshes (GLB, GLTF, OBJ, STL, PLY, 3MF) and Box Impostor .box.zip archives, including model orbit, camera presets, real vertex/triangle counts, mesh hierarchy, material isolation, PBR properties, texture channels and UV inspection. Works offline with no CDN or external asset fetches.
license: MIT
metadata:
  author: Chat 3D Inspector contributors
  version: 0.1.0
---

# Chat 3D Viewer

Produce an **offline HTML 3D Inspector** from an actual input model. Never substitute a generic cube, unrelated stock illustration, or fabricated model statistics. Prefer the original mesh when one is supplied. For a Box Impostor archive, label the derived surface preview as an approximation rather than claiming native three-impostor ray marching.

## When to use

Use when someone asks to view a 3D model in ChatGPT, inspect its mesh/materials/textures, show a GLB or model interactively, or reuse a common 3D preview UI. If the input is not available as actual file bytes, ask them to upload it or use an existing verified file. Do **not** attempt to fetch large assets from an arbitrary public GitHub URL inside an embedded chat application: sandboxed clients often reject remote fetches.

## Procedure

1. Locate the real input file. Supported types: `.glb`, `.gltf`, `.obj`, `.stl`, `.ply`, `.3mf`, `.box.zip`. For glTF/OBJ with separate files, keep the texture and MTL/image files beside the source. For `.box.zip`, do not extract untrusted paths outside a scratch directory; the builder validates archive paths, texture checksums and resource limits.
2. Ensure Python packages `numpy`, `Pillow`, and `trimesh` are available. Install from `requirements.txt` in an isolated environment if missing. No Node.js, Three.js CDN, or external fetch is needed for the final viewer.
3. Run:
   ```bash
   python scripts/build_viewer.py /path/to/model.glb --output /mnt/data/model.inspector.html
   ```
   For Box Impostor `.box.zip`, use `--grid 96` for a denser derived surface at the cost of file size. Keep the **Box geometry sampled** caveat.
4. Verify the printed JSON counts. For native meshes, show actual source vertices/triangles and displayed counts; if reduced for preview, distinguish the two. For Box archives, show the reconstructed preview vertices/triangles separately from original proxy mesh counts where present.
5. If your ChatGPT client supports directly embedded interactive HTML, render the **complete generated HTML source** as a self-contained interactive app (e.g. a supported inline HTML/AppBlock surface); do not reference the file by an unresolvable local path inside an iframe. The complete inline HTML must contain the actual asset payload and its renderer; do not request any scripts or textures from CDNs. If inline embedding is not supported, provide a verified download link to the generated `.html`, ideally accompanied by a screenshot and the original asset ZIP/model.
6. Test loading, orbiting, camera preset changes, material visibility, the Textures tab, channel selection and UV overlay when possible with Playwright. At minimum check that the HTML references no remote scripts and scene payload matches source stats.

## UI requirements

The UI is generic, not named for any particular dog, generative model or project. The **model** is the visual focus; chrome is quiet graphite. Use the existing viewer template, not a new row of unrelated buttons. It includes a compact camera toolbar and a side inspector with Details, Materials and Textures. Show **vertices, triangles, meshes and materials**, dimensions, source counts, material isolation, texture resolution, RGBA/R/G/B/A views, UV overlay, screenshot, optional wireframe and normals. Camera orbit, pan and zoom work via mouse or touch where supported. Offer a working Canvas2D fallback when WebGL2 is blocked.

The design follows the practical redline discipline from [unslop](https://github.com/heydamir/unslop) by Damir Sokolovsky (CC BY 4.0): consistent 4px spacing, subdued controls, one restrained accent, material-appropriate typography, and no gradients, glows, emoji buttons or decorative badges. Do not copy its code.

## Supported formats and fidelity

- GLB/GLTF/OBJ/PLY/STL/3MF: inspect triangle geometry and material data through `trimesh`. Baked textures are embedded as PNGs; material PBR factors and texture roles are listed. The viewport is a lightweight PBR-style shaded renderer, **not a full glTF extension renderer** (clearcoat, transmission, skinning and animation are not supported).
- Box `.box.zip`: read six face depth guides and raw half-float color/PBR atlases. Render an approximate surface mesh with atlas UVs; list original map images under Textures. It is **not** the exact three-impostor shader, and mesh counts apply to the preview reconstruction. Full native depth ray marching needs the original project viewer.
- Oversized scenes may be downsampled (`--max-triangles` per mesh, `--grid`, `--max-texture`); say so. The original input file is never edited.

## Final report

Lead with the interactive preview when the client actually supports it; otherwise link the verified standalone HTML. State file type, actual/preview vertex and triangle counts, materials, textures, limitations and whether in-chat rendering was tested. Never promise universal inline support in every ChatGPT client. The Skill is **installable reusable source**, not a plugin automatically enabled for everyone; eligible ChatGPT users can open Plugins → Skills → Create → Upload from your computer; each ChatGPT/agent environment must have the required file-building capability and UI support.

# Direct3D-V3: Latent-Free Diffusion Transformers in the Native Voxel Domain

<div align="center">

**ACM Transactions on Graphics (SIGGRAPH Asia 2026)**

Shuang Wu<sup>1,2</sup>, Jiachen Qian<sup>2,3</sup>, Youtian Lin<sup>1</sup>, Feihu Zhang<sup>2</sup>, Philip Torr<sup>4</sup>, Yao Yao<sup>1</sup>, and Xun Cao<sup>1</sup>

<sup>1</sup>Nanjing University &nbsp; <sup>2</sup>DreamTech &nbsp; <sup>3</sup>The University of Hong Kong &nbsp; <sup>4</sup>University of Oxford

<a href="https://dreamtechai.github.io/Direct3D-V3"><img src="https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome" alt="Project Page" height="22"></a>
<a href="" title="Demo link coming soon"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg" alt="Demo" height="22"></a>
<a href="https://huggingface.co/wushuang98/Direct3D-V3"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg" alt="Models" height="22"></a>
<a href="https://dl.acm.org/doi/10.1145/3842524"><img src="https://img.shields.io/badge/Paper-00629B.svg" alt="Paper" height="22"></a>
<a href="" title="arXiv link coming soon"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv" height="22"></a>

</div>

![Direct3D-V3 teaser: diverse, high-fidelity 3D meshes generated directly in the native voxel domain.](assets/teaser.png)

Direct3D-V3 generates diverse, high-fidelity 3D meshes by modeling raw SDF voxel grids directly with latent-free diffusion transformers, removing the VAE bottleneck used by latent diffusion pipelines.

> **The released model generates meshes with substantially finer surface detail than the version evaluated in the paper.** For image conditioning, we now combine features from four DINOv3 layers. Switching to the Muon optimizer both accelerated training convergence and improved the quality of the generated meshes.

## News

- **2026.11.19**: Inference code and models are released.
- **2026.11.19**: The [project page](https://dreamtechai.github.io/Direct3D-V3) and [paper](https://dl.acm.org/doi/10.1145/3842524) are released.
- **2026.08.25**: Direct3D-V3 is accepted to ACM Transactions on Graphics (SIGGRAPH Asia 2026).

## TODO

- ✅ Release the inference code and models.
- [ ] Release the coarse mesh generation models.
- [ ] Release the latent-free texture generation model.
- [ ] Release the training code.

## Input preparation

### Coarse mesh

**Our coarse mesh generation models have not yet been released.** In the meantime, you can generate an initial mesh with [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/tree/main/hy3dshape) or [UltraShape](https://github.com/PKU-YuanGroup/UltraShape-1.0).

Use the resulting OBJ, GLB, or PLY as `--coarse-mesh`, keeping its orientation aligned with the reference image. These are optional external tools with their own licenses; see the [license notes](#license) before using them.

### Reference image

For richer geometric detail, we recommend preparing a **surface normal map or white clay image** from your RGB input with **ChatGPT (GPT Image 2.5)**. [Lotus-2](https://github.com/EnVision-Research/Lotus-2) is another option for predicting surface normal maps. Preserve the original viewpoint, silhouette, proportions, and as much fine geometry as possible, including small grooves, sharp edges, and thin structures. Detailed geometric cues in the conditioning image help Direct3D-V3 generate meshes with finer surface detail.

The paired images below, from `assets/input_imgs/`, illustrate the recommended **RGB → GPT Image 2.5 → normal map** preparation workflow:

<table>
  <tr>
    <th align="center">RGB input</th>
    <th align="center">Image preparation</th>
    <th align="center">Normal map</th>
  </tr>
  <tr>
    <td align="center" width="40%"><img src="assets/input_imgs/49.png" alt="RGB reference of a robot with detailed mechanical parts" width="320"></td>
    <td align="center" width="20%">→<br><strong>GPT Image 2.5</strong><br>Preserve fine<br>geometric details<br>→</td>
    <td align="center" width="40%"><img src="assets/input_imgs/49_normal.png" alt="Paired robot normal map showing fine mechanical surface details" width="320"></td>
  </tr>
</table>

Pass the prepared normal map or white clay image as `--image`, together with its aligned coarse mesh. For example, use `assets/input_imgs/0_normal.png` as the image path in the inference commands below.

## Installation

The demo uses Python 3.12, PyTorch 2.14.1, and CUDA 13.0 on Linux. Install the CUDA 13.0 toolkit and a C++20 compiler before building the GPU extensions.

```bash
git clone https://github.com/DreamTechAI/Direct3D-V3.git
cd Direct3D-V3
conda create -n direct3d-v3 python=3.12 -y
conda activate direct3d-v3

pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt gradio==6.32.0 setuptools wheel packaging psutil ninja

# Install FlashAttention 2 with C++20 support for PyTorch 2.14
CXXFLAGS="-std=c++20" pip install --no-build-isolation \
    git+https://github.com/Dao-AILab/flash-attention.git@edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d

# Install voxelization and hole filling
pip install --no-build-isolation ./third_party/voxelization ./third_party/fill_holes
```

## Model weights

`infer.py` and `app.py` load weights from `wushuang98/Direct3D-V3` on Hugging Face by default. Use `--model` to select another Hub repository or a local model directory.

The model directory should have the following layout:

```text
config.json
model_index.json
shape_transformers_512/
    config.json
    model.safetensors
shape_transformers_1024/
    config.json
    model.safetensors
shape_transformers_1536/
    config.json
    model.safetensors
```

`from_pretrained` reads the `model_index` path from the root `config.json`, then loads each denoiser with the architecture and sampling settings in its own directory. Earlier releases use `model_index.json` directly. Calling `.to("cuda")` places all three denoisers on the selected GPU. Pass `revision=` to pin the model repository's weights and configs together.

The `image_encoder` field in `model_index.json` selects the DINOv3 encoder. It accepts a Hugging Face model ID or a directory in the same model repository such as `./image_encoder`.

Image conditioning uses DINOv3 ViT-H+ and foreground segmentation uses BiRefNet. For offline inference, cache both image models and all three denoisers before setting `HF_HUB_OFFLINE=1`. The hosted DINOv3 weights require Hugging Face access. [DINOv3 ViT-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) · [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)

## Inference

Supply an RGB or RGBA reference image and a coarse mesh in the same orientation. The pipeline accepts a PIL image or image path and a `trimesh.Trimesh` or mesh path (OBJ, GLB, PLY). It returns a `trimesh.Trimesh` in the normalized reconstruction frame.

```python
from direct3d import Direct3DV3Pipeline

pipeline = Direct3DV3Pipeline.from_pretrained("wushuang98/Direct3D-V3").to("cuda")
mesh = pipeline(
    image="assets/input_imgs/6_normal.png",
    coarse_mesh="assets/coarse_meshes/6.obj",
    resolution=1024,
    num_inference_steps={512: 15, 1024: 15},
    guidance_scale={512: 4.0, 1024: 4.0},
    max_tokens=150_000,
    target_faces=2_000_000,
)
mesh.export("generated.glb")
```

`max_tokens` limits the number of occupied voxel patches per stage. When the count exceeds the limit, the pipeline scales the centered mesh by 0.99 and repeats voxelization until it fits. **If generation runs out of GPU memory, try lowering `max_tokens` (CLI flag `--max-tokens`) from its default of 150,000.** Using fewer tokens reduces memory use but can also reduce fine surface detail in the generated mesh.

If the final mesh exceeds `target_faces`, the pipeline uses pyfqmr to reduce its triangle count while preserving mesh boundaries. Pass `target_faces=None` to keep all faces.

### Command line

```bash
python infer.py --image assets/input_imgs/6_normal.png --coarse-mesh assets/coarse_meshes/6.obj \
    --resolution 1024 --steps-1024 15 --cfg-1024 4.0 --output generated.glb
```

Use `--steps-<resolution>` and `--cfg-<resolution>` to override individual stages. The shared flags `--steps` and `--cfg` accept one value for all stages or one per active stage in ascending resolution order. A stage-specific flag takes precedence. For a batch, pass an image directory and an OBJ directory with matching filename stems. The command reuses one pipeline across the batch.

```bash
python infer.py --infer_path ./images --coarse-mesh ./coarse_meshes \
    --resolution 1024 --output ./generated
```

To run one stage, supply its checkpoint and the mesh entering that stage. A checkpoint in a downloaded stage directory uses the surrounding model repository for its configs. A standalone checkpoint uses the default Hugging Face repository for configs. Use `--model` to pick a different config repository. Use `--no-simplify` to keep the full extracted surface.

```bash
python infer.py --checkpoint ./model.safetensors --resolution 512 --image assets/input_imgs/6_normal.png \
    --coarse-mesh assets/coarse_meshes/6.obj --output generated-512.glb
```

## Gradio demo

Launch the demo with the default Hugging Face model, `wushuang98/Direct3D-V3`.

```bash
python app.py
```

To use a local copy, run `python app.py --model /path/to/Direct3D-V3-weights`. The same option accepts another Hugging Face model ID.

Open the URL printed by Gradio. Select an original image or its corresponding normal map to load it with the paired coarse OBJ, or upload your own aligned image and mesh. Choose a resolution and select **Generate**. The foreground appears as soon as background removal finishes. The progress bars show completed diffusion steps for each active resolution. Switch between **White clay** and **Normal** in the mesh viewer, or download the generated GLB. Drag to orbit, scroll to zoom, and Shift-drag to pan. **Sampling and mesh options** shows the configured steps and CFG for each stage. Mesh simplification defaults to 2 million triangles and accepts targets from 100,000 to 5 million.

## Citation

```bibtex
@article{wu2026direct3dv3,
  title   = {{Direct3D-V3}: Latent-Free Diffusion Transformers in the Native Voxel Domain},
  author  = {Wu, Shuang and Qian, Jiachen and Lin, Youtian and Zhang, Feihu and
             Torr, Philip and Yao, Yao and Cao, Xun},
  journal = {ACM Transactions on Graphics},
  year    = {2026},
  volume  = {45},
  number  = {6},
  articleno = {244},
  doi     = {10.1145/3842524}
}
```

## License

The Direct3D-V3 source is released under the MIT License. External weights and dependencies retain their respective licenses. [MIT License](LICENSE)

Hunyuan3D-2.1 and UltraShape-1.0 are linked as optional external tools. Their code and weights are not covered by Direct3D-V3's MIT license. Both repositories currently publish the Tencent Hunyuan 3D 2.1 Community License Agreement: [Hunyuan3D-2.1 license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE) and [UltraShape-1.0 license](https://github.com/PKU-YuanGroup/UltraShape-1.0/blob/main/LICENSE). Their terms apply when using those tools and their outputs, including territorial and commercial conditions and restrictions on using them to improve other AI models.


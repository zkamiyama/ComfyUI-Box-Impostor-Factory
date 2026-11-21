import json
import math
from os import PathLike, environ
from pathlib import Path

import torch
import trimesh
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image

from .schedulers import sample_x0
from .utils.image import BackgroundRemover, prepare_image
from .utils.mesh import export_mesh, load_coarse

RESOLUTIONS = (512, 1024, 1536)
DEFAULT_MODEL_ID = "wushuang98/Direct3D-V3"
DEFAULT_MAX_TOKENS = 150_000
DEFAULT_TARGET_FACES = 2_000_000


def _stage_overrides(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        if any(stage not in RESOLUTIONS for stage in value):
            raise ValueError(f"Sampling overrides must use resolution keys from {RESOLUTIONS}.")
        return value
    return dict.fromkeys(RESOLUTIONS, value)


class Direct3DV3Pipeline:
    """Generate a mesh from a reference image and aligned coarse geometry."""

    def __init__(
        self, repo, *, image_encoder, torch_dtype, revision, local_files_only, stages=RESOLUTIONS, checkpoint=None
    ):
        if torch_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("torch_dtype must be torch.float16 or torch.bfloat16.")
        local = Path(repo).expanduser()
        self.repo = local if local.is_dir() else str(repo)
        self._stages = stages
        self.dtype = torch_dtype
        self.revision = revision
        self.local_files_only = local_files_only
        self.device = torch.device("cpu")
        try:
            config = self._read_config("config.json")
        except (FileNotFoundError, EntryNotFoundError):
            # Earlier releases load model_index.json directly.
            config = {}
        index = self._read_config(config.get("model_index", "model_index.json"))
        self._encoder_options = (
            self._image_encoder_options(index)
            if image_encoder is None
            else {"model_name_or_path": image_encoder}
        )
        self._denoisers = {
            stage: self._load_stage(stage, index[f"shape_transformers_{stage}"], checkpoint) for stage in stages
        }
        self._encoder = None
        self._remover = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        image_encoder=None,
        torch_dtype=torch.float16,
        revision=None,
        local_files_only=False,
    ):
        """Load from a Hugging Face model ID or a directory with per-stage safetensors."""
        return cls(
            pretrained_model_name_or_path,
            image_encoder=image_encoder,
            torch_dtype=torch_dtype,
            revision=revision,
            local_files_only=local_files_only,
        )

    @classmethod
    def from_single_file(
        cls,
        checkpoint,
        *,
        resolution,
        config_repo=None,
        image_encoder=None,
        torch_dtype=torch.float16,
        revision=None,
        local_files_only=False,
    ):
        """Load a standalone stage to generate geometry at its native resolution."""
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if resolution not in RESOLUTIONS:
            raise ValueError("resolution must be 512, 1024 or 1536.")
        if config_repo is None:
            root = path.parent.parent
            if (root / "model_index.json").is_file():
                config_repo = root
            else:
                config_repo = environ.get("DIRECT3D_MODEL") or DEFAULT_MODEL_ID
        return cls(
            config_repo,
            image_encoder=image_encoder,
            torch_dtype=torch_dtype,
            revision=revision,
            local_files_only=local_files_only,
            stages=(resolution,),
            checkpoint=path,
        )

    def _model_file(self, filename):
        if isinstance(self.repo, Path):
            return self.repo / filename
        return Path(
            hf_hub_download(self.repo, filename, revision=self.revision, local_files_only=self.local_files_only)
        )

    def _read_config(self, filename):
        return json.loads(self._model_file(filename).read_text(encoding="utf-8"))

    def _image_encoder_options(self, index):
        if "image_encoder" in index:
            model = index["image_encoder"]
        else:
            # Older model indexes store the encoder in a separate config.
            model = self._read_config(index["image_encoder_config"])["image_encoder"]["model_name_or_path"]

        if model.startswith("./"):
            if isinstance(self.repo, Path):
                return {"model_name_or_path": self.repo / model}
            return {"model_name_or_path": self.repo, "subfolder": model[2:], "revision": self.revision}
        return {"model_name_or_path": model}

    def _load_stage(self, resolution, folder, checkpoint):
        from .models.checkpoint import load_denoiser

        config = self._read_config(f"{folder}/config.json")
        if config["resolution"] != resolution:
            raise ValueError(f"Expected the {resolution} stage config in {folder}.")
        path = checkpoint or self._model_file(f"{folder}/model.safetensors")
        return config, load_denoiser(path, config["denoiser"], self.dtype, allow_wrapped=checkpoint is not None)

    def to(self, device):
        """Move all geometry models to the selected CUDA device."""
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("Direct3D-V3 generation requires CUDA.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        for _, denoiser in self._denoisers.values():
            denoiser.to(device)
        self.device = device
        return self

    @property
    def sampling_defaults(self):
        return {
            stage: (config["sampling"]["steps"], config["sampling"]["guidance_scale"])
            for stage, (config, _) in self._denoisers.items()
        }

    @torch.inference_mode()
    def remove_background(self, image):
        """Return the full-size RGBA foreground used for image conditioning."""
        if self.device.type == "cpu":
            self.to("cuda")
        if self._remover is None:
            self._remover = BackgroundRemover(self.local_files_only)
        with torch.cuda.device(self.device):
            return self._remover(image, self.device)

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image | str | PathLike,
        coarse_mesh: trimesh.Trimesh | str | PathLike | None = None,
        *,
        resolution: int = 1024,
        num_inference_steps: int | dict[int, int] | None = None,
        guidance_scale: float | dict[int, float] | None = None,
        clamp_range: tuple[float, float] | None = None,
        seed: int = 2026,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        mc_threshold: float | None = None,
        target_faces: int | None = DEFAULT_TARGET_FACES,
        background_removed: bool = False,
        on_sampling_step=None,
    ) -> trimesh.Trimesh:
        """Generate a mesh. on_sampling_step(stage, completed, total) reports diffusion progress."""
        if coarse_mesh is None:
            raise ValueError("Provide a coarse mesh aligned with the reference image.")
        if resolution not in self._stages:
            raise ValueError(f"Available output resolutions for this pipeline: {self._stages}.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        steps = _stage_overrides(num_inference_steps)
        guidance = _stage_overrides(guidance_scale)
        if any(type(n) is not int or n < 1 for n in steps.values()):
            raise ValueError("num_inference_steps must specify a positive integer for each stage.")
        if any(type(g) not in (int, float) or not math.isfinite(g) or g < 0 for g in guidance.values()):
            raise ValueError("guidance_scale must specify a finite nonnegative number for each stage.")
        if target_faces is not None and (not isinstance(target_faces, int) or target_faces < 10000):
            raise ValueError("target_faces must be an integer of at least 10000.")
        mesh = load_coarse(coarse_mesh)
        if self.device.type == "cpu":
            self.to("cuda")
        if self._encoder is None:
            from .models.image_encoder import DINOv3FeatureExtractor

            self._encoder = DINOv3FeatureExtractor(
                **self._encoder_options,
                dtype=self.dtype,
                local_files_only=self.local_files_only,
            )
        with torch.cuda.device(self.device):
            foreground = image if background_removed else self.remove_background(image)
            pixels = prepare_image(foreground)
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
            faces = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            for stage in self._stages:
                if stage > resolution:
                    break
                sampling = self._denoisers[stage][0]["sampling"].copy()
                if stage in steps:
                    sampling["steps"] = steps[stage]
                if stage in guidance:
                    sampling["guidance_scale"] = guidance[stage]
                if clamp_range is not None:
                    sampling["clamp_range"] = clamp_range
                if on_sampling_step is not None:
                    on_sampling_step(stage, 0, sampling["steps"])
                vertices, faces = self._generate_stage(
                    vertices, faces, pixels, stage, sampling, generator, mc_threshold, max_tokens, on_sampling_step
                )
            return export_mesh(vertices, faces, target_faces)

    def _generate_stage(self, vertices, faces, pixels, stage, sampling, generator, threshold, max_tokens, on_sampling_step):
        from .utils.sparse_mc import sparse_mc_infer
        from .utils.voxelization import mesh_patches

        config, denoiser = self._denoisers[stage]
        patch_size = config["denoiser"]["patch_size"]
        coords = mesh_patches(vertices, faces, stage, patch_size, max_tokens)
        self._encoder.to(self.device)
        try:
            with torch.autocast("cuda", dtype=self.dtype):
                context = self._encoder(pixels, config["image_size"])
        finally:
            self._encoder.cpu()
        with torch.autocast("cuda", dtype=self.dtype):
            sdf = sample_x0(
                denoiser, coords.int(), context, generator, patch_size=patch_size, resolution=stage,
                on_sampling_step=on_sampling_step, **sampling
            )
        if threshold is None:
            threshold = config["mc_threshold"]
        return sparse_mc_infer(sdf.float(), coords[:, 1:], stage, patch_size, mc_threshold=threshold)


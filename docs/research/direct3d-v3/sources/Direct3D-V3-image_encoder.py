import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from transformers import DINOv3ViTModel

DINO_FEATURE_LAYERS = (4, 8, 16, 32)


class DINOv3FeatureExtractor(torch.nn.Module):
    """Extract the four DINOv3 block outputs used by the geometry checkpoints."""

    def __init__(self, model_name_or_path, *, dtype, local_files_only=False, subfolder="", revision=None):
        super().__init__()
        self.backbone = DINOv3ViTModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            local_files_only=local_files_only,
            subfolder=subfolder,
            revision=revision,
        )
        self.normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.eval()

    @torch.inference_mode()
    def forward(self, image: torch.Tensor, image_size: int) -> torch.Tensor:
        backbone = self.backbone
        weight = backbone.embeddings.patch_embeddings.weight
        image = image.to(device=weight.device, dtype=torch.float32)
        if image.shape[-2:] != (image_size, image_size):
            image = F.interpolate(
                image, size=(image_size, image_size), mode="bicubic", align_corners=False, antialias=True
            )
        image = self.normalize(image).to(weight.dtype)
        tokens = backbone.embeddings(image)
        positions = backbone.rope_embeddings(image)

        # Transformers exposes the blocks on the backbone or its nested encoder.
        blocks = getattr(backbone, "model", backbone).layer
        features = []
        for depth, block in enumerate(blocks, start=1):
            tokens = block(tokens, position_embeddings=positions)
            if depth in DINO_FEATURE_LAYERS:
                features.append(backbone.norm(tokens))
        return torch.cat(features, dim=-1)


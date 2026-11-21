from itertools import product

import cudavoxelize
import torch


def voxelize(vertices, faces, dim, patch_size):
    vertices = vertices.to(torch.float32).contiguous() * 0.5
    faces = faces.to(device=vertices.device, dtype=torch.int32).contiguous()
    voxel = torch.zeros(dim**3, device=vertices.device, dtype=torch.bool)
    cudavoxelize.voxelize(vertices, faces, voxel, len(faces), dim, 8.0)
    voxel = voxel.view(dim, dim, dim)
    grid = dim // patch_size
    occupied = torch.zeros(grid**3, device=vertices.device, dtype=torch.bool)
    # Dilate each occupied voxel by one positive neighbor along each axis.
    # Accumulate directly in patch space so the full dilated voxel set is never materialized.
    for start in range(0, dim, 128):
        points = voxel[start : start + 128].nonzero()
        points[:, 0] += start
        lo = points // patch_size
        hi = (points + 1).clamp_max_(dim - 1) // patch_size
        for x, y, z in product((lo, hi), repeat=3):
            occupied[(x[:, 0] * grid + y[:, 1]) * grid + z[:, 2]] = True
    coords = occupied.reshape(grid, grid, grid).nonzero()
    return torch.cat((torch.zeros_like(coords[:, :1]), coords), dim=1)


def mesh_patches(vertices, faces, resolution, patch_size, max_tokens):
    low, high = vertices.amin(dim=0), vertices.amax(dim=0)
    extent = (high - low).max()
    if extent <= 0:
        raise ValueError("The coarse mesh has zero extent.")
    centered = (vertices - (low + high) / 2) * (1.9 / extent)
    scale = 1.0
    while True:
        coords = voxelize(centered * scale, faces, dim=resolution, patch_size=patch_size)
        if not len(coords):
            raise ValueError(f"No occupied patches at resolution {resolution}.")
        if len(coords) <= max_tokens:
            return coords
        # Once the centered mesh fits inside the central voxel, further scaling cannot free patches.
        if scale < 1 / (resolution - 1):
            raise ValueError(f"At resolution {resolution} the mesh needs at least {len(coords)} tokens.")
        scale *= 0.99


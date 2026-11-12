"""Compare a 256px WaterBottle bake to the CC0 source GLB at visible surface points."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image


SOURCE_SHA256 = "b337e526fd6a162013c2984aeec163f5fbb4f717252724dfc3f3458bd51df94b"
# Interior pixels on the six guide faces, including the emissive label, dark AO,
# bare metal, matte sleeve, and visibly perturbed normals.
PROBES = {
    "+X": [(143, 166), (107, 47), (81, 230)],
    "-X": [(78, 182), (193, 197)],
    "+Y": [(236, 100), (125, 169)],
    "-Y": [(212, 52), (40, 198)],
    "+Z": [(176, 193), (242, 165), (62, 197)],
    "-Z": [(116, 62), (160, 11)],
}


def normalized(vector):
    return vector / np.linalg.norm(vector)


def water_bottle(path):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the pinned Khronos WaterBottle.glb")
    size, = struct.unpack_from("<I", raw, 12)
    doc = json.loads(raw[20:20 + size])
    buffer = memoryview(raw)[28 + size:]

    def buffer_view(index):
        view = doc["bufferViews"][index]
        offset = view.get("byteOffset", 0)
        return buffer[offset:offset + view["byteLength"]]

    def accessor(index):
        desc = doc["accessors"][index]
        width = {"VEC2": 2, "VEC3": 3, "VEC4": 4, "SCALAR": 1}[desc["type"]]
        dtype = {5123: "<u2", 5126: "<f4"}[desc["componentType"]]
        return np.frombuffer(buffer_view(desc["bufferView"]), dtype).reshape(-1, width)

    primitive = doc["meshes"][0]["primitives"][0]
    indices = accessor(primitive["indices"]).reshape(-1, 3)
    triangles = {k: accessor(v)[indices] for k, v in primitive["attributes"].items()}
    # WaterBottle has a single node rotated 180 degrees about Y.
    for key in ("POSITION", "NORMAL", "TANGENT"):
        triangles[key][..., :3] *= [-1, 1, -1]
    material = doc["materials"][primitive["material"]]
    pbr = material["pbrMetallicRoughness"]
    slots = [pbr["baseColorTexture"], pbr["metallicRoughnessTexture"],
             material["normalTexture"], material["emissiveTexture"]]
    assert material["occlusionTexture"]["index"] == slots[1]["index"]
    images = []
    for slot in slots:
        info = doc["images"][doc["textures"][slot["index"]]["source"]]
        with Image.open(BytesIO(buffer_view(info["bufferView"]))) as image:
            images.append(np.asarray(image.convert("RGB")))
    return triangles, images


def visible_hit(screen, x, y):
    # Orthographic projection keeps barycentric coordinates affine. Front-facing
    # triangles have negative screen-space area because guide Y grows downward.
    edges = screen[:, 1:, :2] - screen[:, :1, :2]
    dx, dy = (np.array([x + .5, y + .5]) - screen[:, 0, :2]).T
    (ux, uy), (vx, vy) = edges.transpose(1, 2, 0)
    det = ux * vy - uy * vx
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = (dx * vy - dy * vx) / det, (ux * dy - uy * dx) / det
        weights = np.column_stack([1 - u - v, u, v])
        depth = np.sum(weights * screen[..., 2], axis=1)
    front = (det < 0) & (weights >= 0).all(axis=1)
    assert front.any(), f"Missing source surface at ({x}, {y})"
    index = np.where(front, depth, np.inf).argmin()
    return index, weights[index], edges[index]


def sample(image, uv, lod, srgb=False):
    # Each normalized mip texel is the mean of its source texels in linear space.
    # The sampler in this GLB uses repeat wrapping and trilinear minification.
    def at_level(level):
        scale = 2 ** level
        size = np.array(image.shape[:2][::-1]) // scale
        position = uv * size - .5
        integer = np.floor(position).astype(int)
        fx, fy = position - integer
        result = np.zeros(3)
        for dy, wy in enumerate([1 - fy, fy]):
            for dx, wx in enumerate([1 - fx, fx]):
                x, y = (integer + [dx, dy]) % size * scale
                block = image[y:y + scale, x:x + scale] / 255.
                if srgb:
                    block = np.where(block <= .04045, block / 12.92, ((block + .055) / 1.055) ** 2.4)
                result += block.mean(axis=(0, 1)) * wx * wy
        return result

    lower = int(np.floor(lod))
    return at_level(lower) * (1 - (lod - lower)) + at_level(lower + 1) * (lod - lower)


def verify(source, directory):
    geometry, images = water_bottle(source)
    m = json.loads((directory / "manifest.json").read_text())
    if m["resolution"] != 256 or m["provenance"]["sourceSha256"] != SOURCE_SHA256:
        raise ValueError("Bake the original WaterBottle.glb at 256px with all four optional maps")
    atlas = {}
    for name in ("colorMap", "normalMap", "materialMap", "emissiveMap"):
        d = m["textures"][name]
        dtype = {"uint8": "u1", "float16": "<f2"}[d["dtype"]]
        atlas[name] = np.fromfile(directory / d["path"], dtype=dtype).reshape(d["height"], d["width"], 4)
    positions = geometry["POSITION"]
    hom = np.concatenate((positions, np.ones((*positions.shape[:2], 1))), axis=-1)
    transform = np.array(m["transform"]).reshape(4, 4)
    # Normals below use the default canonicalization's uniform scale and translation.
    np.testing.assert_allclose(transform[:3, :3], np.eye(3) * transform[0, 0])
    world = hom @ transform
    errors = {k: [] for k in ("baseColorLinear", "worldNormalDegrees", "surface8", "emissiveLinear")}
    for face_id, probes in PROBES.items():
        face = m["faces"][face_id]
        view = np.array(face["viewMatrix"]).reshape(4, 4)
        projection = np.array(face["projectionMatrix"]).reshape(4, 4)
        clip = world @ view @ projection
        screen = clip[..., :3] / clip[..., 3:]
        screen[..., :2] = (screen[..., :2] * [1, -1] + 1) * 128
        col, row = face["tile"]
        for x, y in probes:
            tri, bary, edges = visible_hit(screen, x, y)
            uv_tri = geometry["TEXCOORD_0"][tri]
            uv = bary @ uv_tri
            gradients = np.linalg.solve(edges, uv_tri[1:] - uv_tri[0])
            lod = max(0., np.log2(np.linalg.norm(gradients * images[0].shape[0], axis=1).max()))
            base, orm, tangent_normal, emissive = [
                sample(image, uv, lod, srgb=i in (0, 3)) for i, image in enumerate(images)]
            normal = normalized(bary @ geometry["NORMAL"][tri])
            tangent4 = bary @ geometry["TANGENT"][tri]
            tangent = normalized(tangent4[:3])
            bitangent = normalized(np.cross(normal, tangent)) * np.sign(tangent4[3])
            expected_normal = normalized((tangent_normal * 2 - 1) @ [tangent, bitangent, normal])
            baked = {k: v[(row + 1) * 256 - 1 - y, col * 256 + x].astype(float) for k, v in atlas.items()}
            assert baked["colorMap"][3] == 1, f"Missing Box coverage at {face_id} ({x}, {y})"
            actual_normal = normalized(baked["normalMap"][:3] * 2 - 1)
            differences = {
                "baseColorLinear": np.max(np.abs(baked["colorMap"][:3] - base)),
                "worldNormalDegrees": np.degrees(np.arccos(np.clip(actual_normal @ expected_normal, -1, 1))),
                "surface8": np.max(np.abs(baked["materialMap"][:3] - orm[[1, 2, 0]] * 255)),
                "emissiveLinear": np.max(np.abs(baked["emissiveMap"][:3] - emissive)),
            }
            for key, tolerance in [("baseColorLinear", .012), ("worldNormalDegrees", 1.5),
                                   ("surface8", 3.), ("emissiveLinear", .012)]:
                value = differences[key]
                assert value <= tolerance, f"{face_id} ({x}, {y}) {key}: {value} exceeds {tolerance}"
                errors[key].append(value)
    return {"backend": m["backend"], "probes": sum(map(len, PROBES.values())),
            "maxError": {k: float(max(v)) for k, v in errors.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("asset_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.asset_directory)))

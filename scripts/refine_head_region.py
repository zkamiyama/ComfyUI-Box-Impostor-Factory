#!/usr/bin/env python3
"""Prepare and inspect a standalone Hi3D head crop; never splice a body mesh.

The inference command remains scripts/hi3d_infer.py. This CPU-only helper clips
the original canonical coarse mesh at a world-Y plane, crops the corresponding
pixels from F's physical +Z source PNG, and records the actual upstream framing
and normalization code. Mapping a result uses the existing isotropic Y-height
fit, explicitly an approximate registration rather than an inverse of Hi3D.
"""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
METHOD = 'standalone-hi3d-head-region-v1'
UPSTREAM_FILES = {
    'voxelizer': 'sparc3d/ops/voxelize.py',
    'pipeline': 'sparc3d/pipelines/sparc3d_pipeline.py',
    'postprocess': 'sparc3d/sparcubes/postprocess.py',
    'imagePreparation': 'sparc3d/utils/image.py',
    'imageConditioner': 'sparc3d/models/conditioners/image.py',
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the pure geometry implementation without importing ComfyUI nodes/CUDA.
depth_mesh = load_module('head_region_depth_mesh', REPO / 'box_nodes/depth_mesh.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def contained(directory, relative):
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('An asset path escapes its directory.')
    return path


def mesh_arrays(data):
    """Read actual active-scene GLB triangles in world coordinates."""
    doc, binary = depth_mesh._decode(data)
    vertices, faces = [], []
    offset = 0
    for _, xyz, tri, transform in depth_mesh._primitives(doc, binary):
        world = xyz @ transform[:3, :3].T + transform[:3, 3]
        if np.linalg.det(transform[:3, :3]) < 0:
            tri = tri[:, [0, 2, 1]]
        vertices.append(world)
        faces.append(tri.astype('i8') + offset)
        offset += len(world)
    if not faces:
        raise ValueError('Expected an active static triangle mesh.')
    points, inverse = np.unique(np.concatenate(vertices), axis=0, return_inverse=True)
    triangles = inverse[np.concatenate(faces)]
    used, compact = np.unique(triangles, return_inverse=True)
    points, triangles = points[used], compact.reshape(-1, 3)
    if not np.isfinite(points).all() or np.any(np.ptp(points, axis=0) <= 0):
        raise ValueError('Expected finite mesh vertices with positive XYZ extents.')
    return points, triangles


def geometry_glb(vertices, triangles):
    """One neutral geometry-only primitive; no synthesized normals or colors."""
    xyz = np.ascontiguousarray(vertices, dtype='<f4')
    indices = np.ascontiguousarray(triangles, dtype='<u4')
    doc = {
        'asset': {'version': '2.0', 'generator': METHOD},
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1, 'material': 0}]}],
        'materials': [{'name': 'diagnostic-clay', 'pbrMetallicRoughness': {
            'baseColorFactor': [.65, .65, .65, 1.], 'metallicFactor': 0., 'roughnessFactor': 1.}}],
        'buffers': [{}],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': 0, 'byteLength': xyz.nbytes, 'target': 34962},
            {'buffer': 0, 'byteOffset': xyz.nbytes, 'byteLength': indices.nbytes, 'target': 34963}],
        'accessors': [
            {'bufferView': 0, 'componentType': 5126, 'type': 'VEC3', 'count': len(xyz),
             'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()},
            {'bufferView': 1, 'componentType': 5125, 'type': 'SCALAR', 'count': indices.size}],
    }
    return depth_mesh._encode(doc, bytearray(xyz.tobytes() + indices.tobytes()))


def boundary_statistics(vertices, triangles, cut_y=None):
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    result = {'boundaryEdges': len(boundary), 'nonmanifoldEdges': int(np.sum(counts > 2))}
    if cut_y is not None:
        tolerance = max(1., abs(cut_y)) * np.finfo('f4').eps * 2
        result['boundaryEdgesOnCutPlane'] = int(np.sum(
            np.all(abs(vertices[boundary, 1] - cut_y) <= tolerance, axis=1)))
    return result


def clip_above_y(vertices, triangles, cut_y):
    """Clip crossing triangles exactly; preserve winding and leave the cut open."""
    vertices = np.asarray(vertices, dtype='f8')
    triangles = np.asarray(triangles, dtype='i8')
    if not np.isfinite(cut_y) or not vertices[:, 1].min() < cut_y < vertices[:, 1].max():
        raise ValueError('The cut Y must lie strictly inside the source mesh Y extent.')
    points = list(vertices.copy())
    inside = vertices[:, 1] >= cut_y
    selected = inside[triangles]
    kept = triangles[selected.all(axis=1)].tolist()
    crossing = triangles[selected.any(axis=1) & ~selected.all(axis=1)]
    intersections = {}

    def intersection(a, b):
        if vertices[a, 1] == cut_y:
            return int(a)
        if vertices[b, 1] == cut_y:
            return int(b)
        edge = tuple(sorted((int(a), int(b))))
        if edge not in intersections:
            lo, hi = vertices[list(edge)]
            fraction = (cut_y - lo[1]) / (hi[1] - lo[1])
            point = lo + fraction * (hi - lo)
            point[1] = cut_y
            intersections[edge] = len(points)
            points.append(point)
        return intersections[edge]

    for face in crossing:
        polygon = []
        previous = face[-1]
        for current in face:
            if inside[current]:
                if not inside[previous]:
                    polygon.append(intersection(previous, current))
                polygon.append(int(current))
            elif inside[previous]:
                polygon.append(intersection(previous, current))
            previous = current
        polygon = [v for i, v in enumerate(polygon) if v != polygon[i - 1]]
        kept.extend([polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1))
    if not kept:
        raise ValueError('The crop contains no triangles.')
    points, faces = np.asarray(points), np.asarray(kept, dtype='i8')
    corners = points[faces]
    area = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    degenerate = area == 0
    faces = faces[~degenerate]
    used, inverse = np.unique(faces, return_inverse=True)
    points, faces = points[used], inverse.reshape(-1, 3)
    if not len(faces) or np.any(np.ptp(points, axis=0) <= 0):
        raise ValueError('The crop has empty or zero-extent geometry.')
    return points, faces, {
        'cutPlane': {'axis': 'canonical-world-Y', 'keep': 'Y >= cut', 'value': float(cut_y)},
        'sourceTriangles': len(triangles), 'fullyRetainedTriangles': int(selected.all(axis=1).sum()),
        'crossingSourceTriangles': len(crossing), 'intersectionVertices': len(intersections),
        'degenerateTrianglesDropped': int(degenerate.sum()), 'cropTriangles': len(faces),
        'cropVertices': len(points), 'syntheticCap': False,
        'bounds': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        'boundary': boundary_statistics(points, faces, cut_y),
    }


def camera_source_affine(face, canvas, source_size, registration):
    """Canonical XYZ1 -> original physical source PNG pixel-center XY1."""
    if (face.get('side') != [0, 0, 1] or face.get('up') != [0, 1, 0]
            or face.get('imageRowOrder') != 'top-to-bottom'):
        raise ValueError('Expected the canonical +Z, Y-up, top-origin camera.')
    if registration.get('scale') != 1. or registration.get('centerShiftYX') != [0., 0.]:
        raise ValueError('This experiment requires identity source silhouette registration.')
    if (canvas.get('version') != 1 or canvas.get('faceId') != '+Z'
            or canvas.get('imageRowOrder') != 'top-to-bottom'):
        raise ValueError('Expected a physical +Z source canvas transform.')
    if source_size[0] != source_size[1] or source_size[0] != canvas['size']:
        raise ValueError('Source PNG must retain the recorded square canvas resolution.')
    rect = canvas['guideRect']
    if len(rect) != 4 or any(type(x) is not int for x in rect):
        raise ValueError('Invalid source canvas guide rectangle.')
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) < 1 or x + w > source_size[0] or y + h > source_size[1]:
        raise ValueError('Source guide rectangle lies outside the PNG.')
    projection = np.asarray(face['projectionMatrix'], dtype='f8').reshape(4, 4).T
    view = np.asarray(face['viewMatrix'], dtype='f8').reshape(4, 4).T
    if (not np.isfinite(projection).all() or not np.isfinite(view).all()
            or not np.allclose(projection[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(view[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(view[:3, :3], np.eye(3), rtol=0, atol=1e-12)):
        raise ValueError('Only unrotated canonical +Z orthographic cameras are supported.')
    clip = projection @ view
    if (not np.allclose(clip[[0, 0, 1, 1], [1, 2, 0, 2]], 0, rtol=0, atol=1e-12)
            or clip[0, 0] <= 0 or clip[1, 1] <= 0):
        raise ValueError('Camera image axes have shear, depth dependence or a reflection.')
    physical = np.asarray(canvas['physicalWidthHeight'], dtype='f8')
    actual = np.array([2 / clip[0, 0], 2 / clip[1, 1]])
    if not np.allclose(physical, actual, rtol=0, atol=1e-9):
        raise ValueError('Physical canvas dimensions disagree with the native camera.')
    scale = canvas['size'] / physical.max()
    expected_size = np.rint(physical * scale).astype(int)
    if not np.array_equal(expected_size, [w, h]):
        raise ValueError('The guide is not the recorded physical-aspect letterbox.')
    pixel = np.array([[w / 2, 0, 0, x + (w - 1) / 2],
                      [0, -h / 2, 0, y + (h - 1) / 2], [0, 0, 0, 1.]])
    return pixel @ clip


def crop_reference(image, face, canvas, registration, crop_bounds, cut_y, pad_pixels=8):
    """Crop native source pixels, masking only samples below the same Y plane."""
    if image.mode != 'RGBA':
        raise ValueError('The physical source reference must be RGBA.')
    if type(pad_pixels) is not int or pad_pixels < 0:
        raise ValueError('Reference padding must be a nonnegative integer in original pixels.')
    affine = camera_source_affine(face, canvas, image.size, registration)
    cut_center = float((affine @ [0., cut_y, 0., 1.])[1])
    cut_edge = cut_center + .5
    source = np.asarray(image).copy()
    x, y, w, h = canvas['guideRect']
    keep = np.zeros(source.shape[:2], dtype=bool)
    keep[y:y + h, x:x + w] = True
    keep &= np.arange(source.shape[0])[:, None] <= cut_center
    source[~keep] = 0
    visible = np.argwhere(source[..., 3] > 0)
    if not len(visible):
        raise ValueError('No source subject pixels remain above the cut plane.')
    bounds = np.asarray(crop_bounds, dtype='f8')
    corners = np.array([[a, b, 0., 1.] for a in bounds[:, 0] for b in bounds[:, 1]])
    projected = corners @ affine.T
    edge_min = projected[:, :2].min(axis=0) + .5
    edge_max = projected[:, :2].max(axis=0) + .5
    if (edge_min[0] < x - 1 or edge_max[0] > x + w + 1
            or edge_min[1] < y - 1 or edge_max[1] > y + h + 1):
        raise ValueError('The mesh crop projects outside the physical source capture.')
    # Include both projected source geometry and the actual generated silhouette;
    # this does not force generated hair/skin onto the coarse silhouette.
    lower = np.minimum(edge_min, visible[:, ::-1].min(axis=0))
    upper = np.maximum(edge_max, visible[:, ::-1].max(axis=0) + 1)
    low = np.maximum(np.floor(lower).astype(int) - pad_pixels, [x, y])
    high = np.minimum(np.ceil(upper).astype(int) + pad_pixels, [x + w, y + h])
    box = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
    result = Image.fromarray(source[box[1]:box[3], box[0]:box[2]])
    retained = np.argwhere(np.asarray(result)[..., 3] > 0)
    bbox = result.getchannel('A').getbbox()
    pixels_per_world = [float(affine[0, 0]), float(-affine[1, 1])]
    proof = {
        'sourceSizePixels': list(image.size), 'sourcePNGMode': image.mode,
        'sourceGuideRectXYWH': canvas['guideRect'], 'sourceCropRectXYXY': box,
        'cropSizePixels': list(result.size), 'subjectBoundsInCropXYXY': list(bbox),
        'subjectSizePixels': [bbox[2] - bbox[0], bbox[3] - bbox[1]],
        'retainedSubjectPixels': len(retained), 'sourcePixelYAtCutEdge': cut_edge,
        'canonicalWorldXYZ1ToSourcePixelXY1RowMajor': affine.tolist(),
        'sourcePixelsPerCanonicalUnitXY': pixels_per_world,
        'sourceLetterboxRoundingAnisotropy': pixels_per_world[0] / pixels_per_world[1],
        'pixelCenters': 'integer indices; rows top-to-bottom',
        'transparentMarginOriginalPixelsRequested': pad_pixels,
        'resampled': False, 'newImageInference': False,
        'maskPolicy': 'retain original RGBA pixels inside guideRect and above canonical cut; zero all others',
        'detailPolicy': 'native crop of existing full-body source; no new head pixels generated',
    }
    return result, proof


def _constant(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing expected upstream constant: ' + name)


def _requires_assignment(tree, target, expression):
    expected = ast.dump(ast.parse(expression, mode='eval').body, include_attributes=False)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(ast.unparse(t) == target for t in node.targets)
                and ast.dump(node.value, include_attributes=False) == expected):
            return
    raise ValueError('Unrecognized upstream normalization assignment: ' + target)


def upstream_contract(root):
    """Read/hash installed sources; import only their PIL-only image preparation."""
    root = Path(root).resolve()
    files = {key: root / relative for key, relative in UPSTREAM_FILES.items()}
    trees = {key: ast.parse(path.read_text()) for key, path in files.items()}
    voxel = trees['voxelizer']
    pad = _constant(voxel, 'DEFAULT_PAD')
    init = next(node for node in ast.walk(voxel)
                if isinstance(node, ast.ClassDef) and node.name == 'MeshVoxelizer')
    init = next(node for node in init.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    defaults = dict(zip([arg.arg for arg in init.args.args[-len(init.args.defaults):]], init.args.defaults))
    eps = ast.literal_eval(defaults['eps'])
    _requires_assignment(voxel, 'self.expanded_min', 'self.vertices_min - self.eps')
    _requires_assignment(voxel, 'self.expanded_max', 'self.vertices_max + self.eps')
    _requires_assignment(voxel, 'self.center', '(self.expanded_min + self.expanded_max) / 2.0')
    _requires_assignment(voxel, 'max_range', '(self.expanded_max - self.expanded_min).max()')
    _requires_assignment(voxel, 'self.scale', '(self.resolution - self.pad * 2 - 2 * self.eps) / max_range')
    _requires_assignment(trees['postprocess'], 'vertices', 'vertices - vertices.mean(axis=0, keepdims=True)')
    _requires_assignment(trees['postprocess'], 'max_radius', 'float(np.linalg.norm(vertices, axis=1).max())')
    _requires_assignment(trees['postprocess'], 'vertices', 'vertices / max_radius')
    structures = _constant(trees['pipeline'], 'STRUCTURE_RESOLUTIONS')
    resolutions = _constant(trees['pipeline'], 'OUTPUT_RESOLUTIONS')
    budgets = _constant(trees['pipeline'], 'PASS_BUDGETS')
    if structures != (64, 128, 192) or resolutions != (512, 1024, 1536):
        raise ValueError('Unrecognized installed Hi3D refinement ladder.')
    preparation = load_module('head_region_upstream_image', files['imagePreparation'])
    margin = _constant(trees['imagePreparation'], 'SUBJECT_PADDING')
    return {
        'root': str(root), 'sourceFiles': {key: {'path': str(path), 'sha256': digest(path)}
                                        for key, path in files.items()},
        'initialVoxelizer': {'requestedResolution': structures[0], 'pad': pad, 'eps': eps,
                            'tokenBudget': budgets[0]},
        'structureResolutions': list(structures[:2]), 'geometryExtractionResolutions': list(resolutions[:2]),
        'imageSubjectPaddingFactor': margin,
        'imageEncoderInputPixels': {
            'DINO': _constant(trees['imageConditioner'], '_DINO_SIZE'),
            'CLIP': _constant(trees['imageConditioner'], '_CLIP_SIZE')},
        'betweenPasses': 'fresh bbox-center/uniform-scale adaptive voxelization of the preceding generated mesh',
        'finalNormalization': 'subtract final vertex mean; divide by maximum radial norm about that mean',
        'exactInverseAvailableFromExportedGLB': False,
        'reason': 'inference exports only final normalized geometry, without per-pass transforms or final mean/radius',
    }, preparation.prepare_image


def nominal_voxel_mapping(vertices, specification):
    """Float32 reproduction of the requested first-grid map; not observed CUDA voxels."""
    points = np.asarray(vertices, dtype='f4')
    eps = np.float32(specification['eps'])
    low, high = points.min(axis=0) - eps, points.max(axis=0) + eps
    center = (low + high) / np.float32(2.)
    resolution, pad = specification['requestedResolution'], specification['pad']
    scale = np.float32(resolution - pad * 2 - 2 * specification['eps']) / (high - low).max()
    matrix = np.eye(4)
    matrix[:3, :3] *= float(scale)
    matrix[:3, 3] = np.float32(resolution / 2.) - center * scale
    return {
        **specification, 'expandedBoundsFloat32': [low.tolist(), high.tolist()],
        'centerFloat32': center.tolist(), 'scaleFloat32': float(scale),
        'canonicalWorldToNominalVoxelRowMajor': matrix.tolist(),
        'nominalVoxelToCanonicalWorldRowMajor': np.linalg.inv(matrix).tolist(),
        'observedVoxelization': False,
        'limitation': 'adaptive_vox may shrink the actual first grid to meet its token budget; later passes renormalize',
    }


def prepare(h_plan_path, output, cut_y, upstream_root, pad_pixels=8):
    h_plan_path, output = Path(h_plan_path).resolve(), Path(output).resolve()
    h = read_json(h_plan_path)
    coarse_dir, appearance_dir = Path(h['coarseDirectory']), Path(h['appearanceDirectory'])
    coarse_path, source_image = Path(h['coarseMesh']), Path(h['reference'])
    for path, expected in [(coarse_path, h['coarseSha256']), (source_image, h['referencePngSha256']),
                           (Path(h['inferScript']), h['inferScriptSha256'])]:
        if digest(path) != expected:
            raise ValueError('Frozen H input changed: ' + str(path))
    for source_dir in (coarse_dir, appearance_dir):
        if output.is_relative_to(source_dir.resolve()):
            raise ValueError('Head experiment output must be separate from its source assets.')
    if output.exists():
        raise ValueError('Use a new experiment output directory; existing work is immutable.')
    coarse = read_json(coarse_dir / 'manifest.json')
    appearance = read_json(appearance_dir / 'manifest.json')
    for key in ('transform', 'inverseTransform', 'coordinateSystem'):
        if coarse[key] != appearance[key]:
            raise ValueError('Coarse and appearance canonical frames differ: ' + key)
    if (coarse['coordinateSystem'] != 'right-handed-y-up'
            or coarse['proxySha256'] != h['coarseSha256']
            or contained(coarse_dir, coarse['proxy']) != coarse_path.resolve()):
        raise ValueError('Unexpected coarse canonical geometry identity.')
    face = appearance['faces']['+Z']
    if (face['edit']['status'] != 'accepted'
            or contained(appearance_dir, face['edit']['view']) != source_image.resolve()):
        raise ValueError('Unexpected F +Z reference source.')
    canvas = face['edit']['provenance']['canvas']
    if canvas != h['referenceCanvas']:
        raise ValueError('Reference canvas changed since the H plan.')
    qa_path = contained(appearance_dir, appearance['qa'])
    qa = read_json(qa_path)['faces']['+Z']
    if qa['status'] != 'accepted':
        raise ValueError('Reference face QA is not accepted.')
    source_vertices, source_faces = mesh_arrays(coarse_path.read_bytes())
    crop_vertices, crop_faces, clipping = clip_above_y(source_vertices, source_faces, cut_y)
    head_data = geometry_glb(crop_vertices, crop_faces)
    # Metadata and mapping use the exact float32 vertices that inference will read.
    crop_vertices, crop_faces = mesh_arrays(head_data)
    clipping['storedFloat32Bounds'] = [crop_vertices.min(axis=0).tolist(), crop_vertices.max(axis=0).tolist()]
    contract, image_preparation = upstream_contract(upstream_root)
    with Image.open(source_image) as original:
        original.load()
        reference, crop_proof = crop_reference(original, face, canvas, qa['registration'],
                                               clipping['bounds'], cut_y, pad_pixels)
        context = original.convert('RGB')
    prepared = image_preparation(reference)
    crop_proof.update(upstreamPreparedSizePixels=list(prepared.size),
                      upstreamPreparedMode=prepared.mode,
                      upstreamSubjectPaddingFactor=contract['imageSubjectPaddingFactor'])
    draw = ImageDraw.Draw(context)
    draw.rectangle(crop_proof['sourceCropRectXYXY'], outline=(255, 40, 40), width=2)
    cut_pixel = crop_proof['sourcePixelYAtCutEdge']
    draw.line([(0, cut_pixel), (context.width - 1, cut_pixel)], fill=(255, 210, 0), width=1)
    draw.text((8, 8), f"Original {original.width} x {original.height}; crop {reference.width} x {reference.height}",
              fill=(255, 210, 0), stroke_width=1, stroke_fill=(0, 0, 0))
    blobs = {'head-source.glb': head_data, 'head-reference.png': png_bytes(reference),
             'head-reference-prepared.png': png_bytes(prepared), 'head-reference-context.png': png_bytes(context)}
    paths = {name: str(output / name) for name in blobs}
    raw_path = output / 'hi3d-raw.glb'
    report_path = output / 'hi3d-inference.json'
    executable = h['executionEnvironment']['pythonExecutable']
    inference_argv = [executable, h['inferScript'], '--source', paths['head-source.glb'],
                      '--reference', paths['head-reference.png'], '--output', str(raw_path),
                      '--report', str(report_path), '--resolution', '1024', '--steps', '25', '--offload']
    plan = {
        'method': METHOD, 'stage': 'prepared-only', 'parentHPlan': str(h_plan_path),
        'parentHPlanSha256': digest(h_plan_path), 'outputDirectory': str(output),
        'sourceCoarse': {'path': str(coarse_path), 'sha256': digest(coarse_path),
                         'manifest': str(coarse_dir / 'manifest.json'),
                         'manifestSha256': digest(coarse_dir / 'manifest.json'),
                         'asset': coarse['asset_id'], 'bounds': [source_vertices.min(axis=0).tolist(),
                                                               source_vertices.max(axis=0).tolist()]},
        'sourceReference': {'path': str(source_image), 'sha256': digest(source_image),
                            'manifest': str(appearance_dir / 'manifest.json'),
                            'manifestSha256': digest(appearance_dir / 'manifest.json'),
                            'qa': str(qa_path), 'qaSha256': digest(qa_path),
                            'asset': appearance['asset_id'], 'face': '+Z',
                            'registration': qa['registration'], 'canvas': canvas},
        'canonicalFrame': {key: coarse[key] for key in ('transform', 'inverseTransform', 'coordinateSystem')},
        'crop': clipping, 'referenceCrop': crop_proof, 'upstream': contract,
        'nominalFirstVoxelMapping': nominal_voxel_mapping(crop_vertices, contract['initialVoxelizer']),
        'nominalLongestSpanAllocationRatio': float(np.ptp(source_vertices, axis=0).max() /
                                                  np.ptp(crop_vertices, axis=0).max()),
        'artifacts': {name: {'path': paths[name], 'sha256': hashlib.sha256(blob).hexdigest(),
                            'bytes': len(blob)} for name, blob in blobs.items()},
        'inference': {'argv': inference_argv, 'scriptSha256': h['inferScriptSha256'],
                      'resolution': 1024, 'steps': 25, 'seed': 0, 'offload': True,
                      'rawOutput': str(raw_path), 'report': str(report_path), 'hasRun': False,
                      'environmentPolicy': 'reuse verified H/B model and offline environment; move all caches/temp to F volume'},
        'geometryPolicy': 'standalone geometry-only input and QA output; no synthesized normals, vertex colors or neck cap',
        'mappingPolicy': 'approximate isotropic Y-height and bbox-center fit only; never an exact inverse of upstream normalization',
        'fullBodyMergeAllowed': False,
        'fullBodyPolicy': 'source/body assets stay untouched; standalone result must be visually assessed before any new splice design',
    }
    output.mkdir(parents=True)
    for name, data in blobs.items():
        (output / name).write_bytes(data)
    (output / 'head-plan.json').write_bytes(json_bytes(plan))
    return plan


def fit_isotropic_y(source_vertices, raw_vertices):
    source = np.asarray(source_vertices, dtype='f8')
    raw = np.asarray(raw_vertices, dtype='f8')
    if not np.isfinite(source).all() or not np.isfinite(raw).all():
        raise ValueError('Cannot fit non-finite vertices.')
    source_bounds = np.array([source.min(axis=0), source.max(axis=0)])
    raw_bounds = np.array([raw.min(axis=0), raw.max(axis=0)])
    source_extent, raw_extent = np.diff(source_bounds, axis=0)[0], np.diff(raw_bounds, axis=0)[0]
    if np.any(source_extent <= 0) or np.any(raw_extent <= 0):
        raise ValueError('Cannot fit geometry with a zero-length axis.')
    scale = float(source_extent[1] / raw_extent[1])
    translation = source_bounds.mean(axis=0) - raw_bounds.mean(axis=0) * scale
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = translation
    mapped = raw * scale + translation
    ratios = np.ptp(mapped, axis=0) / source_extent
    return mapped, {
        'method': 'isotropic-y-height-bbox-center-fit', 'exactInverse': False,
        'rotationEstimated': False, 'axisSpecificStretch': False,
        'sourceBounds': source_bounds.tolist(), 'rawBounds': raw_bounds.tolist(),
        'scale': scale, 'translation': translation.tolist(),
        'rawToCanonicalWorldRowMajor': matrix.tolist(),
        'canonicalWorldToRawRowMajor': np.linalg.inv(matrix).tolist(),
        'extentRatioXYZ': ratios.tolist(),
        'extentResidualXYZ': (np.ptp(mapped, axis=0) - source_extent).tolist(),
        'warning': 'bbox registration can shift anatomy when generated neck/hair bounds differ; not approved for splicing',
    }


def map_result(plan_path, raw_path=None):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if plan.get('method') != METHOD or plan.get('fullBodyMergeAllowed') is not False:
        raise ValueError('Expected an isolated, non-splicing head experiment plan.')
    directory = Path(plan['outputDirectory']).resolve()
    if directory != plan_path.parent:
        raise ValueError('The plan has moved from its recorded output directory.')
    for item in plan['artifacts'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Prepared head experiment input changed: ' + item['path'])
    for item in (plan['sourceCoarse'], plan['sourceReference']):
        if digest(item['path']) != item['sha256']:
            raise ValueError('Original source input changed: ' + item['path'])
    for item in plan['upstream']['sourceFiles'].values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Installed upstream source changed after preparation: ' + item['path'])
    raw_path = Path(raw_path or plan['inference']['rawOutput']).resolve()
    output, report_path = directory / 'head-canonical.glb', directory / 'head-mapping.json'
    if output.exists() or report_path.exists():
        raise ValueError('Mapped result already exists; preserve the experiment unchanged.')
    source, _ = mesh_arrays(Path(plan['artifacts']['head-source.glb']['path']).read_bytes())
    raw, triangles = mesh_arrays(raw_path.read_bytes())
    mapped, mapping = fit_isotropic_y(source, raw)
    data = geometry_glb(mapped, triangles)
    actual, actual_triangles = mesh_arrays(data)
    # This is diagnostic nearest-vertex distance, not a surface/landmark accuracy score.
    distance, _ = cKDTree(source).query(actual, k=1, workers=1)
    source_distance, _ = cKDTree(actual).query(source, k=1, workers=1)
    raw_mean = raw.mean(axis=0)
    report = {
        'method': METHOD, 'stage': 'standalone-result-mapped', 'plan': str(plan_path),
        'planSha256': digest(plan_path), 'rawMesh': str(raw_path), 'rawMeshSha256': digest(raw_path),
        'outputMesh': str(output), 'outputMeshSha256': hashlib.sha256(data).hexdigest(),
        'outputVertices': len(actual), 'outputTriangles': len(actual_triangles),
        'sourceHeadVertices': len(source), 'rawVertices': len(raw), 'rawTriangles': len(triangles),
        'mapping': mapping,
        'rawExportNormalizationMeasurements': {'meanXYZ': raw_mean.tolist(),
            'maxRadiusAboutMean': float(np.linalg.norm(raw - raw_mean, axis=1).max()),
            'measurementDoesNotRecoverPreNormalizationMeanOrScale': True},
        'storedFloat32Bounds': [actual.min(axis=0).tolist(), actual.max(axis=0).tolist()],
        'nearestVertexDistanceDiagnostic': {
            'mappedToSourceP50P95Max': np.quantile(distance, [.5, .95, 1]).tolist(),
            'sourceToMappedP50P95Max': np.quantile(source_distance, [.5, .95, 1]).tolist(),
            'normalizedBySourceYSpan': float(np.ptp(source, axis=0)[1]),
            'notSurfaceOrLandmarkAccuracy': True},
        'outputBoundary': boundary_statistics(actual, actual_triangles),
        'inputSourceHashesRechecked': True, 'fullBodyMergeAllowed': False,
        'assessment': 'not visually evaluated; geometry and face correspondence must be judged from the standalone output',
        'normalPolicy': 'no NORMAL accessor synthesized in either standalone source or result; identical clay material',
    }
    output.write_bytes(data)
    report_path.write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    preparing = stages.add_parser('prepare', help='CPU preparation only; never starts inference')
    preparing.add_argument('--h-plan', type=Path, required=True)
    preparing.add_argument('--output', type=Path, required=True)
    preparing.add_argument('--cut-y', type=float, required=True)
    preparing.add_argument('--upstream-root', type=Path, required=True)
    preparing.add_argument('--reference-pad-pixels', type=int, default=8)
    mapping = stages.add_parser('map', help='map a completed standalone head result; never merge a body')
    mapping.add_argument('--plan', type=Path, required=True)
    mapping.add_argument('--raw', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.h_plan, args.output, args.cut_y, args.upstream_root, args.reference_pad_pixels)
        print(json.dumps({'stage': result['stage'], 'plan': str(args.output / 'head-plan.json'),
                          'crop': result['crop'], 'referenceCrop': result['referenceCrop'],
                          'inferenceArgv': result['inference']['argv']}, indent=2, allow_nan=False))
    else:
        print(json.dumps(map_result(args.plan, args.raw), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

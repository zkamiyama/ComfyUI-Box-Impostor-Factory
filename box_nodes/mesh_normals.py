"""Shared CPU geometric normals; no position, index, UV, color, or material edits."""
import numpy as np

from . import depth_mesh

NORMALS_VERSION = 'area-weighted-unit-v1'


def mesh_arrays(data):
    doc, binary = depth_mesh._decode(data)
    primitives = list(depth_mesh._primitives(doc, binary))
    positions = [p @ transform[:3, :3].T+transform[:3, 3] for _, p, _, transform in primitives]
    unique, inverse = np.unique(np.concatenate(positions), axis=0, return_inverse=True)
    offsets = np.cumsum([0, *(len(p) for p in positions)])
    triangles = np.concatenate([inverse[offsets[i]+tri] for i, (_, _, tri, _) in enumerate(primitives)])
    return unique, triangles.astype(np.int32)


def vertex_normals(points, triangles, *, return_report=False):
    """Area-weighted normals with deterministic valid-unit fallbacks."""
    normal = np.zeros_like(points, dtype='f8')
    a, b, c = points[triangles].transpose(1, 0, 2)
    area = np.cross(b-a, c-a)
    face_length = np.linalg.norm(area, axis=1)
    largest = np.zeros(len(points), dtype='f8')
    for corners in triangles.T:
        np.add.at(normal, corners, area)
        np.maximum.at(largest, corners, face_length)
    lengths = np.linalg.norm(normal, axis=1)
    cancelled = (largest > 0) & (lengths <= largest * 1e-12)
    incident = np.bincount(triangles.ravel(), minlength=len(points))
    unused = incident == 0
    degenerate_only = (incident > 0) & (largest == 0)
    if cancelled.any():
        # Largest-area face wins; the first face index breaks equal-area ties.
        winner = np.full(len(points), len(triangles), dtype=np.int64)
        ids = np.arange(len(triangles))
        for corners in triangles.T:
            candidates = cancelled[corners] & (face_length == largest[corners]) & (face_length > 0)
            np.minimum.at(winner, corners[candidates], ids[candidates])
        normal[cancelled] = area[winner[cancelled]]
    normal[unused | degenerate_only] = [0., 1., 0.]
    result = depth_mesh._unit(normal).astype('f4')
    report = {
        'version':NORMALS_VERSION,
        'algorithm':'area-weighted; largest-area incident face for cancellation; +Y for unrendered vertices',
        'vertices':len(points), 'triangles':len(triangles),
        'cancellationFallbackVertices':int(cancelled.sum()),
        'unusedPlaceholderVertices':int(unused.sum()),
        'degenerateOnlyPlaceholderVertices':int(degenerate_only.sum()),
        'degenerateTriangles':int(np.count_nonzero(face_length == 0)),
        'relativeCancellationThreshold':1e-12, 'unitPlaceholder':[0, 1, 0]}
    return (result, report) if return_report else result


def geometry_normals(data, *, return_report=False):
    """Rebuild vertex normals from the changed shared surface, preserving UV/materials."""
    points,triangles = mesh_arrays(data)
    normals,normal_report = vertex_normals(points,triangles,return_report=True)
    doc,binary = depth_mesh._decode(data)
    original_binary = binary
    binary = bytearray(binary)
    primitives = list(depth_mesh._primitives(doc,original_binary))
    positions = [p@t[:3,:3].T+t[:3,3] for _,p,_,t in primitives]
    _,inverse = np.unique(np.concatenate(positions),axis=0,return_inverse=True)
    offsets = np.cumsum([0,*(len(p) for p in positions)])
    for i,(primitive,xyz,tri,transform) in enumerate(primitives):
        local = depth_mesh._unit(normals[inverse[offsets[i]:offsets[i+1]]]@transform[:3,:3])
        if 'NORMAL' not in primitive['attributes']:
            # POSITION has the same count and float32 VEC3 layout; the writer allocates new storage.
            primitive['attributes']['NORMAL'] = primitive['attributes']['POSITION']
        depth_mesh._write_attribute(doc,binary,primitive,'NORMAL',local)
        if 'TANGENT' in primitive['attributes']:
            tangents = depth_mesh._accessor(doc,original_binary,primitive['attributes']['TANGENT'],'VEC4',{5126}).astype('f8')
            tangent = tangents[:,:3]-local*np.sum(local*tangents[:,:3],axis=1,keepdims=True)
            depth_mesh._write_attribute(doc,binary,primitive,'TANGENT',
                                       np.column_stack((depth_mesh._unit(tangent),tangents[:,3])))
    encoded = depth_mesh._encode(doc,binary)
    return (encoded, normal_report) if return_report else encoded

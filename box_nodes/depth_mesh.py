"""Deform static GLB triangles in scene coordinates without re-encoding their materials or UVs."""
import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation


def _decode(data):
    if len(data) < 28 or struct.unpack_from('<4sII', data) != (b'glTF', 2, len(data)):
        raise ValueError('Expected a complete binary glTF 2.0 mesh.')
    chunks = []
    pos = 12
    while pos + 8 <= len(data):
        length, kind = struct.unpack_from('<I4s', data, pos)
        pos += 8
        if length % 4 or pos + length > len(data):
            raise ValueError('Invalid GLB chunk length.')
        chunks.append((kind, data[pos:pos+length]))
        pos += length
    if pos != len(data) or [c[0] for c in chunks] != [b'JSON', b'BIN\0']:
        raise ValueError('Depth deformation requires embedded JSON and BIN chunks.')
    doc = json.loads(chunks[0][1])
    buffers = doc.get('buffers', [])
    if len(buffers) != 1 or 'uri' in buffers[0] or not 0 <= len(chunks[1][1])-buffers[0]['byteLength'] <= 3:
        raise ValueError('Expected one embedded GLB geometry buffer.')
    return doc, chunks[1][1][:buffers[0]['byteLength']]


def _encode(doc, binary):
    doc['buffers'][0]['byteLength'] = len(binary)
    meta = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode('utf-8')
    meta += b' ' * (-len(meta) % 4)
    binary.extend(bytes(-len(binary) % 4))
    return (struct.pack('<4sIII4s', b'glTF', 2, 28 + len(meta) + len(binary), len(meta), b'JSON')
            + meta + struct.pack('<I4s', len(binary), b'BIN\0') + binary)


def _accessor(doc, binary, index, kind, allowed):
    a = doc['accessors'][index]
    if (a['componentType'] not in allowed or a['type'] != kind or a.get('sparse') or
            a.get('normalized') or 'bufferView' not in a):
        raise ValueError('Unsupported sparse, normalized or quantized geometry accessor.')
    v = doc['bufferViews'][a['bufferView']]
    if v.get('buffer', 0) != 0 or v.get('extensions'):
        raise ValueError('Geometry must use an uncompressed embedded buffer.')
    dtype = np.dtype({5126: '<f4', 5121: 'u1', 5123: '<u2', 5125: '<u4'}[a['componentType']])
    dims = {'VEC3': 3, 'VEC4': 4, 'SCALAR': 1}[kind]
    count = a['count']
    offset, start = a.get('byteOffset', 0), v.get('byteOffset', 0)
    stride = v.get('byteStride', dims * dtype.itemsize)
    size = dims * dtype.itemsize
    if (count < 1 or min(offset, start) < 0 or stride < size or stride % dtype.itemsize
            or offset + (count-1)*stride + size > v['byteLength'] or start+v['byteLength'] > len(binary)):
        raise ValueError('Invalid geometry buffer bounds or stride.')
    result = np.ndarray((count, dims), dtype, buffer=binary, offset=start+offset, strides=(stride, dtype.itemsize))
    if not np.isfinite(result).all():
        raise ValueError('Mesh geometry has non-finite values.')
    return result


def _matrix(node):
    if 'matrix' in node:
        value = np.array(node['matrix'], dtype='f8').reshape(4, 4).T
    else:
        value = np.eye(4)
        value[:3, :3] = Rotation.from_quat(node.get('rotation', [0, 0, 0, 1])).as_matrix()
        value[:3, :3] *= np.asarray(node.get('scale', [1, 1, 1]))[None, :]
        value[:3, 3] = node.get('translation', [0, 0, 0])
    if (not np.isfinite(value).all() or not np.allclose(value[3], [0, 0, 0, 1]) or
            np.linalg.matrix_rank(value[:3, :3]) < 3):
        raise ValueError('Mesh nodes require finite invertible affine transforms.')
    return value


def _primitives(doc, binary):
    excluded = {'KHR_mesh_quantization', 'KHR_draco_mesh_compression', 'EXT_meshopt_compression',
                'EXT_mesh_gpu_instancing', 'KHR_meshopt_compression'}
    if (doc.get('skins') or doc.get('animations') or excluded.intersection(
            [*doc.get('extensionsUsed', []), *doc.get('extensionsRequired', [])])):
        raise ValueError('Depth deformation requires static, uncompressed, non-quantized triangles.')
    visited = set()
    stack = [(idx, np.eye(4)) for idx in doc['scenes'][doc.get('scene', 0)].get('nodes', [])]
    while stack:
        idx, parent = stack.pop()
        if idx in visited:
            raise ValueError('Repeated node or cycle in the active mesh scene.')
        visited.add(idx)
        node = doc['nodes'][idx]
        if excluded.intersection(node.get('extensions', {})):
            raise ValueError('Depth deformation does not support instanced or compressed geometry nodes.')
        matrix = parent @ _matrix(node)
        stack.extend((child, matrix) for child in node.get('children', []))
        if 'mesh' not in node:
            continue
        mesh_id = node['mesh']
        if sum(n.get('mesh') == mesh_id for n in doc['nodes']) > 1:
            raise ValueError('Instanced meshes require separate geometry for each copy before deformation.')
        for prim in doc['meshes'][mesh_id]['primitives']:
            attrs = prim['attributes']
            if ('skin' in node or prim.get('targets') or prim.get('mode', 4) != 4
                    or 'POSITION' not in attrs or any(k.startswith(('JOINTS_', 'WEIGHTS_')) for k in attrs)
                    or excluded.intersection(prim.get('extensions', {}))):
                raise ValueError('Only static triangle primitives without skin or morph targets are supported.')
            xyz = _accessor(doc, binary, attrs['POSITION'], 'VEC3', {5126}).astype('f8')
            idxs = (_accessor(doc, binary, prim['indices'], 'SCALAR', {5121, 5123, 5125}).ravel()
                    if 'indices' in prim else np.arange(len(xyz)))
            if not len(idxs) or len(idxs) % 3 or np.max(idxs) >= len(xyz):
                raise ValueError('Invalid triangle indices.')
            yield prim, xyz, idxs.reshape(-1, 3), matrix


def _unit(vectors):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


def _frames(xyz, triangles):
    a, b, c = xyz[triangles].transpose(1, 0, 2)
    u = _unit(b-a)
    w = np.cross(b-a, c-a)
    n = _unit(w)
    return np.stack((u, np.cross(n, u), n), axis=2), np.linalg.norm(w, axis=1)


def _transport(original, triangles, rotations, areas, fallback):
    # Transport the authored shading vectors: each face contributes its local
    # rotation, weighted by its original area. UV seams and hard normals survive.
    total = np.zeros_like(original, dtype='f8')
    for corners in triangles.T:
        rotated = np.einsum('ijk,ik->ij', rotations, original[corners])
        np.add.at(total, corners, rotated * areas[:, None])
    valid = np.linalg.norm(total, axis=1) > 1e-16
    out = fallback.astype('f8').copy()
    out[valid] = _unit(total[valid])
    return out


def _write_attribute(doc, binary, prim, name, data):
    old = doc['accessors'][prim['attributes'][name]]
    binary.extend(bytes(-len(binary) % 4))
    samples = np.ascontiguousarray(data, dtype='<f4')
    offset = len(binary)
    binary.extend(samples.tobytes())
    views, accessors = doc['bufferViews'], doc['accessors']
    desc = {k: v for k, v in old.items() if k not in ('byteOffset', 'bufferView', 'min', 'max')}
    desc['bufferView'] = len(views)
    if name == 'POSITION' or 'min' in old or 'max' in old:
        desc.update(min=samples.min(axis=0).tolist(), max=samples.max(axis=0).tolist())
    views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': samples.nbytes, 'target': 34962})
    prim['attributes'][name] = len(accessors)
    accessors.append(desc)


def deform_glb(data, displacement_callback):
    """Move exact coincident vertices together in canonical scene coordinates and backtrack foldovers."""
    doc, original = _decode(data)
    primitives = list(_primitives(doc, original))
    if not primitives:
        raise ValueError('The active scene has no static triangle mesh.')
    positions = [xyz @ transform[:3, :3].T + transform[:3, 3] for _, xyz, _, transform in primitives]
    unique, inverse = np.unique(np.concatenate(positions), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    displacements = np.asarray(displacement_callback(unique.copy()), dtype='f8')
    if displacements.shape != unique.shape or not np.isfinite(displacements).all():
        raise ValueError('Displacement must return one finite world-space XYZ offset for each unique vertex.')
    offsets = np.cumsum([0, *(len(x) for x in positions)])
    triangles = np.concatenate([inverse[offsets[i] + tri] for i, (_, _, tri, _) in enumerate(primitives)])
    def area_vectors(points):
        a, b, c = points[triangles].transpose(1, 0, 2)
        return np.cross(b-a, c-a)
    old = area_vectors(unique)
    area2 = np.einsum('ij,ij->i', old, old)
    valid = area2 > 0
    lengths = np.linalg.norm(displacements, axis=1)
    stats = dict(uniquePositions=len(unique), vertices=len(inverse), triangles=len(triangles),
                 originalDegenerateTriangles=int((~valid).sum()), stepScale=0., maxDisplacement=0.,
                 meanDisplacement=0., movedVertices=0)
    if not lengths.any():
        return data, stats
    if not valid.any():
        raise ValueError('Cannot deform a mesh containing only degenerate triangles.')
    # Test the final float32 positions after converting back to the original
    # local coordinate system; triangles can collapse at storage precision.
    scale = 1.
    for _ in range(25):
        changed = []
        crosses = []
        for i, (_, xyz, tri, transform) in enumerate(primitives):
            delta = displacements[inverse[offsets[i]:offsets[i+1]]] * scale
            moved = (xyz + np.linalg.solve(transform[:3, :3], delta.T).T).astype('<f4')
            changed.append(moved)
            local_edges = moved[tri].astype('f8') @ transform[:3, :3].T
            crosses.append(np.cross(local_edges[:,1]-local_edges[:,0], local_edges[:,2]-local_edges[:,0]))
        dot = np.einsum('ij,ij->i', old, np.concatenate(crosses))
        if np.all(dot[valid] > area2[valid] * 1e-7):
            break
        scale *= .5
    else:
        raise ValueError('Depth deformation would fold or collapse a triangle at float32 precision.')
    # Report the movements that survived conversion to the GLB accessor precision.
    actual = np.zeros(len(unique))
    for i, ((_, xyz, _, transform), moved) in enumerate(zip(primitives, changed)):
        distances = np.linalg.norm((moved - xyz) @ transform[:3, :3].T, axis=1)
        np.maximum.at(actual, inverse[offsets[i]:offsets[i+1]], distances)
    if not actual.any():
        return data, stats
    binary = bytearray(original)
    for (primitive, xyz, tri, _), moved in zip(primitives, changed):
        attrs = primitive['attributes']
        if 'NORMAL' in attrs or 'TANGENT' in attrs:
            before, area = _frames(xyz, tri)
            after, _ = _frames(moved.astype('f8'), tri)
            rotations = after @ before.transpose(0, 2, 1)
            normal = None
            if 'NORMAL' in attrs:
                authored = _accessor(doc, original, attrs['NORMAL'], 'VEC3', {5126})
                if len(authored) != len(xyz):
                    raise ValueError('Normal and position counts differ.')
                normal = _transport(authored, tri, rotations, area, authored)
                _write_attribute(doc, binary, primitive, 'NORMAL', normal)
            if 'TANGENT' in attrs:
                tangents = _accessor(doc, original, attrs['TANGENT'], 'VEC4', {5126})
                if normal is None or len(tangents) != len(xyz):
                    raise ValueError('Tangents require a matching normal and position for each vertex.')
                tangent = _transport(tangents[:, :3], tri, rotations, area, tangents[:, :3])
                tangent -= normal * np.einsum('ij,ij->i', tangent, normal)[:, None]
                # Old tangent handedness remains valid because the accepted step preserves orientation.
                tangent = np.column_stack((_unit(tangent), tangents[:, 3]))
                _write_attribute(doc, binary, primitive, 'TANGENT', tangent)
        _write_attribute(doc, binary, primitive, 'POSITION', moved)
    stats.update(stepScale=scale, maxDisplacement=float(actual.max()),
                 meanDisplacement=float(actual.mean()), movedVertices=int(np.count_nonzero(actual)))
    return _encode(doc, binary), stats

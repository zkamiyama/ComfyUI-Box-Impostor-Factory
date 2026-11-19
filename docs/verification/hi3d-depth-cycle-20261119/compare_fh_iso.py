"""Build a fixed-ROI oblique F/H board from unchanged official Inspector captures."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_fh import CAMERA_KEYS, project, read
from build_box_control_front import BG, TEXT, MUTED, FONT, sha, backdrop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    work, output = args.work.resolve(), args.output.resolve()
    meta = {stage: read(work / f'{stage}-provenance.json') for stage in ('B', 'F', 'H')}
    assert meta['B']['proxySHA256'] == meta['F']['proxySHA256']
    assert meta['B']['clay']['sha256'] == meta['F']['clay']['sha256']
    assert meta['B']['clay']['geometry'] == meta['F']['clay']['geometry']
    assert meta['F']['clay']['material'] == meta['H']['clay']['material']
    manifests = {stage: read(Path(meta[stage]['assetDirectory']) / 'manifest.json') for stage in ('F', 'H')}
    for stage in ('F', 'H'):
        assert sha(Path(meta[stage]['assetDirectory']) / 'manifest.json') == meta[stage]['manifestSHA256']
    assert manifests['F']['transform'] == manifests['H']['transform']
    assert manifests['F']['inverseTransform'] == manifests['H']['inverseTransform']
    earlier = output / 'F-H-comparison-provenance.json'
    earlier_proof = read(earlier)
    assert earlier_proof['sameCanonicalTransform']
    assert all(row['sourcePNGAndNPYByteExact'] for row in earlier_proof['sourceRGB'].values())
    base = read(work / 'F-box/capture.json')
    camera = base['camera']
    assert camera['projection'] == 'orthographic' and camera['pixels'] == [1024, 1024]
    reference = Path(base['referenceManifest'])
    assert sha(reference) == base['referenceManifestSHA256']
    crop_source = output / 'A-full-comparison-provenance.json'
    head = np.asarray(read(crop_source)['headWorldBounds'], dtype='f8')
    assert np.array_equal(head, np.asarray(earlier_proof['headWorldBounds']))
    corners = np.array([[head[x, 0], head[y, 1], head[z, 2]]
                        for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    direction = np.ones(3) / np.sqrt(3)
    projected = project(corners, direction, camera)
    roi = [*np.floor(projected.min(0)).astype(int).tolist(),
           *np.ceil(projected.max(0)).astype(int).tolist()]
    assert all(0 <= value <= 1024 for value in roi)
    frames, records = {}, {}
    for stage in ('F', 'H'):
        frames[stage], records[stage] = {}, {}
        for kind in ('clay', 'box'):
            source_stage = 'B' if stage == 'F' and kind == 'clay' else stage
            source_meta = meta[source_stage]
            path = work / f'{source_stage}-{kind}-iso-resume/capture.json'
            cap = read(path)
            assert cap['errors'] == [] and 'SwiftShader' in cap['graphicsRenderer']
            assert cap['hardwareGPUUsed'] is False
            assert cap['backend'] == 'Three.js / WebGL2'
            assert cap['stageState'] == ('Native Box' if kind == 'box' else 'Three.js PBR')
            assert cap['referenceManifest'] == str(reference)
            assert cap['referenceManifestSHA256'] == base['referenceManifestSHA256']
            assert cap['referenceBounds'] == base['referenceBounds']
            assert cap['canvasPixels'] == [1024, 1024]
            assert all(cap['camera'][key] == camera[key] for key in CAMERA_KEYS)
            assert cap['htmlSHA256'] == source_meta[kind]['viewerSHA256'] == sha(cap['html'])
            if kind == 'clay':
                clay = source_meta['clay']
                assert sha(clay['path']) == clay['sha256']
                policy = clay['normalPreservation']
                assert policy['binaryChunksIdentical'] and policy['positionNormalIndexBindingsIdentical']
                assert not policy['normalRecomputed']
                triangles = clay['sourceTriangles']
                assert triangles == clay['payloadTriangles'] == cap['triangleCounts']['expected']
                assert triangles == cap['triangleCounts']['meshPayload']
                assert triangles == cap['triangleCounts']['sourceOriginal']['triangles']
                assert triangles == earlier_proof['captures']['clay'][stage]['front']['sourceTriangles']
            else:
                assert cap['assetId'] == meta[stage]['assetId']
            entry = cap['modes']['lit']['iso']
            assert np.allclose(entry['direction'], direction, rtol=0, atol=1e-12)
            assert np.allclose(entry['expectedPosition'], np.asarray(camera['target']) + direction * camera['distance'],
                               rtol=0, atol=1e-12)
            assert sha(entry['path']) == entry['sha256']
            with Image.open(entry['path']) as image:
                assert image.size == (1024, 1024)
                frames[stage][kind] = image.convert('RGBA')
            records[stage][kind] = {'capturePath': str(path), 'captureSHA256': sha(path),
                                    'metadataStage': source_stage, 'capture': cap}
    for kind in ('clay', 'box'):
        assert records['F'][kind]['capture']['lighting'] == records['H'][kind]['capture']['lighting']
    scale, pad = 3, 24
    width, height = (roi[2] - roi[0]) * scale, (roi[3] - roi[1]) * scale
    cell_width, row_height = max(width, 420), height + 62
    board = Image.new('RGB', (2 * cell_width + 3 * pad, 2 * row_height + 180), BG)
    draw = ImageDraw.Draw(board)
    font = lambda size: ImageFont.truetype(FONT, size)
    draw.text((pad, 16), 'F / H oblique head comparison', font=font(26), fill=TEXT)
    draw.text((pad, 54), 'Common orthographic camera and head ROI. 3x nearest-neighbor enlargement.', font=font(16), fill=MUTED)
    for col, title in enumerate(('Full mesh / gray material', 'Native Box / generated color')):
        draw.text((pad + col * (cell_width + pad), 88), title, font=font(20), fill=TEXT)
    for row, stage in enumerate(('F', 'H')):
        y = 126 + row * row_height
        for col, kind in enumerate(('clay', 'box')):
            x = pad + col * (cell_width + pad)
            title = ('F / previous Hi3D reference' if stage == 'F' else 'H / generated front reference')
            draw.text((x, y), title, font=font(17), fill=TEXT)
            if kind == 'clay':
                count = records[stage][kind]['capture']['triangleCounts']['meshPayload']
                subtitle = f'{count:,} triangles' + (' / B geometry reused' if stage == 'F' else '')
            else:
                subtitle = 'Same six source RGB files; native geometry differs'
            draw.text((x, y + 24), subtitle, font=font(14), fill=MUTED)
            crop = backdrop(frames[stage][kind].crop(roi)).resize((width, height), Image.Resampling.NEAREST)
            board.paste(crop, (x + (cell_width - width) // 2, y + 50))
    draw.text((pad, board.height - 25), 'Compare F/H within each column; the mesh and Box use different lighting paths.',
              font=font(15), fill=MUTED)
    board_path = output / 'F-H-iso-faces.webp'
    board.save(board_path, 'WEBP', lossless=True)
    assert np.array_equal(np.asarray(Image.open(board_path)), np.asarray(board))
    for stage in records.values():
        for row in stage.values():
            entry = row['capture']['modes']['lit']['iso']
            assert sha(entry['path']) == entry['sha256']
    proof = {'kind': __doc__, 'status': 'verified existing captures', 'view': 'iso',
             'direction': direction.tolist(), 'camera': {key: camera[key] for key in CAMERA_KEYS},
             'referenceManifest': str(reference), 'referenceManifestSHA256': sha(reference),
             'headWorldBounds': head.tolist(), 'headCrop': roi,
             'headCropSource': {'path': str(crop_source), 'sha256': sha(crop_source)},
             'earlierFourViewProof': {'path': str(earlier), 'sha256': sha(earlier)},
             'FClayReuse': {'sameProxySHA256AsB': True, 'sameClayGLBSHA256': meta['B']['clay']['sha256']},
             'captures': records, 'processing': 'One common integer ROI projected from the recorded world-space head bounds; 3x nearest; neutral backdrop; lossless WebP.',
             'limitation': 'Camera and lighting comparisons apply within each rendering path. Mesh and native Box use different official renderers. No enhancement, inference, rebake, or viewer modification.',
             'panel': {'path': str(board_path), 'sha256': sha(board_path), 'bytes': board_path.stat().st_size,
                       'size': list(board.size)}, 'scriptSHA256': sha(__file__)}
    proof_path = output / 'F-H-iso-provenance.json'
    proof_path.write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'panel': proof['panel'], 'proof': str(proof_path), 'proofSHA256': sha(proof_path), 'headCrop': roi}))


if __name__ == '__main__':
    main()

"""Prepare branch B with a newly inferred Hi3D mesh and a fresh 1024px Box bake.

Run only after branch A is complete and the H3 server has released the GPU.
"""
import argparse
import json
from pathlib import Path
import time

from run_geometry_ab import digest, prepare, read_json, use_cuda_runtime_on_wsl, write_json
import folder_paths
from box_nodes import store
from box_nodes.hi3d_refine import RefineProxyHi3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--coarse-baseline', required=True, type=Path)
    parser.add_argument('--backend', default='webgl2', choices=['auto', 'webgl2', 'webgpu'])
    args = parser.parse_args()
    use_cuda_runtime_on_wsl()
    run = args.run.resolve()
    case = run / 'B'
    case.mkdir(exist_ok=True)
    source = args.coarse_baseline.resolve()
    manifest = read_json(source / 'manifest.json')
    folder_paths.set_output_directory(str(run / 'output'))
    mesh = source / manifest['proxy']
    ref = manifest['reference']
    reference = source / ref['image']
    photo_sha = digest(reference)
    coarse_sha = digest(mesh)
    if coarse_sha != manifest['proxySha256']:
        raise ValueError('Coarse geometry differs from the A manifest.')
    meta = {
        'glbSha256': coarse_sha,
        'sourceSha256': manifest['provenance']['originalGlbSha256'],
        'transform': manifest['transform'], 'inverseTransform': manifest['inverseTransform'],
        'tightBounds': manifest['tightBounds'],
    }
    write_json(case / 'coarse-canonical.json', meta)
    proof = {'model': 'hi3d-official/Hi3Dv1.0', 'resolution': 1024, 'steps': 25,
             'smoothing': 'none', 'cpuOffload': True,
             'sourceBaseline': manifest['asset_id'], 'coarseMesh': str(mesh),
             'coarseSha256': coarse_sha, 'referenceImage': str(reference),
             'referencePngSha256': photo_sha, 'referenceSampleSha256': ref['sha256'],
             'startedEpoch': time.time()}
    write_json(case / 'hi3d-inputs.json', proof)
    proxy = {
        'path': str(mesh), 'sha256': coarse_sha, 'metadata': str(case / 'coarse-canonical.json'),
        'stageKey': store.key('geometry-ab-coarse-v1', coarse_sha, photo_sha),
        'reference': {**ref, 'path': str(reference.parent)},
        'provenance': manifest['provenance']['meshGeneration'],
        'rotationDegreesXYZ': manifest['provenance'].get('rotationDegreesXYZ', [0, 0, 0]),
    }
    if ref.get('camera'):
        proxy.update(camera=ref['camera'], croppedImage=str(source / ref['camera']['image']))
    refined, = RefineProxyHi3D().refine(proxy, mode='hi3d_1024', steps=25,
                                      smoothing='none', cpu_offload=True)
    report = refined['refinement']
    if report['inference']['resolution'] != 1024 or report['inference']['steps'] != 25:
        raise ValueError('Hi3D inference settings do not match the experiment.')
    if digest(mesh) != coarse_sha or digest(reference) != photo_sha:
        raise ValueError('Hi3D source inputs changed during inference.')
    proof.update({'elapsedSeconds': round(time.time() - proof['startedEpoch'], 2),
                  'outputProxy': refined['path'], 'outputProxySha256': digest(refined['path']),
                  'canonicalMetadata': read_json(refined['metadata']), 'report': report})
    write_json(case / 'hi3d-completed.json', proof)
    baseline = store.baseline(refined, 1024, args.backend, 'none')
    recipe = prepare(run, 'B', store.asset_dir(baseline))
    recipe['geometryRefinement'] = proof
    write_json(case / 'prepared.json', recipe)
    print(json.dumps({'B_baseline': baseline, 'prepared': str(case / 'prepared.json'),
                      'hi3dProof': str(case / 'hi3d-completed.json')}), flush=True)


if __name__ == '__main__':
    main()

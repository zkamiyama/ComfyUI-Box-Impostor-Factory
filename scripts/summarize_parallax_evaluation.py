#!/usr/bin/env python3
"""Create a compact evidence summary without substituting a visual quality score."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted(rows, value, count):
    total = sum(count(row) for row in rows)
    return sum(value(row) * count(row) for row in rows) / total if total else None


def summarize(report_path, output):
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text())
    motions = report['depthReprojectedMotion']
    summaries = []
    for variant in dict.fromkeys(row['variant'] for row in motions):
        for band in ('within +/-30 degrees', 'outside +/-30 through +/-60 degrees'):
            rows = [r for r in motions if r['variant'] == variant
                    and (max(abs(r['fromAngle']), abs(r['toAngle'])) <= 30) == band.startswith('within')]
            if not rows:
                continue
            visible = [r for r in rows if r['colorAfterDepthReprojection']['pixels']]
            common = [r for r in rows if r['frozenFrontPixelControl']['commonOpaquePixels']]
            count = lambda r: r['colorAfterDepthReprojection']['pixels']
            error = lambda r: r['colorAfterDepthReprojection']['linearRGBAbsoluteError']['mean']
            worst = max(visible, key=error)
            summaries.append({'variant': variant, 'band': band, 'adjacentAnglePairs': len(rows),
                              'angleStepDegrees': sorted(set(abs(r['toAngle'] - r['fromAngle']) for r in rows)),
                              'acceptedCorrespondenceSamples': sum(count(r) for r in rows),
                              'acceptanceFraction': sum(count(r) for r in rows) / sum(r['sourceInteriorPixels'] for r in rows),
                              'sampleWeightedMeanLinearRGBResidual': weighted(visible, error, count),
                              'worstPairMeanLinearRGBResidual': {'angles': [worst['fromAngle'], worst['toAngle']], 'mean': error(worst)},
                              'frozenScreenImageControl': {
                                  'commonOpaqueSamples': sum(r['frozenFrontPixelControl']['commonOpaquePixels'] for r in common),
                                  'actualMeanLinearRGBResidual': weighted(common, lambda r: r['frozenFrontPixelControl']['actualVariant']['linearRGBAbsoluteError']['mean'], lambda r: r['frozenFrontPixelControl']['commonOpaquePixels']),
                                  'frozenMeanLinearRGBResidual': weighted(common, lambda r: r['frozenFrontPixelControl']['frozenFrontPixels']['linearRGBAbsoluteError']['mean'], lambda r: r['frozenFrontPixelControl']['commonOpaquePixels'])}})
    repeated = report['repeatedCameraStability']
    result = {'kind': 'compact-parallax-evidence-summary-v1', 'fullReport': str(report_path),
              'fullReportSHA256': sha(report_path), 'sweep': report['sweep'], 'sweepSHA256': report['sweepSHA256'],
              'sourceAssetId': report['source'].get('assetId'),
              'sourceManifestSHA256': report['source']['manifestSHA256'], 'evaluatorSHA256': report['evaluatorSHA256'],
              'frontHeadFidelity': [x for x in report['frontFidelity'] if x['region'] == 'head'],
              'independentReferences': report['references'],
              'geometryComparisons': len(report['geometryAcrossVariants']),
              'allVariantAlphaAndDepthBytesEqual': all(r['alphaBytesEqual'] and r['depthFloat32BytesEqual'] for r in report['geometryAcrossVariants']),
              'originalNativeGeometryControl': report['originalNativeGeometryControl'],
              'repeatedCamera': {'comparisons': len(repeated),
                                 'allRGBBytesEqual': all(r['RGBABytesEqual'] for r in repeated) if repeated else None,
                                 'allDepthBytesEqual': all(r['depthBytesEqual'] for r in repeated) if repeated else None,
                                 'interpretation': 'Independently captured repeated-camera results when comparisons exist; no repeat test is claimed for an empty set.'},
              'motionBands': summaries,
              'motionExampleNearFront': next((r for r in motions if r['variant'] == 'camera-hybrid' and r['fromAngle'] == 0), None),
              'visibilityPolicy': report['visibilityPolicy'], 'cameraValidation': report.get('cameraValidation'),
              'limitations': report['limitations'] + ['Band aggregation weights each accepted correspondence sample; the same world surface can appear in many adjacent pairs.', 'These color residuals measure consistency after approximate depth reprojection, not ghosting, naturalness, or ground-truth face accuracy.'],
              'sourceAssets': report['assets'], 'summaryHelperSHA256': sha(__file__)}
    payload = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    if output.exists() and output.read_bytes() != payload:
        raise FileExistsError('Preserve prior summary: ' + str(output))
    output.write_bytes(payload)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.report, args.output)
    print(json.dumps({'summary': str(args.output), 'sha256': sha(args.output), 'motionBands': result['motionBands']}, indent=2))

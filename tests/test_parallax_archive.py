"""Byte recovery and corruption checks for the bounded evidence archive."""
import base64
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib

PATH = Path(__file__).resolve().parents[1] / 'scripts/archive_parallax_evidence.py'
spec = importlib.util.spec_from_file_location('parallax_archive', PATH)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ParallaxArchiveTests(unittest.TestCase):
    def fixture(self, root):
        base = root / 'experiment'
        run = base / 'trial'
        run.mkdir(parents=True)
        texture = root / 'maps' / 'color.bin'
        texture.parent.mkdir()
        texture.write_bytes(bytes(range(256)))
        manifest = {'asset_id': 'tiny', 'resolution': 1, 'layout': 'box',
                    'bakeBox': {'min': [-1, -1, -1], 'max': [1, 1, 1]},
                    'textures': {'colorMap': {'path': 'maps/color.bin', 'byteLength': 256, 'sha256': archive.sha(texture)}}}
        manifest_path = root / 'manifest.json'
        manifest_path.write_bytes(archive.encoded(manifest))
        camera = root / 'camera.json'
        camera.write_bytes(b'{"camera":"fixture"}\n')
        source = base / 'harness/scripts/capture.mjs'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'// fixture source bytes\n')
        sweep = {'status': 'failed', 'width': 1, 'height': 1, 'frames': [],
                 'sourceFilesSHA256': {'scripts/capture.mjs': archive.sha(source)},
                 'source': {'manifestPath': str(manifest_path), 'manifestSHA256': archive.sha(manifest_path)},
                 'cameraReference': str(camera), 'cameraReferenceSHA256': archive.sha(camera)}
        (run / 'sweep.json').write_bytes(archive.encoded(sweep))
        (run / 'diagnosis.log').write_bytes(b'failed fixture retained\n')
        scene = {'previewType': 'native-box', 'assetId': 'tiny',
                 'nativeBox': {'manifest': manifest, 'files': {'maps/color.bin': base64.b64encode(texture.read_bytes()).decode()}}}
        html = b'<script id="scene-payload" type="text/plain">' + base64.b64encode(zlib.compress(archive.encoded(scene))) + b'</script>'
        gz = gzip.compress(html, mtime=0)
        html_path = root / 'tiny.html.gz'
        html_path.write_bytes(gz)
        plan = {'schemaVersion': 1, 'selectionStatus': 'frozen', 'experimentRoot': str(base),
                'runs': [{'name': 'trial', 'purpose': 'synthetic failed-capture preservation'}],
                'assets': [{'name': 'H', 'manifestPath': str(manifest_path),
                            'officialInspectorRecovery': {'gitCommit': 'fixture', 'gitPath': 'tiny.html.gz',
                                                          'gzipSHA256': archive.digest(gz), 'htmlSHA256': archive.digest(html)}}]}
        plan_path = root / 'plan.json'
        plan_path.write_bytes(archive.encoded(plan))
        return plan_path, html_path, manifest_path

    def test_deterministic_archive_reopens_all_bytes_and_detects_corrupt_part(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _, _ = self.fixture(root)
            frozen = root / 'inventory.json'
            frozen.write_bytes(archive.encoded(archive.collect(plan)))
            a = archive.pack(plan, frozen, root / 'out-a', 'evidence')
            b = archive.pack(plan, frozen, root / 'out-b', 'evidence')
            self.assertEqual(a['parts'], b['parts'])
            self.assertTrue(a['verification']['allMembersReopenedAndHashed'])
            path = root / 'out-a' / a['parts'][0]['file']
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'part differs'):
                archive.verify(a['index'])

    def test_draft_and_changed_sources_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, _, _ = self.fixture(root)
            plan = archive.read_json(plan_path)
            plan['selectionStatus'] = 'draft'
            plan_path.write_bytes(archive.encoded(plan))
            frozen = root / 'inventory.json'
            frozen.write_bytes(archive.encoded(archive.collect(plan_path)))
            with self.assertRaisesRegex(ValueError, 'Inventory changed'):
                archive.pack(plan_path, frozen, root / 'output', 'evidence')
            plan['selectionStatus'] = 'frozen'
            plan_path.write_bytes(archive.encoded(plan))
            frozen.write_bytes(archive.encoded(archive.collect(plan_path)))
            (root / 'experiment/trial/diagnosis.log').write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'Inventory changed'):
                archive.pack(plan_path, frozen, root / 'output', 'evidence')

    def test_native_recovery_preserves_full_manifest_and_map_bytes_without_html_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, html, manifest = self.fixture(root)
            inventory = root / 'inventory.json'
            inventory.write_bytes(archive.encoded(archive.collect(plan)))
            before = html.read_bytes()
            result = archive.extract_native(inventory, 'H', html, manifest, root / 'recovered')
            self.assertEqual((root / 'recovered/maps/color.bin').read_bytes(), bytes(range(256)))
            self.assertEqual((root / 'recovered/manifest.json').read_bytes(), manifest.read_bytes())
            self.assertEqual(html.read_bytes(), before)
            self.assertFalse(result['officialHTMLModified'])
            (root / 'recovered/maps/color.bin').write_bytes(b'corrupt')
            with self.assertRaises(FileExistsError):
                archive.extract_native(inventory, 'H', html, manifest, root / 'recovered')

    def test_bounded_parts_and_unsafe_names(self):
        rows = [{'archivePath': str(i), 'bytes': 4000} for i in range(5)]
        parts = archive.partition(rows, budget=21000)
        self.assertEqual([len(p) for p in parts], [2, 2, 1])
        self.assertEqual([row for part in parts for row in part], rows)
        for name in ('../secret', '/absolute', 'safe/../../bad', '.env', 'a\\b'):
            with self.assertRaises(ValueError):
                archive.relative_name(name)

    def test_explicit_qa_directory_keeps_original_webm_recording_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path, _, _ = self.fixture(root)
            qa = root / 'factory-qa'
            qa.mkdir()
            video = qa / 'actual-canvas.webm'
            video.write_bytes(b'\x1a\x45\xdf\xa3original recording byte fixture\x00\xff')
            plan = archive.read_json(plan_path)
            plan['evidenceDirectories'] = [{'path': str(qa), 'archivePath': 'factory-qa', 'role': 'actual-canvas-recording'}]
            plan_path.write_bytes(archive.encoded(plan))
            inventory = archive.collect(plan_path)
            row = next(r for r in inventory['members'] if r['archivePath'] == 'factory-qa/actual-canvas.webm')
            self.assertEqual(row['sha256'], archive.sha(video))
            self.assertEqual(row['bytes'], len(video.read_bytes()))
            frozen = root / 'inventory.json'
            frozen.write_bytes(archive.encoded(inventory))
            result = archive.pack(plan_path, frozen, root / 'output', 'evidence')
            self.assertEqual(result['verification']['status'], 'verified')


if __name__ == '__main__':
    unittest.main()

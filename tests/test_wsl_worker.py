import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'wsl-windows-node.sh'


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'requires Bash')
class WslWorkerTests(unittest.TestCase):
    def test_paths_environment_and_argument_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            # The WSL and Windows binaries are replaced by argument recorders.
            (folder/'wslpath').write_text('#!/bin/sh\n[ "$1" = -w ] || exit 1\nprintf "win:%s" "$2"\n')
            (folder/'node.exe').write_text(
                '#!/usr/bin/env python3\nimport json, os, sys\n'
                'print(json.dumps([sys.argv[1:], os.environ["BI_CHROMIUM"], os.environ["WSLENV"], '
                'os.environ["BI_SOFTWARE_GPU"]]))\n')
            for program in folder.iterdir():
                program.chmod(0o755)
            env = dict(os.environ, PATH=directory + ':' + os.environ['PATH'], BI_SOFTWARE_GPU='0',
                       WSLENV='USER_CONFIG/pl:BI_SOFTWARE_GPU/u:BI_CHROMIUM/p')
            args = ['/project/worker/cli.mjs', 'bake', '--input', '/model folder/test $1.glb',
                    '--output', '/new/directory', '--maps', '']
            expected = [('win:' + value if value.startswith('/') else value) for value in args]
            for browser, transformed in [('/windows/Browser X.exe', 'win:/windows/Browser X.exe'),
                                         (r'C:\Browser X.exe', r'C:\Browser X.exe')]:
                with self.subTest(browser=browser):
                    result = subprocess.run([SCRIPT, *args], env={**env, 'BI_CHROMIUM':browser},
                                            check=True, capture_output=True, text=True)
                    self.assertEqual(json.loads(result.stdout),
                                     [expected, transformed, 'BI_CHROMIUM:BI_SOFTWARE_GPU:USER_CONFIG/pl', '0'])

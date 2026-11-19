"""Replay the rejected-step diagnostic with its immutable pre-fix solver.

Uses the existing F:/C/actual/predictions; no GPU or bake. The diagnostic writes
its original JSON/log destination on F:. The source snapshot is from 527f2d9.
"""
import importlib.util
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path[:0]=[str(REPO),str(REPO.parents[1])]
from comfy.cli_args import args
args.cpu=True
import box_nodes
spec=importlib.util.spec_from_file_location('box_nodes.full_depth',HERE/'initial/full_depth-v1.py')
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
box_nodes.full_depth=module
spec.loader.exec_module(module)
runpy.run_path(str(HERE/'diagnose_first_step.py'),run_name='__main__')

"""Subprocess-only Hi3D v1.0 inference (requires a separately provisioned GPU venv)."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=25)
    parser.add_argument('--resolution',type=int,choices=[512,1024,1536],default=512)
    parser.add_argument('--offload',action='store_true')
    args=parser.parse_args()
    if not 1 <= args.steps <= 50:
        parser.error('steps must be 1..50')
    start=time.monotonic()
    import torch
    # WSL2 driver/NVML can segfault although the CUDA runtime works.
    import platform
    if 'microsoft' in platform.release().lower():
        torch.cuda._device_count_nvml=lambda:-1
    from PIL import Image
    import trimesh
    from sparc3d import Sparc3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError('Hi3D requires an available CUDA device')
    source=trimesh.load(args.source,force='mesh',process=False)
    if not isinstance(source,trimesh.Trimesh) or len(source.faces)==0:
        raise ValueError('Expected a coarse GLB triangle mesh')
    photo=Image.open(args.reference).convert('RGBA')
    model=Sparc3DPipeline.from_pretrained('hi3d-official/Hi3Dv1.0')
    if args.offload:
        model.enable_sequential_cpu_offload('cuda')
    else:
        model.to('cuda')
    torch.cuda.reset_peak_memory_stats()
    generated=model(photo,source,resolution=args.resolution,num_inference_steps=args.steps,seed=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    generated.mesh.export(args.output)
    report={
      'status':'success','model':'hi3d-official/Hi3Dv1.0',
      'resolution':args.resolution,'steps':args.steps,
      'source_faces':len(source.faces),'output_faces':len(generated.mesh.faces),
      'output_vertices':len(generated.mesh.vertices),
      'elapsed_s':round(time.monotonic()-start,2),
      'torch_cuda':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0),
      'gpu_max_allocated_mib':round(torch.cuda.max_memory_allocated()/(2**20),2),
    }
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    main()

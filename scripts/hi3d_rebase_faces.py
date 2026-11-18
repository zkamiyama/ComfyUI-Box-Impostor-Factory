"""Reapply previously generated H3 face images to a freshly baked Hi3D Box.

Experimental diagnostic; does not regenerate the H3 images or PBR materials.
Original rendered face images and their generation canvas are preserved.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from box_nodes import store,faces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path,help='old H3 Box directory')
    p.add_argument('--baseline',required=True,type=Path,help='new Hi3D 512px Box directory')
    p.add_argument('--work-root',required=True,type=Path,help='isolated temporary asset root')
    p.add_argument('--min-iou',type=float,default=.70)
    p.add_argument('--max-shift',type=float,default=.10)
    p.add_argument('--max-scale-change',type=float,default=.15)
    args=p.parse_args()
    original=json.loads((args.source/'manifest.json').read_text())
    baseline=json.loads((args.baseline/'manifest.json').read_text())
    args.work_root.mkdir(parents=True,exist_ok=True)
    dest=args.work_root/baseline['asset_id']
    if not dest.exists():
        shutil.copytree(args.baseline,dest)
    with patch.object(store,'root',return_value=args.work_root):
        store.validate(dest,baseline['asset_id'],check_bytes=True)
        candidates={}
        for face_id,data in original['faces'].items():
            edit=data.get('edit') or {}
            source_rel=edit.get('view')
            if not source_rel:
                continue
            filepath=store.contained(args.source,source_rel).with_suffix('.npy')
            if not filepath.is_file():
                raise FileNotFoundError(f'Generated face not found: {filepath}')
            img=np.load(filepath,allow_pickle=False)
            item={**faces.save_view(img),'provenance':edit.get('provenance',{})}
            candidates[face_id]=item
        aligned=faces.align(candidates,baseline['asset_id'],args.min_iou,args.max_shift,args.max_scale_change)
        changed=faces.compose(baseline['asset_id'],aligned)
        new_manifest=store.validate(store.asset_dir(changed),changed,check_bytes=True)
        reports={}
        for fid,info in new_manifest['faces'].items():
            edit=info.get('edit') or {}
            report_file=aligned['faces'].get(fid)
            if report_file:
                report=json.loads((Path(report_file['directory'])/'report.json').read_text())
                reports[fid]={'status':report['status'],'reason':report['reason'],
                              'iou':report.get('after',{}).get('iou'),
                              'boundary_p95':report.get('after',{}).get('boundaryP95Px')}
        summary={'source':original['asset_id'],'baseline':baseline['asset_id'],
                 'rebased':changed, 'min_iou':args.min_iou,
                 'generated_count':len(candidates),
                 'accepted_count':sum(v['status']=='accepted' for v in reports.values()),
                 'faces':reports,'path':str(store.asset_dir(changed))}
        (args.work_root/'rebase-report.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    main()

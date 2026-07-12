from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

def run(out_dir='artifacts/packcell_final_attribution'):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path('artifacts/packcell_layout_results.jsonl').read_text().splitlines()]
    layouts=rows[:5]; variants=('free_space_arm','pick_only_workcell','placement_only_workcell','full_workcell'); results=[]
    for row in layouts:
        for variant in variants:
            job={'job_id':f'{variant}_{row["layout_id"]}','layout_id':row['layout_id'],'variant':variant,'parameters':row['parameters']}; path=out/(job['job_id']+'.json')
            if not path.exists(): subprocess.run([sys.executable,'-m','osc.packcell.worker','--job',json.dumps(job),'--out',str(path)],check=False)
            if path.exists(): results.append(json.loads(path.read_text()))
    report={'jobs_expected':20,'jobs_completed':len(results),'results':results,'decision':'pending' if len(results)<20 else 'attribution_complete'}
    Path('artifacts/packcell_final_attribution.json').write_text(json.dumps(report,indent=2)); return report

from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

def run(out_dir='artifacts/packcell_final_attribution_v2'):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in Path('artifacts/packcell_layout_results.jsonl').read_text().splitlines()]
    layouts=rows[:5]; variants=('free_space_arm','pick_only_workcell','placement_only_workcell','full_workcell'); results=[]
    for row in layouts:
        for variant in variants:
            job={'job_id':f'{variant}_{row["layout_id"]}','layout_id':row['layout_id'],'variant':variant,'parameters':row['parameters']}; path=out/(job['job_id']+'.json')
            retry=True
            if path.exists():
                try: retry=json.loads(path.read_text()).get('status')!='complete'
                except Exception: retry=True
            if retry: subprocess.run([sys.executable,'-m','osc.packcell.worker','--job',json.dumps(job),'--out',str(path)],check=False)
            if path.exists(): results.append(json.loads(path.read_text()))
    report={'schema':'packcell.attribution.v2','jobs_expected':20,'jobs_completed':len(results),'clean_jobs':sum(r.get('status')=='complete' for r in results),'results':results,'decision':'pending' if len(results)<20 or any(r.get('status')!='complete' for r in results) else 'attribution_complete'}
    Path('artifacts/packcell_final_attribution_v2.json').write_text(json.dumps(report,indent=2)); return report

from __future__ import annotations
import hashlib,json,itertools
from pathlib import Path
from .candidate_search import search

def manifest():
    rows=[]; i=0
    for h,yaw,x,y,sector in itertools.product((0,.05,.10,.15),(-30,0,30),(.18,.22,.26),(-.12,-.06,0,.06,.12),('left','right')):
        rows.append({'layout_id':f'L{i:03d}','parameters':{'base_height_m':h,'base_yaw_deg':yaw,'pick_x_m':x,'pick_y_m':y,'box_sector':sector},'status':'pending'}); i+=1
    config={'candidate_fast':True,'position_error_threshold_m':.002,'clearance_threshold_m':.003,'deterministic_seed':0,'evaluation_order':'layout_id'}
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {'schema':'packcell.layout_sweep.v1','layouts':rows,'configuration':config,'code_hash':code}

def run(path='artifacts/packcell_layout_manifest.json',results='artifacts/packcell_layout_results.jsonl'):
    m=manifest(); Path(path).write_text(json.dumps(m,indent=2)); done={}
    rp=Path(results)
    if rp.exists():
        for line in rp.read_text().splitlines():
            try: done[json.loads(line)['layout_id']]=json.loads(line)
            except Exception: pass
    with rp.open('a') as f:
        for row in m['layouts']:
            if row['layout_id'] in done: continue
            p=row['parameters']; layout={'base_height':p['base_height_m'],'base_yaw':p['base_yaw_deg']*3.1415926535/180,'pick_zone':[p['pick_x_m'],p['pick_y_m']],'box_sector':p['box_sector']}
            try:
                r=search(0,layout,fast=True); out={'layout_id':row['layout_id'],'parameters':p,'valid_candidates':r['valid_count'],'workspace_design_intervention_required':r['valid_count']==0,'prohibited_link_frequency':r['prohibited_link_frequency']}
            except Exception as exc: out={'layout_id':row['layout_id'],'parameters':p,'status':'error','error':str(exc)}
            f.write(json.dumps(out)+'\n'); f.flush()
    return m

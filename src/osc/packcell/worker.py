from __future__ import annotations
import argparse,json,os,time
from .candidate_search import search

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--job',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); job=json.loads(a.job); t=time.time()
    p=job['parameters']; layout={'base_height':p['base_height_m'],'base_yaw':p['base_yaw_deg']*3.1415926535/180,'pick_zone':[p['pick_x_m'],p['pick_y_m']],'box_sector':p['box_sector']}; targets={'free_space_arm':([.20,0,.18],False),'pick_only_workcell':([p['pick_x_m'],p['pick_y_m'],.087],True),'placement_only_workcell':([.13,-.075,.12],False),'full_workcell':(None,True)}; target,object_checks=targets[job['variant']]
    if target is None and job['variant'] != 'full_workcell': raise ValueError('missing variant target')
    try:
            r=search(0,layout,fast=False,variant=job['variant'],target_pose=target); result={'job_id':job['job_id'],'layout_id':job['layout_id'],'variant':job['variant'],'target':target,'object_checks':object_checks,'valid_candidates':r['valid_count'],'candidate_count':r['candidate_count'],'best':r['best'],'prohibited_link_frequency':r['prohibited_link_frequency'],'runtime_s':time.time()-t,'status':'complete'}
    except Exception as exc: result={'job_id':job['job_id'],'layout_id':job['layout_id'],'variant':job['variant'],'status':'error','error':str(exc),'runtime_s':time.time()-t}
    tmp=a.out+'.tmp'; open(tmp,'w').write(json.dumps(result,indent=2)); os.replace(tmp,a.out)
if __name__=='__main__': main()

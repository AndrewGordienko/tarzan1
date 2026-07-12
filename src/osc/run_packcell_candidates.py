from __future__ import annotations
import argparse,json
from .packcell.candidate_search import search
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/packcell_candidate_search.json'); a=ap.parse_args(); r=search(); open(a.out,'w').write(json.dumps(r,indent=2)); print(json.dumps({k:r[k] for k in ('candidate_count','valid_count','best','prohibited_link_frequency','workspace_design_intervention_required')},indent=2))
if __name__=='__main__': main()

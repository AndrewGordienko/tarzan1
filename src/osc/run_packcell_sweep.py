from __future__ import annotations
import argparse,json
from .packcell.sweep_manifest import run
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='artifacts/packcell_layout_manifest.json'); ap.add_argument('--results',default='artifacts/packcell_layout_results.jsonl'); a=ap.parse_args(); m=run(a.manifest,a.results); print(json.dumps({'layouts':len(m['layouts']),'manifest':a.manifest,'results':a.results},indent=2))
if __name__=='__main__': main()

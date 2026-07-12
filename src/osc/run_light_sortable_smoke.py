from pathlib import Path
import gzip, json
from osc.packcell.ur10e_adapter import UR10eAdapter
from osc.packcell.ur10e_scripted import UR10eScriptedController

def main():
    result=UR10eScriptedController(UR10eAdapter()).run();trace=result.pop('trace',[]);trace_path=Path('artifacts/light_sortable_physical_smoke_trace.jsonl.gz')
    with trace_path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            for row in trace:compressed.write((json.dumps(row,separators=(',',':'))+'\n').encode())
    result['trace_rows']=len(trace);result['trace_artifact']=str(trace_path)
    out=Path('artifacts/light_sortable_physical_smoke.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result.get(k) for k in ('success','failure_phase','camera_verification','scorer_verification')},indent=2))
if __name__=='__main__':main()

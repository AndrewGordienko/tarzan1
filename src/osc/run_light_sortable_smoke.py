from pathlib import Path
import json
from osc.packcell.ur10e_adapter import UR10eAdapter
from osc.packcell.ur10e_scripted import UR10eScriptedController

def main():
    result=UR10eScriptedController(UR10eAdapter()).run();out=Path('artifacts/light_sortable_physical_smoke.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result.get(k) for k in ('success','failure_phase','camera_verification','scorer_verification')},indent=2))
if __name__=='__main__':main()

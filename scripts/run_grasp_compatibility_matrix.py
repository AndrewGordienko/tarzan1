from pathlib import Path
import json
from osc.packcell.grasp_feasibility import compatibility_matrix, load_end_effector_contract

ROOT = Path(__file__).resolve().parents[1]
result = compatibility_matrix(ROOT / "configs/amazon_small_sortable_v1.json", load_end_effector_contract())
out = ROOT / "artifacts/panda_grasp_compatibility_matrix.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"artifact": str(out), "summary": result["summary"], "objects": result["objects"]}, indent=2))

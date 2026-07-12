from pathlib import Path
import json
from osc.packcell.grasp_feasibility import compatibility_matrix, load_end_effector_contract


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = compatibility_matrix(root / "configs/amazon_small_sortable_v1.json", load_end_effector_contract())
    out = root / "artifacts/panda_grasp_compatibility_matrix.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "summary": result["summary"]}, indent=2))

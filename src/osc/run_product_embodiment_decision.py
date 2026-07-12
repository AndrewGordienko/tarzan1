from pathlib import Path
import json
from osc.packcell.product_embodiment_decision import build_decision

def main() -> None:
    result = build_decision(); out = Path(__file__).resolve().parents[2] / "artifacts/product_embodiment_decision_v1.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "arm_preference_if_tool_verified": result["arm_preference_if_tool_verified"], "decision": result["decision"], "reason": result["decision_reason"]}, indent=2))

if __name__ == "__main__": main()

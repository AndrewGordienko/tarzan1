from pathlib import Path
import json
from osc.packcell.embodiment_selection import build_matrix

def main() -> None:
    result = build_matrix(); out = Path(__file__).resolve().parents[2] / "artifacts/coupled_embodiment_selection_v1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "product_shortlist": result["product_shortlist"], "decision": result["decision"]}, indent=2))

if __name__ == "__main__":
    main()

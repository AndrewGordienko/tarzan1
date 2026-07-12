"""v0.8 industrial embodiment interface placeholder.

The PackCell observation/controller/scorer contracts are portable; a concrete
7-DoF Menagerie asset is intentionally selected before controller work begins.
"""
from pathlib import Path
import json

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "amazon_small_sortable_v1.json"

def frozen_scope():
    return json.loads(CONFIG.read_text())

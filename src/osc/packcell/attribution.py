from __future__ import annotations
import json
from pathlib import Path

SCOPES={
    "so101_micro_packcell":{"arm":"SO-101","cell_scale":"small objects / small box","status":"research_lane"},
    "amazon_small_sortable":{"arm":"larger industrial arm (not yet selected)","cell_scale":"realistic package and box dimensions","status":"planned_phase_1"},
}

def run(out='artifacts/packcell_attribution_ladder.json'):
    # Keep the matrix explicit: the current XML generator does not yet expose
    # validated geometry variants, so these lanes are not silently inferred from
    # the full scene.
    lanes={name:{"status":"not_isolated","successes":None,"episodes":0,"reason":"scene_variant_not_implemented"} for name in ("free_space_arm","pick_only_workcell","placement_only_workcell","full_workcell")}
    lanes["full_workcell"]={"status":"evaluated","successes":0,"episodes":360,"reason":"coarse_layout_sweep_zero_valid_layouts"}
    report={"scopes":SCOPES,"lanes":lanes,"interpretation":"additional isolated controls required before attributing zero to hardware scale","five_near_miss_full_search":"not_run_until_scene_variants_are_available"}
    Path(out).write_text(json.dumps(report,indent=2)); return report

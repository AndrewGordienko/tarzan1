"""Small top-down GIF renderer for the PoC."""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw


def render_states(states, path, title="Tarzan packing"):
    frames = []
    W, H = 800, 600
    for state in states:
        im = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(im)
        cx, cy, _ = state.container.dimensions
        scale = min(650 / cx, 430 / cy)
        ox, oy = 70, 100
        def box(p, fill, outline="black"):
            x, y = p.position[0], p.position[1]
            w, h = p.size[0], p.size[1]
            draw.rectangle((ox + x * scale, oy + y * scale,
                            ox + (x + w) * scale, oy + (y + h) * scale),
                           fill=fill, outline=outline, width=2)
            draw.text((ox + x * scale + 3, oy + y * scale + 3), p.item_id, fill="black")
        draw.text((70, 35), title, fill="black")
        draw.rectangle((ox, oy, ox + cx * scale, oy + cy * scale), outline="navy", width=4)
        colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#fdd0a2", "#dadaeb", "#f7b6d2"]
        for idx, p in enumerate(state.placements.values()):
            box(p, colors[idx % len(colors)])
        frames.append(im)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=500, loop=0)
    return path


def render_policy_comparison(rows, late_row, path):
    """Render a single side-by-side GIF: three demo-conditioned policies plus
    the late-item intervention panel."""
    from PIL import ImageFont
    panels = [("Heavy-bottom demo", rows["correct_heavy_bottom"]),
              ("Max-volume demo", rows["different_max_volume"]),
              ("Min-rehandling demo", rows["minimize_rehandling"]),
              ("Late item / repack", late_row)]
    width, height = 420, 360
    frames = []
    max_frames = max(len(v.get("_states", [])) for _, v in panels)
    for step in range(max_frames):
        canvas = Image.new("RGB", (width * 2, height * 2), "white")
        for idx, (label, row) in enumerate(panels):
            states = row.get("_states", [])
            if not states:
                continue
            state = states[min(step, len(states) - 1)]
            panel = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(panel)
            cx, cy, _ = state.container.dimensions
            scale = min(320 / cx, 230 / cy)
            ox, oy = 35, 65
            draw.text((20, 18), label, fill="black")
            draw.text((20, 38), f"step {min(step, len(states)-1)}  {row.get('program_policy', '')}", fill="black")
            draw.rectangle((ox, oy, ox + cx * scale, oy + cy * scale), outline="navy", width=3)
            colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#fdd0a2", "#dadaeb"]
            for j, p in enumerate(state.placements.values()):
                x, y = p.position[0], p.position[1]
                draw.rectangle((ox + x * scale, oy + y * scale,
                                ox + (x + p.size[0]) * scale,
                                oy + (y + p.size[1]) * scale),
                               fill=colors[j % len(colors)], outline="black", width=2)
                draw.text((ox + x * scale + 2, oy + y * scale + 2), p.item_id, fill="black")
            canvas.paste(panel, ((idx % 2) * width, (idx // 2) * height))
        frames.append(canvas)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=650, loop=0)
    return path

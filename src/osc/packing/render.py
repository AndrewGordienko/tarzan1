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

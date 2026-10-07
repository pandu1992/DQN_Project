#!/usr/bin/env python3
"""
Render BPAN two-vessel GIFs over the real Bintulu chart:
  multi_interaction.gif : two independent vessels avoiding each other (ext. d)
  twoway_headon.gif     : opposing inbound/outbound vessels in one channel (ext. e)
Reads results_bintulu/trajectories/{multi_interaction,twoway_headon}.json and
composites smooth frames with Pillow. Each vessel has its own colour + trail;
the inter-vessel link turns amber on near-miss and red on collision.
"""
import os, json, math
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAJ = os.path.join(ROOT, "results_bintulu", "trajectories")
CHART = os.path.join(ROOT, "assets", "bintulu", "bintulu_chart.png")
OUT = os.path.join(ROOT, "results_bintulu", "gifs")
os.makedirs(OUT, exist_ok=True)

SCALE = 0.40
INTERP = 3
MAX_FRAMES = 100
N_COLORS = 64
GREEN = (30, 203, 107); RED = (255, 77, 77)
VA = (77, 163, 255)      # vessel A (inbound/learned) blue
VB = (255, 159, 67)      # vessel B (outbound/partner) orange
TA = (150, 200, 255); TB = (255, 205, 150)   # trails
PLANA = (90, 150, 230); PLANB = (230, 160, 90)
GOAL = (255, 210, 74); INK = (18, 18, 18)
AMBER = (255, 200, 60)

try:
    FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_S = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
except Exception:
    FONT = ImageFont.load_default(); FONT_S = ImageFont.load_default()

NEARMISS_R = 60; COLLIDE_R = 26


def load_chart(w, h):
    img = Image.open(CHART).convert("RGB").resize((w, h), Image.LANCZOS)
    ov = Image.new("RGB", (w, h), (6, 14, 26))
    return Image.blend(img, ov, 0.10)


def lerp(a, b, t): return a + (b - a) * t


def interp(frames):
    out = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        for k in range(INTERP):
            t = k / INTERP
            out.append({"x": lerp(a["x"], b["x"], t), "y": lerp(a["y"], b["y"], t), "idx": i})
    out.append({"x": frames[-1]["x"], "y": frames[-1]["y"], "idx": len(frames) - 1})
    return out


def tri(d, cx, cy, ang, col):
    pts = [(11, 0), (-7, 7), (-7, -7)]
    rot = [(cx + px * math.cos(ang) - py * math.sin(ang), cy + px * math.sin(ang) + py * math.cos(ang)) for px, py in pts]
    d.polygon(rot, fill=col + (255,), outline=(234, 243, 255))


def heading(poses, i):
    if i <= 0: return 0.0
    p, q = poses[max(0, i - 1)], poses[i]
    if p["x"] == q["x"] and p["y"] == q["y"]: return 0.0
    return math.atan2(q["y"] - p["y"], q["x"] - p["x"])


def draw(base, scene, ep, pA, pB, upto, title, sx, sy, labelA, labelB):
    im = base.copy(); d = ImageDraw.Draw(im, "RGBA")
    def X(x): return x * sx
    def Y(y): return y * sy
    # planned paths
    for plan, col in ((ep["planA"], PLANA), (ep["planB"], PLANB)):
        if len(plan) > 1:
            d.line([(X(p["x"]), Y(p["y"])) for p in plan], fill=col + (210,), width=2, joint="curve")
    # buoys
    for b in scene["buoys"]:
        c = GREEN if b["color"] == "GREEN" else RED
        d.ellipse([X(b["x"]) - 4, Y(b["y"]) - 4, X(b["x"]) + 4, Y(b["y"]) + 4], fill=c + (255,), outline=(4, 16, 24))
    # obstacles
    for o in scene["obstacles"]:
        r = o["r"] * sx
        d.ellipse([X(o["x"]) - r, Y(o["y"]) - r, X(o["x"]) + r, Y(o["y"]) + r], outline=RED + (190,), width=2)
    # goals
    for g, lab in ((ep["goalA"], labelA[0]), (ep["goalB"], labelB[0])):
        d.ellipse([X(g["x"]) - 7, Y(g["y"]) - 7, X(g["x"]) + 7, Y(g["y"]) + 7], fill=GOAL + (255,), outline=(4, 16, 24), width=2)
    # trails
    for poses, col in ((pA, TA), (pB, TB)):
        trail = [(X(p["x"]), Y(p["y"])) for p in poses[:upto + 1]]
        if len(trail) > 1:
            d.line(trail, fill=col + (220,), width=3, joint="curve")
    # inter-vessel link coloured by separation
    a, b = pA[upto], pB[upto]
    sep = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    link = RED if sep <= COLLIDE_R else (AMBER if sep <= NEARMISS_R else None)
    if link is not None:
        d.line([(X(a["x"]), Y(a["y"])), (X(b["x"]), Y(b["y"]))], fill=link + (230,), width=2)
    # vessels
    tri(d, X(a["x"]), Y(a["y"]), heading(pA, upto), VA)
    tri(d, X(b["x"]), Y(b["y"]), heading(pB, upto), VB)
    if sep <= COLLIDE_R:
        mx, my = X((a["x"] + b["x"]) / 2), Y((a["y"] + b["y"]) / 2)
        d.ellipse([mx - 14, my - 14, mx + 14, my + 14], outline=RED + (255,), width=3)

    # HUD
    d.rectangle([0, 0, im.width, 26], fill=(6, 14, 26, 205))
    d.text((8, 4), title, fill=(231, 238, 252), font=FONT)
    status = f"step {a['idx']}/{ep['steps']}   vColl {ep['vesselCollisions']}   CPA {ep['minCPA']:.0f}px"
    d.text((im.width - 8, 4), status, fill=(170, 190, 230), font=FONT_S, anchor="ra")
    # legend
    d.rectangle([0, im.height - 20, im.width, im.height], fill=(6, 14, 26, 190))
    d.ellipse([8, im.height - 15, 18, im.height - 5], fill=VA + (255,))
    d.text((22, im.height - 18), labelA, fill=(210, 225, 250), font=FONT_S)
    off = 22 + d.textlength(labelA, font=FONT_S) + 18
    d.ellipse([off, im.height - 15, off + 10, im.height - 5], fill=VB + (255,))
    d.text((off + 14, im.height - 18), labelB, fill=(210, 225, 250), font=FONT_S)
    return im


def render(name, title, labelA, labelB, hold_end=12):
    with open(os.path.join(TRAJ, name + ".json")) as f:
        data = json.load(f)
    scene = data["scene"]; ep = data["episode"]
    w = int(scene["mapW"] * SCALE); h = int(scene["mapH"] * SCALE)
    sx = w / scene["mapW"]; sy = h / scene["mapH"]
    base = load_chart(w, h)
    pA = interp(ep["fA"]); pB = interp(ep["fB"])
    n = min(len(pA), len(pB)); pA, pB = pA[:n], pB[:n]
    if n > MAX_FRAMES:
        idxs = [round(i * (n - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
    else:
        idxs = list(range(n))
    frames = [draw(base, scene, ep, pA, pB, i, title, sx, sy, labelA, labelB) for i in idxs]
    frames += [frames[-1]] * hold_end
    pal_src = frames[0].convert("P", palette=Image.ADAPTIVE, colors=N_COLORS)
    gif = [im.quantize(colors=N_COLORS, palette=pal_src, dither=Image.NONE) for im in frames]
    out = os.path.join(OUT, name + ".gif")
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=120, loop=0, optimize=True, disposal=2)
    print(f"  {name}.gif ({len(frames)} frames, {w}x{h}, {os.path.getsize(out)/1024:.0f} KB) vColl={ep['vesselCollisions']} minCPA={ep['minCPA']}")


def main():
    print("Rendering BPAN two-vessel GIFs over the Bintulu chart:")
    render("multi_interaction", "BPAN multi-vessel - two agents share the port", "Vessel A (DQN)", "Vessel B (DQN)")
    render("twoway_headon", "BPAN two-way traffic - head-on in one channel", "Inbound (DQN)", "Outbound (Rule)")
    print("-> results_bintulu/gifs/{multi_interaction,twoway_headon}.gif")


if __name__ == "__main__":
    main()

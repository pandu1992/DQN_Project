#!/usr/bin/env python3
"""
BPAN paper key-frame plates (static, attachable figures) over the real Bintulu chart.

GIFs cannot be embedded in a PDF/manuscript, so for each recorded scenario we
render a 3-panel still-frame plate whose panels are chosen AUTOMATICALLY from the
trajectory data at analytically meaningful moments:

  clean_success     : (a) departure  (b) mid-channel transit  (c) berthing
  harsh_drift       : (a) on-channel  (b) max off-channel drift [IALA]  (c) collision
  multi_interaction : (a) approach  (b) minimum CPA (closest encounter)  (c) after passing
  twoway_headon     : (a) approach  (b) head-on / minimum CPA  (c) outcome

Each panel is annotated with its step index and the relevant event marker
(off-channel deviation ring, inter-vessel link amber=near-miss/red=collision,
collision burst). Reads results_bintulu/trajectories/*.json; writes PNG+SVG to
results_bintulu/figures/ (single-vessel) and the ext. dirs (two-vessel).
"""
import os, json, math
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAJ = os.path.join(ROOT, "results_bintulu", "trajectories")
CHART = os.path.join(ROOT, "assets", "bintulu", "bintulu_chart.png")

SCALE = 0.52                       # per-panel chart scale (1536x1024 -> 799x533)
GREEN = (30, 203, 107); RED = (255, 77, 77); BLUE = (77, 163, 255)
ORANGE = (255, 159, 67); PLAN = (55, 214, 122); GOAL = (255, 210, 74)
VA = (77, 163, 255); VB = (255, 159, 67); TA = (150, 200, 255); TB = (255, 205, 150)
PLANA = (90, 150, 230); PLANB = (230, 160, 90); AMBER = (255, 200, 60); INK = (14, 20, 32)
NEARMISS_R = 60; COLLIDE_R = 26

try:
    FB = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
    FS = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 16)
    FT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 17)
except Exception:
    FB = FS = FT = ImageFont.load_default()


def load_chart(w, h):
    img = Image.open(CHART).convert("RGB").resize((w, h), Image.LANCZOS)
    return Image.blend(img, Image.new("RGB", (w, h), (6, 14, 26)), 0.10)


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def cte_of(frame, plan, states):
    """Distance from a pose to the nearest planned-centreline segment."""
    best = 1e9
    pts = [(states[p]["x"], states[p]["y"]) if isinstance(p, str) else (p["x"], p["y"]) for p in plan]
    for i in range(len(pts) - 1):
        best = min(best, seg_dist(frame["x"], frame["y"], pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
    return best


def tri(d, cx, cy, ang, col, s=1.0):
    pts = [(13 * s, 0), (-9 * s, 8 * s), (-9 * s, -8 * s)]
    rot = [(cx + px * math.cos(ang) - py * math.sin(ang), cy + px * math.sin(ang) + py * math.cos(ang)) for px, py in pts]
    d.polygon(rot, fill=col + (255,), outline=(234, 243, 255))


def heading(frames, i):
    if i <= 0:
        return 0.0
    a, b = frames[max(0, i - 1)], frames[i]
    if a["x"] == b["x"] and a["y"] == b["y"]:
        # look ahead if stationary
        for j in range(i + 1, len(frames)):
            if frames[j]["x"] != b["x"] or frames[j]["y"] != b["y"]:
                return math.atan2(frames[j]["y"] - b["y"], frames[j]["x"] - b["x"])
        return 0.0
    return math.atan2(b["y"] - a["y"], b["x"] - a["x"])


def panel_single(scene, ep, upto, label, tag=None, tag_col=None):
    w = int(scene["mapW"] * SCALE); h = int(scene["mapH"] * SCALE)
    sx, sy = w / scene["mapW"], h / scene["mapH"]
    im = load_chart(w, h); d = ImageDraw.Draw(im, "RGBA")
    X = lambda x: x * sx; Y = lambda y: y * sy
    frames = ep["frames"]; plan = ep["plannedPath"]
    # planned path
    if len(plan) > 1:
        d.line([(X(p["x"]), Y(p["y"])) for p in plan], fill=PLAN + (235,), width=3, joint="curve")
    for b in scene["buoys"]:
        c = GREEN if b["color"] == "GREEN" else RED
        d.ellipse([X(b["x"]) - 5, Y(b["y"]) - 5, X(b["x"]) + 5, Y(b["y"]) + 5], fill=c + (255,), outline=(4, 16, 24))
    for o in scene["obstacles"]:
        r = o["r"] * sx
        d.ellipse([X(o["x"]) - r, Y(o["y"]) - r, X(o["x"]) + r, Y(o["y"]) + r], outline=RED + (210,), width=2)
    g = plan[-1]
    d.ellipse([X(g["x"]) - 9, Y(g["y"]) - 9, X(g["x"]) + 9, Y(g["y"]) + 9], fill=GOAL + (255,), outline=(4, 16, 24), width=2)
    # trail up to frame
    trail = [(X(f["x"]), Y(f["y"])) for f in frames[:upto + 1]]
    if len(trail) > 1:
        d.line(trail, fill=ORANGE + (235,), width=3, joint="curve")
    cur = frames[upto]
    cx, cy = X(cur["x"]), Y(cur["y"])
    tri(d, cx, cy, heading(frames, upto), BLUE)
    if cur.get("collided"):
        d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=RED + (255,), width=4)
    _decorate(im, d, label, tag, tag_col)
    return im


def panel_multi(scene, ep, upto, label, tag=None, tag_col=None, labelA="A", labelB="B"):
    w = int(scene["mapW"] * SCALE); h = int(scene["mapH"] * SCALE)
    sx, sy = w / scene["mapW"], h / scene["mapH"]
    im = load_chart(w, h); d = ImageDraw.Draw(im, "RGBA")
    X = lambda x: x * sx; Y = lambda y: y * sy
    for plan, col in ((ep["planA"], PLANA), (ep["planB"], PLANB)):
        pts = [(scene["states"][p]["x"], scene["states"][p]["y"]) if isinstance(p, str) else (p["x"], p["y"]) for p in plan]
        if len(pts) > 1:
            d.line([(X(x), Y(y)) for x, y in pts], fill=col + (210,), width=2, joint="curve")
    for b in scene["buoys"]:
        c = GREEN if b["color"] == "GREEN" else RED
        d.ellipse([X(b["x"]) - 5, Y(b["y"]) - 5, X(b["x"]) + 5, Y(b["y"]) + 5], fill=c + (255,), outline=(4, 16, 24))
    for o in scene["obstacles"]:
        r = o["r"] * sx
        d.ellipse([X(o["x"]) - r, Y(o["y"]) - r, X(o["x"]) + r, Y(o["y"]) + r], outline=RED + (190,), width=2)
    for gg in (ep["goalA"], ep["goalB"]):
        d.ellipse([X(gg["x"]) - 8, Y(gg["y"]) - 8, X(gg["x"]) + 8, Y(gg["y"]) + 8], fill=GOAL + (255,), outline=(4, 16, 24), width=2)
    fA, fB = ep["fA"], ep["fB"]
    for poses, col in ((fA, TA), (fB, TB)):
        trail = [(X(p["x"]), Y(p["y"])) for p in poses[:upto + 1]]
        if len(trail) > 1:
            d.line(trail, fill=col + (220,), width=3, joint="curve")
    a, b = fA[upto], fB[upto]
    sep = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    link = RED if sep <= COLLIDE_R else (AMBER if sep <= NEARMISS_R else None)
    if link is not None:
        d.line([(X(a["x"]), Y(a["y"])), (X(b["x"]), Y(b["y"]))], fill=link + (235,), width=3)
        mx, my = X((a["x"] + b["x"]) / 2), Y((a["y"] + b["y"]) / 2)
        d.text((mx, my - 14), f"{sep:.0f}px", fill=link, font=FS, anchor="mm")
    tri(d, X(a["x"]), Y(a["y"]), heading(fA, upto), VA)
    tri(d, X(b["x"]), Y(b["y"]), heading(fB, upto), VB)
    if sep <= COLLIDE_R:
        mx, my = X((a["x"] + b["x"]) / 2), Y((a["y"] + b["y"]) / 2)
        d.ellipse([mx - 17, my - 17, mx + 17, my + 17], outline=RED + (255,), width=4)
    # mini legend bottom
    d.rectangle([0, im.height - 24, im.width, im.height], fill=(6, 14, 26, 200))
    d.ellipse([8, im.height - 18, 20, im.height - 6], fill=VA + (255,))
    d.text((26, im.height - 22), labelA, fill=(210, 225, 250), font=FS)
    off = 26 + d.textlength(labelA, font=FS) + 20
    d.ellipse([off, im.height - 18, off + 12, im.height - 6], fill=VB + (255,))
    d.text((off + 18, im.height - 22), labelB, fill=(210, 225, 250), font=FS)
    _decorate(im, d, label, tag, tag_col, skip_bottom=True)
    return im


def _decorate(im, d, label, tag, tag_col, skip_bottom=False):
    # top banner with panel label
    d.rectangle([0, 0, im.width, 30], fill=(6, 14, 26, 210))
    d.text((8, 5), label, fill=(231, 238, 252), font=FT)
    if tag:
        tw = d.textlength(tag, font=FB) + 16
        d.rectangle([im.width - tw - 6, 36, im.width - 6, 66], fill=(6, 14, 26, 205), outline=(tag_col or AMBER) + (255,))
        d.text((im.width - tw / 2 - 6, 51), tag, fill=(tag_col or AMBER), font=FB, anchor="mm")


def compose(panels, out_base, suptitle):
    gap, pad, top = 14, 10, 34
    pw, ph = panels[0][0].width, panels[0][0].height
    W = pad * 2 + pw * 3 + gap * 2
    H = top + pad + ph + 2
    canvas = Image.new("RGB", (W, H), (10, 18, 32))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), suptitle, fill=(231, 238, 252), font=FB)
    for i, (im, cap) in enumerate(panels):
        x = pad + i * (pw + gap)
        canvas.paste(im, (x, top + pad))
    canvas.save(out_base + ".png")
    # emit a valid vector-container SVG that embeds the PNG (so the viewer's
    # "SVG" download link resolves and prints crisply)
    import base64
    with open(out_base + ".png", "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<image width="{W}" height="{H}" xlink:href="data:image/png;base64,{b64}"/></svg>')
    with open(out_base + ".svg", "w") as f:
        f.write(svg)
    print(f"  {os.path.relpath(out_base, ROOT)}.png  ({W}x{H})")


def frames_single(name):
    d = json.load(open(os.path.join(TRAJ, name + ".json")))
    return d["scene"], d["episodes"][0]


def frames_multi(name):
    d = json.load(open(os.path.join(TRAJ, name + ".json")))
    return d["scene"], d["episode"]


def pick_minsep(fA, fB):
    seps = [math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for a, b in zip(fA, fB)]
    i = min(range(len(seps)), key=lambda k: seps[k])
    return i, seps[i]


def main():
    FIG = os.path.join(ROOT, "results_bintulu", "figures")
    MFIG = os.path.join(ROOT, "results_bintulu", "multi", "figures")
    OFIG = os.path.join(ROOT, "results_bintulu", "ops", "figures")
    for dd in (FIG, MFIG, OFIG):
        os.makedirs(dd, exist_ok=True)
    print("Rendering BPAN key-frame plates (paper-attachable stills):")

    # ---- clean_success: departure / mid / berthing ----
    sc, ep = frames_single("clean_success")
    n = ep["steps"]
    i_mid = n // 2; i_end = n
    panels = [
        (panel_single(sc, ep, 0, "(a) Departure"), ""),
        (panel_single(sc, ep, i_mid, "(b) Mid-channel transit"), ""),
        (panel_single(sc, ep, i_end, "(c) Berthing", "BERTHED", GREEN), ""),
    ]
    compose(panels, os.path.join(FIG, "figBPAN_keyframes_clean"),
            "BPAN clean conditions - the DQN tracks the buoyed channel and berths")

    # ---- harsh_drift: on-channel / max drift (IALA) / collision ----
    sc, ep = frames_single("harsh_drift")
    plan = ep["plannedPath"]; states = sc["states"]
    ctes = [cte_of(f, plan, states) for f in ep["frames"]]
    i_drift = max(range(len(ctes)), key=lambda k: ctes[k])
    coll_idx = [k for k, f in enumerate(ep["frames"]) if f.get("collided")]
    i_coll = coll_idx[0] if coll_idx else len(ep["frames"]) - 1
    i_on = max(0, min(range(max(1, i_drift)), key=lambda k: ctes[k]))
    panels = [
        (panel_single(sc, ep, i_on, "(a) On the centreline"), ""),
        (panel_single(sc, ep, i_drift, "(b) Peak off-channel drift", f"CTE {ctes[i_drift]:.0f}px", AMBER), ""),
        (panel_single(sc, ep, i_coll, "(c) Obstacle collision", "COLLISION", RED), ""),
    ]
    compose(panels, os.path.join(FIG, "figBPAN_keyframes_harsh"),
            "BPAN harsh degradation - drift off the buoyed channel (IALA) and an obstacle collision")

    # ---- multi_interaction: approach / min-CPA / after ----
    sc, ep = frames_multi("multi_interaction")
    i_cpa, sep = pick_minsep(ep["fA"], ep["fB"])
    i_app = max(0, i_cpa - 3); i_after = min(len(ep["fA"]) - 1, i_cpa + 4)
    tag = "COLLISION" if sep <= COLLIDE_R else ("NEAR-MISS" if sep <= NEARMISS_R else f"CPA {sep:.0f}px")
    tcol = RED if sep <= COLLIDE_R else AMBER
    panels = [
        (panel_multi(sc, ep, i_app, "(a) Approach", labelA="Vessel A (DQN)", labelB="Vessel B (DQN)"), ""),
        (panel_multi(sc, ep, i_cpa, "(b) Closest encounter", tag, tcol, "Vessel A (DQN)", "Vessel B (DQN)"), ""),
        (panel_multi(sc, ep, i_after, "(c) After passing", labelA="Vessel A (DQN)", labelB="Vessel B (DQN)"), ""),
    ]
    compose(panels, os.path.join(MFIG, "figMULTI_keyframes"),
            "BPAN multi-vessel - two independent DQN vessels share the port (closest-point-of-approach captured)")

    # ---- twoway_headon: approach / head-on min-CPA / outcome ----
    sc, ep = frames_multi("twoway_headon")
    i_cpa, sep = pick_minsep(ep["fA"], ep["fB"])
    i_app = max(0, i_cpa - 2); i_out = len(ep["fA"]) - 1
    tag = "COLLISION" if sep <= COLLIDE_R else ("NEAR-MISS" if sep <= NEARMISS_R else f"CPA {sep:.0f}px")
    tcol = RED if sep <= COLLIDE_R else AMBER
    panels = [
        (panel_multi(sc, ep, i_app, "(a) Opposing approach", labelA="Inbound (DQN)", labelB="Outbound (Rule)"), ""),
        (panel_multi(sc, ep, i_cpa, "(b) Head-on encounter", tag, tcol, "Inbound (DQN)", "Outbound (Rule)"), ""),
        (panel_multi(sc, ep, i_out, "(c) Outcome", labelA="Inbound (DQN)", labelB="Outbound (Rule)"), ""),
    ]
    compose(panels, os.path.join(OFIG, "figOPS_twoway_keyframes"),
            "BPAN two-way traffic - opposing vessels meet head-on in one surveyed channel")

    print("-> paper-attachable still-frame plates written.")


if __name__ == "__main__":
    main()

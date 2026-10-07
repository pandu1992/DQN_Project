#!/usr/bin/env python3
"""
Render BPAN GIFs: a trained DQN vessel navigating over the real Bintulu chart.
Reads results_bintulu/trajectories/*.json (recorded by the Node script) and the
chart image, composites smooth animated frames with Pillow, and writes GIFs to
results_bintulu/gifs/.
"""
import os, json, math
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAJ = os.path.join(ROOT, "results_bintulu", "trajectories")
CHART = os.path.join(ROOT, "assets", "bintulu", "bintulu_chart.png")
OUT = os.path.join(ROOT, "results_bintulu", "gifs")
os.makedirs(OUT, exist_ok=True)

# render at a downscaled size to keep GIFs small (web-friendly)
SCALE = 0.40                     # 1536x1024 -> 614x410
INTERP = 2                       # sub-frames between consecutive macro poses (smoothness)
MAX_FRAMES = 90                  # cap total animation frames (subsample if longer)
N_COLORS = 64                    # GIF palette size
GREEN = (30, 203, 107); RED = (255, 77, 77); BLUE = (77, 163, 255)
ORANGE = (255, 159, 67); PLAN = (55, 214, 122); GOAL = (255, 210, 74)
INK = (18, 18, 18)

try:
    FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_S = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
except Exception:
    FONT = ImageFont.load_default(); FONT_S = ImageFont.load_default()


def load_chart(w, h):
    img = Image.open(CHART).convert("RGB").resize((w, h), Image.LANCZOS)
    # slight darken so overlays read
    ov = Image.new("RGB", (w, h), (6, 14, 26))
    return Image.blend(img, ov, 0.10)


def lerp(a, b, t): return a + (b - a) * t


def interp_poses(frames):
    """Expand macro-step poses into INTERP sub-frames for smooth motion."""
    out = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        for k in range(INTERP):
            t = k / INTERP
            out.append({"x": lerp(a["x"], b["x"], t), "y": lerp(a["y"], b["y"], t),
                        "collided": b["collided"] if k == INTERP - 1 else 0, "idx": i})
    out.append({"x": frames[-1]["x"], "y": frames[-1]["y"], "collided": frames[-1]["collided"], "idx": len(frames) - 1})
    return out


def draw_frame(base, scene, ep, poses, upto, title, sx, sy):
    im = base.copy()
    d = ImageDraw.Draw(im, "RGBA")
    def X(x): return x * sx
    def Y(y): return y * sy

    # planned path (green)
    pp = ep["plannedPath"]
    if len(pp) > 1:
        d.line([(X(p["x"]), Y(p["y"])) for p in pp], fill=PLAN + (235,), width=3, joint="curve")
    # buoys
    for b in scene["buoys"]:
        c = GREEN if b["color"] == "GREEN" else RED
        d.ellipse([X(b["x"]) - 4, Y(b["y"]) - 4, X(b["x"]) + 4, Y(b["y"]) + 4], fill=c + (255,), outline=(4, 16, 24))
    # obstacles
    for o in scene["obstacles"]:
        r = o["r"] * sx
        d.ellipse([X(o["x"]) - r, Y(o["y"]) - r, X(o["x"]) + r, Y(o["y"]) + r], outline=RED + (210,), width=2)
    # goal berth
    g = ep["plannedPath"][-1]
    d.ellipse([X(g["x"]) - 8, Y(g["y"]) - 8, X(g["x"]) + 8, Y(g["y"]) + 8], fill=GOAL + (255,), outline=(4, 16, 24), width=2)
    d.text((X(g["x"]), Y(g["y"])), "B", fill=INK, font=FONT_S, anchor="mm")

    # actual trail up to current sub-frame (orange)
    trail = [(X(p["x"]), Y(p["y"])) for p in poses[:upto + 1]]
    if len(trail) > 1:
        d.line(trail, fill=ORANGE + (235,), width=3, joint="curve")

    # vessel (triangle, oriented along recent motion)
    cur = poses[upto]
    ang = 0.0
    if upto > 0:
        prev = poses[max(0, upto - 1)]
        if (cur["x"] != prev["x"]) or (cur["y"] != prev["y"]):
            ang = math.atan2(cur["y"] - prev["y"], cur["x"] - prev["x"])
    cx, cy = X(cur["x"]), Y(cur["y"])
    pts = [(10, 0), (-7, 6), (-7, -6)]
    rot = [(cx + px * math.cos(ang) - py * math.sin(ang), cy + px * math.sin(ang) + py * math.cos(ang)) for px, py in pts]
    d.polygon(rot, fill=BLUE + (255,), outline=(234, 243, 255))
    if cur["collided"]:
        d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], outline=RED + (255,), width=3)

    # HUD banner
    d.rectangle([0, 0, im.width, 26], fill=(6, 14, 26, 205))
    d.text((8, 4), title, fill=(231, 238, 252), font=FONT)
    status = f"step {cur['idx']}/{ep['steps']}   coll {ep['collisions']}   IALA {ep['iala']}   CTE {ep['cte']}"
    d.text((im.width - 8, 4), status, fill=(170, 190, 230), font=FONT_S, anchor="ra")
    # outcome tag at end
    if upto >= len(poses) - 1:
        tag = "BERTHED" if ep["success"] else "TIMEOUT"
        col = (55, 214, 122) if ep["success"] else (255, 93, 93)
        d.rectangle([im.width / 2 - 70, im.height / 2 - 18, im.width / 2 + 70, im.height / 2 + 18], fill=(6, 14, 26, 180))
        d.text((im.width / 2, im.height / 2), tag, fill=col, font=FONT, anchor="mm")
    return im


def render(name, title, hold_end=10):
    with open(os.path.join(TRAJ, name + ".json")) as f:
        data = json.load(f)
    scene = data["scene"]; ep = data["episodes"][0]
    w = int(scene["mapW"] * SCALE); h = int(scene["mapH"] * SCALE)
    sx = w / scene["mapW"]; sy = h / scene["mapH"]
    base = load_chart(w, h)
    poses = interp_poses(ep["frames"])
    # subsample pose indices so total frames <= MAX_FRAMES
    n = len(poses)
    if n > MAX_FRAMES:
        idxs = [round(i * (n - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
    else:
        idxs = list(range(n))
    frames = [draw_frame(base, scene, ep, poses, i, title, sx, sy) for i in idxs]
    frames += [frames[-1]] * hold_end     # hold on the final frame
    # quantize to a FIXED shared palette (from the chart base) so inter-frame
    # diffs are small and the GIF stays compact
    pal_src = frames[0].convert("P", palette=Image.ADAPTIVE, colors=N_COLORS)
    gif_frames = [im.quantize(colors=N_COLORS, palette=pal_src, dither=Image.NONE) for im in frames]
    out = os.path.join(OUT, name + ".gif")
    gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:], duration=110, loop=0, optimize=True, disposal=2)
    size_kb = os.path.getsize(out) / 1024
    print(f"  {name}.gif  ({len(frames)} frames, {w}x{h}, {size_kb:.0f} KB)  {ep['start']}->{ep['goal']} success={ep['success']}")


def main():
    print("Rendering BPAN GIFs over the Bintulu chart:")
    render("clean_success", "BPAN - clean conditions: DQN berths")
    render("mid_transit", "BPAN - mid degradation: DQN transit")
    render("harsh_drift", "BPAN - harsh degradation: drift & collision")
    print("-> results_bintulu/gifs/*.gif")


if __name__ == "__main__":
    main()

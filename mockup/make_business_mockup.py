"""Builds the SMALL BUSINESS planner listing mockup (2400x1800), same style as make_budget_mockup.py.
Usage: python3 make_business_mockup.py <business_source.jpg> <budget_source.jpg> <out.png>
<business_source> is the original 1024x1024 Business mockup (tablets rotated 11 degrees): the two screens are
straightened and re-framed. <budget_source> is only used for the four app icons (sharper artwork)."""
import sys
import math
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SRC, ICONS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 2400, 1800
src = Image.open(SRC).convert("RGB")             # the original Business mockup (1024x1024, tablets rotated 11 degrees)
icons_src = Image.open(ICONS).convert("RGB")     # the Budget original: same four app icons, higher resolution
ROT = math.radians(11.0)                         # both tablets are rotated 11 degrees clockwise in the original
COVER_TL, COVER_W, COVER_H = (528.25, 314.0), 439.65, 586.5     # cover screen: top-left corner and size (source px)
REVIEW_TL, REVIEW_W, REVIEW_H = (205.44, 250.16), 435.73, 580.97  # monthly-review screen (only ~70% is visible)
REVIEW_VISIBLE = 612                              # in the 2x straightened image, columns >= 612 are hidden behind the cover tablet
SCALE = 2


def straighten(img, tl, w, h, scale=SCALE):
    """Cut a rotated screen out of the source and return it upright (exact: pure rotation, no perspective)."""
    c, s = math.cos(ROT), math.sin(ROT)
    tl = np.array(tl); u = np.array([c, s]); v = np.array([-s, c])
    quad = [tl, tl + u * w, tl + u * w + v * h, tl + v * h]
    W, H = int(w * scale), int(h * scale)
    out = [(0, 0), (W, 0), (W, H), (0, H)]
    A, B = [], []
    for (x, y), (px, py) in zip(out, quad):
        A.append([x, y, 1, 0, 0, 0, -px * x, -px * y]); B.append(px)
        A.append([0, 0, 0, x, y, 1, -py * x, -py * y]); B.append(py)
    co = np.linalg.solve(np.array(A, float), np.array(B, float))
    return img.transform((W, H), Image.PERSPECTIVE, tuple(co), Image.BICUBIC)


def _k(sigma):
    r = max(1, int(3 * sigma)); x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); return k / k.sum()


def _conv(a, k, axis):
    r = len(k) // 2
    pad = [(0, 0)] * a.ndim; pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        sl = [slice(None)] * a.ndim; sl[axis] = slice(i, i + a.shape[axis])
        out += w * p[tuple(sl)]
    return out


def fill_region(img, mask, sx, sy, ink=None, tex=None):
    """Paint `mask` (True = unknown) from the surrounding known pixels (normalised convolution).
    `tex` is an optional high-pass paper-grain array that is tiled over the filled pixels."""
    a = np.asarray(img.convert("RGB"), float)
    known = (~mask).astype(float)
    if ink is not None:
        known *= (a.mean(2) >= ink)
    res, ok = None, None
    for sxx, syy in ((sx, sy), (sx * 4, sy * 4), (60, 60)):
        num = a * known[..., None]; den = known.copy()
        for axis, sg in ((1, sxx), (0, syy)):
            k = _k(sg)
            num = _conv(num, k, axis); den = _conv(den, k, axis)
        f = num / np.maximum(den, 1e-6)[..., None]
        if res is None:
            res, ok = f, den > 0.05
        else:
            res = np.where(ok[..., None], res, f); ok = ok | (den > 0.05)
    if tex is not None:
        th, tw = tex.shape
        yy = (np.arange(a.shape[0])[:, None] + 7) % th; xx = (np.arange(a.shape[1])[None, :] + 13) % tw
        res = res + tex[yy, xx][..., None]
    soft = _conv(_conv(mask.astype(float), _k(1.5), 0), _k(1.5), 1)[..., None]   # feather the edge
    out = a * (1 - soft) + res * soft
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))



cover_raw = straighten(src, COVER_TL, COVER_W, COVER_H)       # 879 x 1173
review_raw = straighten(src, REVIEW_TL, REVIEW_W, REVIEW_H)   # 871 x 1161

# paper grain from a clean patch of the cover, used to texture anything repainted there
_clean = np.asarray(cover_raw.crop((100, 600, 340, 760)).convert("L"), float)
_bl = np.asarray(Image.fromarray(_clean.astype("uint8")).filter(ImageFilter.GaussianBlur(4)), float)
GRAIN = _clean - _bl


def cover_year(img, text="2026 - 2027"):
    """Replace the single '2027' (and its two thin rules) on the cover with the two-year range."""
    cx = 418.5                                         # the cover text column is centred here (the month-tab strip takes the right edge)
    m = np.zeros((img.height, img.width), bool); m[466:584, 290:550] = True
    img = fill_region(img, m, 10, 10, tex=GRAIN)
    k = 5
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for yy in (484, 563):                              # the two thin rules, widened to suit the longer text
        ld.rectangle(((cx - 130) * k, yy * k, (cx + 130) * k, (yy + 1.6) * k), fill=(206, 182, 154, 255))
    f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", int(33 * k))
    tr = 8.5 * k
    widths = [ld.textlength(ch, font=f) for ch in text]
    x = cx * k - (sum(widths) + tr * (len(text) - 1)) / 2
    for ch, w in zip(text, widths):
        ld.text((x, 540 * k), ch, font=f, fill=(96, 76, 64, 255), anchor="ls")
        x += w + tr
    layer = layer.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")


cover = cover_year(cover_raw).crop((4, 4, cover_raw.width - 4, cover_raw.height - 4))
# the right-hand third of the review page is hidden behind the cover tablet in the original: blank it so no
# stray bezel/cover pixels can leak; the layout below keeps the cover tablet in front of that strip
review = review_raw.copy()
ImageDraw.Draw(review).rectangle((REVIEW_VISIBLE, 0, review.width, review.height), fill=(247, 245, 242))
review = review.crop((4, 4, review.width - 4, review.height - 4))
daily = review                                         # (name kept so the layout code below reads the same)

# the four app icons straight from the Budget original (Apple Calendar, Reminders, Google Calendar, ChatGPT)
icon_acal = icons_src.crop((778, 478, 870, 570))
icon_rem = icons_src.crop((895, 478, 987, 570))
icon_gcal = icons_src.crop((1013, 478, 1105, 570))
icon_gpt = icons_src.crop((1131, 478, 1223, 570))

F = "/usr/share/fonts/truetype/"
serif = F + "liberation/LiberationSerif-Regular.ttf"
sans = F + "dejavu/DejaVuSans.ttf"

# --- background: warm greige with soft window shadow
bg = Image.new("RGB", (W, H), (200, 189, 178))
px = ImageDraw.Draw(bg)
for y in range(H):
    t = y / H
    px.line([(0, y), (W, y)], fill=(int(206 - 20 * t), int(195 - 22 * t), int(184 - 22 * t)))
sh = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(sh)
d.polygon([(0, 0), (480, 0), (140, H)], fill=70)
d.polygon([(640, 0), (760, 0), (420, H)], fill=34)
d.polygon([(1750, 0), (W, 0), (W, 560)], fill=50)
sh = sh.filter(ImageFilter.GaussianBlur(110))
bg = Image.composite(Image.new("RGB", (W, H), (112, 98, 88)), bg, sh)
# window light falling diagonally across the surface
lt = Image.new("L", (W, H), 0)
ld = ImageDraw.Draw(lt)
ld.polygon([(480, 0), (640, 0), (300, H), (140, H)], fill=120)
ld.polygon([(760, 0), (1100, 0), (760, H), (420, H)], fill=70)
ld.polygon([(1100, 0), (1750, 0), (W, 560), (W, 900), (1500, 0)], fill=40)
lt = lt.filter(ImageFilter.GaussianBlur(90))
bg = Image.composite(Image.new("RGB", (W, H), (236, 227, 216)), bg, lt.point(lambda v: int(v * .55)))
canvas = bg.convert("RGBA")

# ---------- sunlight through a window: hard-edged light panes + soft god-rays
K = 0.5  # light travels down-right, so patches lean right as they fall
def skew(x0, y0, x1, y1):
    return [(x0 + K * y0, y0), (x1 + K * y0, y0), (x1 + K * y1, y1), (x0 + K * y1, y1)]

panes = Image.new("L", (W, H), 0)
pd_ = ImageDraw.Draw(panes)
for r in range(4):
    for c in range(3):
        x0 = -330 + c * 400
        y0 = -120 + r * 470
        pd_.polygon(skew(x0, y0, x0 + 340, y0 + 420), fill=255)
fall = Image.linear_gradient("L").resize((W, H)).rotate(0)           # top -> bottom
fx = Image.linear_gradient("L").rotate(90).resize((W, H))            # right -> left
fall = Image.eval(ImageChops.add(fall.point(lambda v: v // 2), fx.point(lambda v: v // 2)), lambda v: v)
fall = fall.point(lambda v: max(0, 255 - int(v * 1.2)))             # bright top-left, gone by centre
sharp = ImageChops.multiply(panes.filter(ImageFilter.GaussianBlur(5)), fall)
glow = ImageChops.multiply(panes.filter(ImageFilter.GaussianBlur(70)), fall)
sun = Image.new("RGBA", (W, H), (255, 240, 214, 255))
canvas = Image.composite(sun, canvas, sharp.point(lambda v: int(v * .42)))
canvas = Image.composite(sun, canvas, glow.point(lambda v: int(v * .22)))

rays = Image.new("L", (W, H), 0)
rd = ImageDraw.Draw(rays)
for x0, wd, a in [(120, 90, 150), (420, 150, 110), (760, 70, 130), (1060, 190, 80), (1500, 110, 90), (1900, 160, 70)]:
    rd.polygon(skew(x0 - 300, -50, x0 - 300 + wd, H + 50), fill=a)
rays = rays.filter(ImageFilter.GaussianBlur(34))
haze = Image.linear_gradient("L").resize((W, H)).point(lambda v: 255 - int(v * .8))
rays = ImageChops.multiply(rays, haze)
canvas = Image.composite(Image.new("RGBA", (W, H), (255, 238, 210, 255)), canvas, rays.point(lambda v: int(v * .22)))

# a touch darker overall (background only: the tablets are placed after this)
canvas = ImageChops.multiply(canvas.convert("RGB"), Image.new("RGB", (W, H), (222, 220, 218))).convert("RGBA")


def spaced(draw, xy, text, font, fill, tracking, anchor_center=True):
    widths = [draw.textlength(c, font=font) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x, y = xy
    if anchor_center:
        x -= total / 2
    for c, w in zip(text, widths):
        draw.text((x, y), c, font=font, fill=fill)
        x += w + tracking


def tablet(screen, height, angle):
    pad = int(height * 0.028)
    sh_ = height - 2 * pad
    sw = int(screen.width * sh_ / screen.height)
    scr = screen.resize((sw, sh_), Image.LANCZOS)
    # high-resolution source: resample cleanly, only a light unsharp to keep small print crisp
    scr = scr.filter(ImageFilter.UnsharpMask(radius=1.6, percent=85, threshold=2))
    w, h = sw + 2 * pad, height
    m = Image.new("L", (sw, sh_), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, sw - 1, sh_ - 1), radius=int(pad * 1.1), fill=255)
    dev = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(dev).rounded_rectangle((0, 0, w - 1, h - 1), radius=int(pad * 2.2), fill=(14, 14, 14, 255))
    ImageDraw.Draw(dev).rounded_rectangle((3, 3, w - 4, h - 4), radius=int(pad * 2.2), outline=(95, 92, 90, 255), width=3)
    # bezel shading: slightly lifted toward the light (top-left)
    bz = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bzd = ImageDraw.Draw(bz)
    for i in range(0, w + h, 4):
        k = max(0, 1 - i / (w + h) * 2.2)
        bzd.line([(i, 0), (0, i)], fill=(255, 255, 255, int(26 * k)), width=5)
    bzm = Image.new("L", (w, h), 0)
    ImageDraw.Draw(bzm).rounded_rectangle((0, 0, w - 1, h - 1), radius=int(pad * 2.2), fill=255)
    dev.paste(bz, (0, 0), Image.composite(bz.split()[3], Image.new("L", (w, h), 0), bzm))
    # rim light on the top-left metal edge, falling off to the bottom-right
    rim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(rim).rounded_rectangle((1, 1, w - 2, h - 2), radius=int(pad * 2.2), outline=(214, 208, 202, 255), width=3)
    fade = Image.new("L", (w, h), 0)
    fd = ImageDraw.Draw(fade)
    for i in range(0, w + h, 4):
        fd.line([(i, 0), (0, i)], fill=int(255 * max(0, 1 - i / (w + h) * 1.6)), width=5)
    rim.putalpha(Image.composite(rim.split()[3], Image.new("L", (w, h), 0), fade))
    dev.alpha_composite(rim)
    dev.paste(scr, (pad, pad), m)
    gl = Image.new("RGBA", (sw, sh_), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl)
    gd.polygon([(sw * .55, 0), (sw, 0), (sw, sh_ * .35), (sw * .1, sh_)], fill=(255, 255, 255, 7))
    gd.polygon([(0, sh_ * .6), (sw * .45, 0), (sw * .6, 0), (0, sh_ * .8)], fill=(255, 255, 255, 4))
    dev.paste(gl, (pad, pad), Image.composite(gl.split()[3], Image.new("L", gl.size, 0), m))
    ImageDraw.Draw(dev).ellipse((w / 2 - 5, pad / 2 - 5, w / 2 + 5, pad / 2 + 5), fill=(38, 38, 42, 255))
    return dev.rotate(angle, resample=Image.BICUBIC, expand=True)


def drop(shape_alpha, x, y, layers):
    """Composite layered shadows (offset_x, offset_y, blur, opacity) under a shape."""
    global canvas
    for ox, oy, blur, op in layers:
        shd = Image.new("RGBA", shape_alpha.size, (46, 32, 24, 0))
        shd.putalpha(shape_alpha.point(lambda v, op=op: int(v * op)))
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        lay.paste(shd, (x + ox, y + oy), shd)
        canvas = Image.alpha_composite(canvas, lay.filter(ImageFilter.GaussianBlur(blur)))


def place(dev, cx, cy):
    global canvas
    x, y = int(cx - dev.width / 2), int(cy - dev.height / 2)
    drop(dev.split()[3], x, y, [(90, 130, 80, .26), (34, 52, 30, .34), (10, 16, 9, .5), (3, 5, 3, .55)])
    canvas.alpha_composite(dev, (x, y))


place(tablet(review, 1160, 4), 892, 1075)       # monthly review on the left, behind
place(tablet(cover, 1280, -1.5), 1462, 1055)    # cover on the right, in front (hides the unseen strip of the review page)

# pencil lying on the cover
pen = Image.new("RGBA", (900, 30), (0, 0, 0, 0))
pd = ImageDraw.Draw(pen)
body = Image.new("RGBA", (860, 30), (0, 0, 0, 0))
bd = ImageDraw.Draw(body)
for yy in range(30):
    k = 1 - abs(yy - 11) / 18
    c = int(212 + 43 * max(k, 0))
    bd.line([(0, yy), (860, yy)], fill=(c, c, int(c - 3), 255))
mk = Image.new("L", (860, 30), 0)
ImageDraw.Draw(mk).rounded_rectangle((0, 0, 859, 29), radius=14, fill=255)
pen.paste(body, (0, 0), mk)
pd.polygon([(850, 4), (900, 15), (850, 26)], fill=(232, 228, 222, 255))
pd.polygon([(888, 12), (900, 15), (888, 18)], fill=(60, 56, 54, 255))
pd.rectangle((120, 5, 720, 7), fill=(255, 255, 255, 255))
pen = pen.rotate(-100, expand=True, resample=Image.BICUBIC)
drop(pen.split()[3], 1985, 690, [(36, 52, 22, .30), (14, 20, 7, .45), (4, 6, 3, .6)])
canvas.alpha_composite(pen, (1985, 690))


# "+1000 STICKERS" seal
sx, sy, sr = 545, 300, 97
seal = Image.new("RGBA", (sr * 2 + 40, sr * 2 + 40), (0, 0, 0, 0))
sd = ImageDraw.Draw(seal)
sc = sr + 20
for i in range(26):
    a = i * 2 * math.pi / 26
    sd.ellipse((sc + (sr - 6) * math.cos(a) - 12, sc + (sr - 6) * math.sin(a) - 12, sc + (sr - 6) * math.cos(a) + 12, sc + (sr - 6) * math.sin(a) + 12), fill=(122, 88, 72, 255))
sd.ellipse((sc - sr + 6, sc - sr + 6, sc + sr - 6, sc + sr - 6), fill=(122, 88, 72, 255))
sd.ellipse((sc - sr + 18, sc - sr + 18, sc + sr - 18, sc + sr - 18), outline=(236, 220, 200, 255), width=2)
tf = ImageFont.truetype(serif, 52); sf = ImageFont.truetype(sans, 16)
cream = (250, 242, 232, 255)
sd.text((sc, sc - 18), "+1000", font=tf, fill=cream, anchor="mm")
spaced(sd, (sc, sc + 18), "STICKERS", sf, cream, 4)
sd.line([(sc - 45, sc + 12), (sc + 45, sc + 12)], fill=cream, width=2)
seal = seal.rotate(10, expand=True, resample=Image.BICUBIC)
drop(seal.split()[3], sx - seal.width // 2, sy - seal.height // 2, [(30, 46, 22, .3), (10, 16, 8, .45), (3, 5, 3, .55)])
canvas.alpha_composite(seal, (sx - seal.width // 2, sy - seal.height // 2))

d = ImageDraw.Draw(canvas)
ink = (96, 70, 58)
spaced(d, (W / 2, 44), "2026 - 2027   \u2022   +1000 STICKERS", ImageFont.truetype(serif, 40), ink, 9)
spaced(d, (W / 2, 94), "SMALL BUSINESS PLANNER", ImageFont.truetype(serif, 92), ink, 13)
spaced(d, (W / 2, 232), "A CALM SYSTEM FOR A SMALL BUSINESS", ImageFont.truetype(sans, 29), (252, 249, 245), 8)
# app icons (original artwork, cropped from the source mockup)
IC = 88
tiles = [i.resize((IC, IC), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 90, 0)) for i in (icon_acal, icon_rem, icon_gcal, icon_gpt)]
gap = 22
lab_font = ImageFont.truetype(sans, 19)
LEFT = 505                                         # left edge of the cover tablet
lab_w = sum(d.textlength(c, font=lab_font) for c in "WORKS WITH") + 5 * 9
x0 = 1850 - (IC * 4 + gap * 3)                   # icons sit above the cover tablet, on the right
mk_i = Image.new("L", (IC, IC), 0); ImageDraw.Draw(mk_i).rounded_rectangle((0, 0, IC - 1, IC - 1), radius=21, fill=255)
x = x0
for ic in tiles:
    tile = Image.new("RGBA", (IC, IC)); tile.paste(ic, (0, 0)); tile.putalpha(mk_i)
    drop(mk_i, x, 296, [(2, 4, 3, .35), (6, 12, 12, .25)])
    canvas.alpha_composite(tile, (x, 296))
    x += IC + gap
d = ImageDraw.Draw(canvas)
spaced(d, (W / 2, 1728), "dreamwy", ImageFont.truetype(serif, 46), ink, 11)
import random
out = canvas.convert("RGB")
vg = Image.new("L", (W, H), 0)
ImageDraw.Draw(vg).ellipse((-W * .25, -H * .25, W * 1.25, H * 1.25), fill=255)
vg = vg.filter(ImageFilter.GaussianBlur(260))
out = Image.composite(out, Image.new("RGB", (W, H), (150, 134, 122)), vg.point(lambda v: 150 + v * 105 // 255))
nz = Image.effect_noise((W, H), 14).convert("RGB")
out = Image.blend(out, nz, 0.025)
out.save(OUT)

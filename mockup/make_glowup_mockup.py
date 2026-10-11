"""Builds the GLOW-UP planner listing mockup (2400x1800): same style as the other mockups, but with a blush-rose palette,
the original orientation (cover left, daily page right), the stylus on the left and the badge top-right.
Usage: python3 make_glowup_mockup.py <glowup_source.jpg> <glowup_cover.png> <out.png>
<glowup_source> is the original 2000x2000 Glow-Up mockup (daily page + app icons); <glowup_cover> is the clean cover page."""
import sys
import math
import numpy as np
try:                                  # opencv-contrib (xphoto) gives patch-based inpainting; optional, falls back to a blur fill
    import cv2
    _ = cv2.xphoto.INPAINT_SHIFTMAP
except Exception:
    cv2 = None
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SRC, COVER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
VARIANT = "a"
W, H = 2400, 1800
src = Image.open(SRC).convert("RGB")             # the original 2000x2000 Glow-Up mockup (daily page + app icons)
DAILY_BOX = (985, 642, 1825, 1758)   # daily page screen only (variant a)
TIP, END = np.array([707.0, 1775.5]), np.array([1056.7, 943.3])   # the stylus lying on the original


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


def pencil_mask(size, box, hl=48, hr=26, ext=6):
    """Mask of the stylus (and its soft shadow, mostly on the left) cropped to `box`."""
    u = (END - TIP) / np.linalg.norm(END - TIP); n = np.array([-u[1], u[0]])
    n = n if n[0] > 0 else -n
    p0, p1 = TIP - u * 10, END + u * ext
    poly = [tuple(p0 - n * hl), tuple(p1 - n * hl), tuple(p1 + n * hr), tuple(p0 + n * hr)]
    m = Image.new("L", size, 0); ImageDraw.Draw(m).polygon(poly, fill=255)
    return np.asarray(m.crop(box)) > 0


# ---- cover
import os
GOLD = (196, 150, 98, 255)
if os.path.exists(COVER):
    # the clean cover supplied by the author (no stylus, nothing hidden)
    cv_raw = Image.open(COVER).convert("RGB")                      # 495 x 657
    cv_raw = cv_raw.crop((1, 2, cv_raw.width - 1, cv_raw.height))  # trims the dark screenshot border; ~3:4 like the screens
    K2 = 2
    cover = cv_raw.resize((cv_raw.width * K2, cv_raw.height * K2), Image.LANCZOS)
    YEAR_CX, YEAR_BASE, YEAR_BOX = 246 * K2, 197.8 * K2, (222 * K2, 185 * K2, 270 * K2, 203 * K2)
else:
    # provisional: crop the visible part of the cover from the original and paint the stylus out
    FB = (175, 664, 953, 1701)                                     # 778 x 1037 = 3:4
    cv_raw = src.crop(FB); K2 = 1
    cm = pencil_mask(src.size, FB, hl=34, hr=24)
    if cv2 is not None:
        m8 = cv2.dilate(cm.astype("uint8") * 255, np.ones((3, 3), np.uint8))
        dst = np.zeros((cv_raw.height, cv_raw.width, 3), np.uint8)
        cv2.xphoto.inpaint(cv2.cvtColor(np.asarray(cv_raw), cv2.COLOR_RGB2BGR), 255 - m8, dst, cv2.xphoto.INPAINT_SHIFTMAP)
        cv_raw = Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    cover = cv_raw
    YEAR_CX, YEAR_BASE, YEAR_BOX = 596 - FB[0], 957 - FB[1], (548 - FB[0], 934 - FB[1], 646 - FB[0], 965 - FB[1])


def cover_year(img, text="2026 - 2027"):
    """Replace the gold '2027' above the title with '2026 - 2027' (same colour, quiet serif, same spacing)."""
    x0, y0, x1, y1 = YEAR_BOX
    m = np.zeros((img.height, img.width), bool); m[y0:y1, x0:x1] = True
    img = fill_region(img, m, 7 * K2, 7 * K2)
    k = 4
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", int(13.8 * (K2 if K2 > 1 else 1.45) * k))
    tr = 3.6 * (K2 if K2 > 1 else 1.45) * k
    widths = [ld.textlength(ch, font=f) for ch in text]
    x = YEAR_CX * k - (sum(widths) + tr * (len(text) - 1)) / 2
    for ch, w in zip(text, widths):
        ld.text((x, YEAR_BASE * k), ch, font=f, fill=GOLD, anchor="ls")
        x += w + tr
    layer = layer.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")


cover = cover_year(cover)

# ---- daily page: take the stylus off, then re-typeset the schedule time labels it was hiding (7-10 AM)
daily = src.crop(DAILY_BOX)
ox, oy = DAILY_BOX[0], DAILY_BOX[1]
pm = pencil_mask(src.size, DAILY_BOX, hl=50, hr=26, ext=6)
pm[:926 - oy, :] = False                                           # nothing above the pencil's cap
zone = np.zeros_like(pm); zone[936 - oy:1084 - oy, 1024 - ox:1066 - ox] = True    # the time-label column, rows 7-10 AM (re-typeset below)
margin = pm.copy(); margin[:, 1026 - ox:] = False                  # outside the table: plain page colour
table = (pm | zone) & ~margin
daily = fill_region(daily, margin, 18, 18, ink=240)
arr = np.asarray(daily, float).copy()
ink_px = Image.fromarray(((arr.mean(2) < 185) & ~zone).astype("uint8") * 255).filter(ImageFilter.MaxFilter(3))
fillable = table & (np.asarray(ink_px) == 0)                       # never paint over handwriting that is still visible next to the stylus
arr[fillable] = np.array([255.0, 251.0, 248.0])                    # plain page colour, then the thin rules again
for ry in (953.3, 986.7, 1020.0, 1053.3, 1086.7, 1120.0, 1153.3, 1186.7):
    yy = int(round(ry)) - oy
    arr[yy, fillable[yy]] = (235, 232, 227)
daily = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
_k8 = 8
_lay = Image.new("RGBA", (daily.width * _k8, daily.height * _k8), (0, 0, 0, 0))
_ld = ImageDraw.Draw(_lay)
_lf = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", int(8.2 * _k8))
for i, t in enumerate(["7 AM", "8 AM", "9 AM", "10 AM", "11 AM"]):
    xx = (1029.0 - ox) * _k8
    for ch in t:
        _ld.text((xx, (945.8 + 33.33 * i - oy) * _k8), ch, font=_lf, fill=(150, 144, 138, 235), anchor="ls")
        xx += _ld.textlength(ch, font=_lf) + 1.0 * _k8
_lay = _lay.resize(daily.size, Image.LANCZOS)
daily = daily.convert("RGBA"); daily.alpha_composite(_lay); daily = daily.convert("RGB")

# the four app icons straight from the original (Apple Calendar, Reminders, Google Calendar, ChatGPT)
icon_acal = src.crop((778, 478, 870, 570))
icon_rem = src.crop((895, 478, 987, 570))
icon_gcal = src.crop((1013, 478, 1105, 570))
icon_gpt = src.crop((1131, 478, 1223, 570))

F = "/usr/share/fonts/truetype/"
serif = F + "liberation/LiberationSerif-Regular.ttf"
sans = F + "dejavu/DejaVuSans.ttf"

# --- background: warm taupe wall lit by one broad, soft beam of window light.
#     Shade on the left (darker toward the bottom), lighter shade on the right, a soft window-frame shadow in the top-right corner.
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)


def _smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


_xl = 600 + 0.045 * _yy                      # left edge of the beam (leans a little as the light falls)
_xr = 1830 + 0.035 * _yy                     # right edge
_beam = _smooth(_xl - 85, _xl + 130, _xx) * (1 - _smooth(_xr - 80, _xr + 80, _xx))
_beam *= 1 - 0.26 * (_yy / H) ** 1.4         # the light weakens toward the bottom
_bloom = np.exp(-(((_xx - 1180) / 760) ** 2 + ((_yy - 230) / 520) ** 2))     # a little warm glow up where the title sits
_lit = np.array([240, 219, 214], np.float32)
_shade_l = np.array([172, 144, 140], np.float32)
_shade_r = np.array([200, 174, 169], np.float32)
_sx = _smooth(900, 2400, _xx)[..., None]
_shade = _shade_l * (1 - _sx) + _shade_r * _sx
_shade = _shade * (1 - 0.20 * (_yy / H)[..., None] * (1 - _beam[..., None]))     # shade sinks toward the bottom corners
_bg = _shade + (_lit - _shade) * _beam[..., None]
_bg = _bg * (1 + 0.05 * _bloom[..., None] * _beam[..., None])
_corner = Image.new("L", (W, H), 0)
ImageDraw.Draw(_corner).polygon([(0, 0), (470, 0), (0, 430)], fill=255)
_corner = np.asarray(_corner.filter(ImageFilter.GaussianBlur(75)), np.float32) / 255
_bg = _bg * (1 - 0.34 * _corner[..., None])
# faint "god-ray" streaks inside the beam: broad, very low contrast, same lean as the beam edges
_rays = np.zeros((H, W), np.float32)
for _cx, _wd, _a in [(760, 110, 1.0), (1090, 70, 0.7), (1400, 140, 0.9), (1690, 60, 0.6)]:
    _c = _cx + 0.04 * _yy
    _rays += _a * np.exp(-((_xx - _c) / _wd) ** 2)
_rays *= _beam * (1 - 0.7 * (_yy / H))
_bg = _bg * (1 + 0.03 * _rays[..., None])
canvas = Image.fromarray(np.clip(_bg, 0, 255).astype("uint8")).convert("RGBA")


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
    scr = scr.filter(ImageFilter.UnsharpMask(radius=1.1, percent=55, threshold=2))
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


place(tablet(cover, 1130, 4), 690, 1085)       # cover on the left, behind
place(tablet(daily, 1300, -1.5), 1585, 1050)     # daily page on the right, in front

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
drop(pen.split()[3], 40, 690, [(36, 52, 22, .30), (14, 20, 7, .45), (4, 6, 3, .6)])
canvas.alpha_composite(pen, (40, 690))


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
ink = (116, 68, 64)
spaced(d, (W / 2, 44), "2026 - 2027   \u2022   +1000 STICKERS", ImageFont.truetype(serif, 40), ink, 9)
spaced(d, (W / 2, 94), "GLOW-UP PLANNER", ImageFont.truetype(serif, 108), ink, 16)
_sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
spaced(ImageDraw.Draw(_sub), (W / 2, 234), "SELF-CARE, ROUTINES AND DAILY GLOW WINS", ImageFont.truetype(sans, 24), (104, 56, 54, 255), 5)
_sub = _sub.filter(ImageFilter.GaussianBlur(5)); _sub.putalpha(_sub.split()[3].point(lambda v: int(v * .75)))
canvas.alpha_composite(_sub)
d = ImageDraw.Draw(canvas)
spaced(d, (W / 2, 232), "SELF-CARE, ROUTINES AND DAILY GLOW WINS", ImageFont.truetype(sans, 24), (252, 249, 245), 5)
# app icons (original artwork, cropped from the source mockup)
IC = 88
tiles = [i.resize((IC, IC), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 90, 0)) for i in (icon_acal, icon_rem, icon_gcal, icon_gpt)]
gap = 22
lab_font = ImageFont.truetype(sans, 19)
LEFT = 505                                         # left edge of the cover tablet
lab_w = sum(d.textlength(c, font=lab_font) for c in "WORKS WITH") + 5 * 9
x0 = int(W / 2 - 200 - (IC * 4 + gap * 3) / 2)       # icons left of centre
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
out = Image.composite(out, Image.new("RGB", (W, H), (150, 134, 122)), vg.point(lambda v: 205 + v * 50 // 255))
nz = Image.effect_noise((W, H), 14).convert("RGB")
out = Image.blend(out, nz, 0.025)
out.save(OUT)

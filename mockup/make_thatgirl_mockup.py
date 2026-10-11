"""Builds the THAT GIRL planner listing mockup (2400x1800): same style as the Teacher mockup (Reading's amber background, straight
iPads, strong shadows, big app icons), with the daily page on the left, the bigger cover iPad in front on the right, and the stylus
on the left of the page iPad.
Usage: python3 make_thatgirl_mockup.py <thatgirl_source.jpg> <thatgirl_cover.png> <out.png>
<thatgirl_source> is the original 2000x2000 That Girl mockup (daily page + app icons); <thatgirl_cover> is the clean cover page.
The stylus lying across the page is painted out, the schedule labels it hid (7-10 AM) and the first letter of "Morning walk" are
rebuilt, and the cover year is changed to the two-year range."""
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
src = Image.open(SRC).convert("RGB")             # the original 2000x2000 That Girl mockup (daily page + app icons)
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


# ---- cover: the clean cover page supplied by the author (no stylus, nothing hidden); only the year line changes
cv_raw = Image.open(COVER).convert("RGB")                          # 646 x 849 (a little taller than 3:4)
OXC = 5
cv_raw = cv_raw.crop((OXC, 0, 641, 849))                           # 636 x 849 = 3:4; only plain paper margin is trimmed
K2 = 2
cover = cv_raw.resize((cv_raw.width * K2, cv_raw.height * K2), Image.LANCZOS)


def cover_year(img, text="2026 - 2027"):
    """Replace the letter-spaced '2027' above the title with '2026 - 2027' (same colour, quiet serif); coordinates are in the 2x image."""
    m = np.zeros((img.height, img.width), bool)
    m[243 * K2:267 * K2, (288 - OXC) * K2:(356 - OXC) * K2] = True
    img = fill_region(img, m, 8 * K2, 8 * K2)                       # the cover paper is a smooth gradient: no grain to match
    k = 5
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", int(19.0 * K2 * k))
    tr = 3.2 * K2 * k
    widths = [ld.textlength(ch, font=f) for ch in text]
    x = (321.0 - OXC) * K2 * k - (sum(widths) + tr * (len(text) - 1)) / 2
    for ch, w in zip(text, widths):
        ld.text((x, 261.4 * K2 * k), ch, font=f, fill=(184, 150, 142, 255), anchor="ls")
        x += w + tr
    layer = layer.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")


cover = cover_year(cover)

# ---- daily page: take the stylus off; the schedule labels it hid (7-10 AM) and the "M" of "Morning walk" are rebuilt
daily = src.crop(DAILY_BOX)
ox, oy = DAILY_BOX[0], DAILY_BOX[1]
pm = pencil_mask(src.size, DAILY_BOX, hl=50, hr=22, ext=6)
pm = np.asarray(Image.fromarray(pm.astype('uint8') * 255).filter(ImageFilter.MaxFilter(7))) > 0     # include the soft edge of the stylus
pm[:930 - oy, :] = False                                           # nothing above the pencil's cap
pm[1340 - oy:, :] = False                                          # (below this the stylus and its shadow have left the page)
arr = np.asarray(daily, float).copy()
cols = np.arange(daily.width)[None, :]
margin = pm & (cols < 1027 - ox)                                    # left of the schedule: plain page colour
arr[margin] = np.array([255.0, 251.0, 248.0])
damage = np.zeros_like(pm)
damage[930 - oy:1056 - oy, 1027 - ox:1066 - ox] = True             # the time-label column, rows 7-10 AM (the handwriting starts at x=1067)
damage[1056 - oy:1068 - oy, 1027 - ox:1035 - ox] = True            # the last sliver of the stylus' shadow, just above the 11 AM label
damage[935 - oy:974 - oy, 1066 - ox:1074 - ox] = True              # the hidden left stroke of the "M" of "Morning walk" and the stylus' shadow edge below it (the next handwriting starts at y=975)
for y in np.where(damage.any(1))[0]:
    xs = np.where(damage[y])[0]
    arr[y, xs] = arr[y, 1250 - ox]                                  # plain page + thin row rule, copied from the clear part of the same row
# the capital "M" of "Morning walk": the same handwriting glyph from "Moved my body today" one row above (rows are 33.3 px apart)
patch = np.asarray(src.crop((1064, 903, 1076, 919)), float)
tx, ty = 1064 - ox, 903 + 33 - oy
tgt = arr[ty:ty + patch.shape[0], tx:tx + patch.shape[1]]
arr[ty:ty + patch.shape[0], tx:tx + patch.shape[1]] = np.minimum(tgt, patch)
daily = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
_k8 = 8
_lay = Image.new("RGBA", (daily.width * _k8, daily.height * _k8), (0, 0, 0, 0))
_ld = ImageDraw.Draw(_lay)
_lf = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Regular.otf", int(6.0 * _k8))   # best fit to the planner's own small time labels
for txt, rule in (("7 AM", 953.3), ("8 AM", 986.6), ("9 AM", 1019.9), ("10 AM", 1053.2)):
    xx = (1028.2 - ox) * _k8
    for ch in txt:
        _ld.text((xx, (rule - 8.5 - oy) * _k8), ch, font=_lf, fill=(158, 152, 148, 225), anchor="ls")
        xx += _ld.textlength(ch, font=_lf) + 0.6 * _k8
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
_lit = np.array([240, 216, 176], np.float32)
_shade_l = np.array([166, 132, 104], np.float32)
_shade_r = np.array([192, 160, 128], np.float32)
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
    drop(dev.split()[3], x, y, [(170, 240, 130, .38), (95, 140, 70, .42), (42, 64, 30, .48), (16, 26, 11, .58), (5, 8, 3, .65)])   # long cast shadow down-right + tight contact shadow
    canvas.alpha_composite(dev, (x, y))


place(tablet(daily, 1080, 0), 745, 1105)       # daily page on the left, behind (the smaller iPad)
place(tablet(cover, 1220, 0), 1600, 1085)     # cover on the right, in front and bigger: it overlaps the right edge of the page iPad

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
drop(pen.split()[3], 110, 690, [(36, 52, 22, .30), (14, 20, 7, .45), (4, 6, 3, .6)])
canvas.alpha_composite(pen, (110, 690))


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
ink = (104, 66, 38)
spaced(d, (W / 2, 44), "2026 - 2027   \u2022   +1000 STICKERS", ImageFont.truetype(serif, 40), ink, 9)
spaced(d, (W / 2, 94), "THAT GIRL PLANNER", ImageFont.truetype(serif, 108), ink, 16)
_sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
spaced(ImageDraw.Draw(_sub), (W / 2, 234), "ROUTINES \u00b7 SELF-CARE \u00b7 MIND \u00b7 HEALTH \u00b7 MONEY", ImageFont.truetype(sans, 24), (96, 58, 34, 255), 5)
_sub = _sub.filter(ImageFilter.GaussianBlur(5)); _sub.putalpha(_sub.split()[3].point(lambda v: int(v * .75)))
canvas.alpha_composite(_sub)
d = ImageDraw.Draw(canvas)
spaced(d, (W / 2, 232), "ROUTINES \u00b7 SELF-CARE \u00b7 MIND \u00b7 HEALTH \u00b7 MONEY", ImageFont.truetype(sans, 24), (252, 249, 245), 5)
# app icons (original artwork, cropped from the source mockup)
IC = 118
tiles = [i.resize((IC, IC), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 90, 0)) for i in (icon_acal, icon_rem, icon_gcal, icon_gpt)]
gap = 28
lab_font = ImageFont.truetype(sans, 19)
LEFT = 505                                         # left edge of the cover tablet
lab_w = sum(d.textlength(c, font=lab_font) for c in "WORKS WITH") + 5 * 9
x0 = 500                                              # icons in the band above the page iPad, as marked
mk_i = Image.new("L", (IC, IC), 0); ImageDraw.Draw(mk_i).rounded_rectangle((0, 0, IC - 1, IC - 1), radius=int(IC * 0.24), fill=255)
x = x0
for ic in tiles:
    tile = Image.new("RGBA", (IC, IC)); tile.paste(ic, (0, 0)); tile.putalpha(mk_i)
    drop(mk_i, x, 420, [(2, 5, 4, .35), (8, 16, 14, .25)])
    canvas.alpha_composite(tile, (x, 420))
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

"""Builds the MANIFESTATION JOURNAL listing mockup (2400x1800) in the ADHD-planner style: taupe wall with a beam of window light,
the cover iPad tilted on the left with a pencil on it, the daily page iPad tilted on the right, "+1000 STICKERS" badge top-left,
the four app icons, "dreamwy" at the bottom.
Usage: python3 make_manifestation_mockup.py <manifestation_source.webp> <budget_source.jpg> <out.png>
<manifestation_source> is the existing 1269x952 mockup: both screens are cut out of it and straightened (cover 1.2 deg, page 0.6 deg).
<budget_source> only supplies the four app icons (sharper artwork)."""
import sys
import math
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SRC, ICONS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 2400, 1800
src = Image.open(SRC).convert("RGB")
icons_src = Image.open(ICONS).convert("RGB")


def straighten(img, tl, w, h, deg, scale=2):
    """Cut a slightly rotated screen out of the source and return it upright (2x size)."""
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    tl = np.array(tl); u = np.array([c, s]); v = np.array([-s, c])
    quad = [tl, tl + u * w, tl + u * w + v * h, tl + v * h]
    W_, H_ = int(w * scale), int(h * scale)
    out = [(0, 0), (W_, 0), (W_, H_), (0, H_)]
    A, B = [], []
    for (x, y), (px, py) in zip(out, quad):
        A.append([x, y, 1, 0, 0, 0, -px * x, -px * y]); B.append(px)
        A.append([0, 0, 0, x, y, 1, -py * x, -py * y]); B.append(py)
    co = np.linalg.solve(np.array(A, float), np.array(B, float))
    return img.transform((W_, H_), Image.PERSPECTIVE, tuple(co), Image.BICUBIC)


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



# ---- cover: left iPad of the source, tilted 1.2 deg counter-clockwise; its right edge is clipped by the page iPad's bezel (a thin wedge, wider toward the bottom)
cover = straighten(src, (148.0, 278.6), 438, 583, -1.2)            # 876 x 1166
_arr = np.asarray(cover, float).copy()
for _y in range(_arr.shape[0]):                                     # the wedge: mirror the paper just left of the bezel over it
    _xb = int(880 - 0.0296 * _y) - 3
    _w = _arr.shape[1] - _xb
    if _w > 0:
        _arr[_y, _xb:] = _arr[_y, _xb - _w:_xb][::-1]
cover = Image.fromarray(np.clip(_arr, 0, 255).astype("uint8"))
_g = np.asarray(cover.crop((300, 900, 560, 940)).convert("L"), float)    # clean kraft-paper patch, for grain
GRAIN = _g - np.asarray(Image.fromarray(_g.astype("uint8")).filter(ImageFilter.GaussianBlur(3)), float)


def cover_year(img, text="2026 - 2027"):
    """Replace '2027' (and nothing else) with '2026 - 2027' in the same brown, quiet letter-spaced serif; coordinates in the 2x image."""
    m = np.zeros((img.height, img.width), bool); m[960:998, 392:482] = True
    img = fill_region(img, m, 8, 8, tex=GRAIN)
    k = 5
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", int(25 * k))
    tr = 5.2 * k
    widths = [ld.textlength(ch, font=f) for ch in text]
    x = 436.5 * k - (sum(widths) + tr * (len(text) - 1)) / 2
    for ch, w in zip(text, widths):
        ld.text((x, 988.6 * k), ch, font=f, fill=(126, 104, 76, 255), anchor="ls")
        x += w + tr
    layer = layer.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")


cover = cover_year(cover)
# ---- daily page: right iPad of the source, tilted 0.6 deg clockwise
daily = straighten(src, (602.5, 211.4), 491.2, 654.9, 0.61).crop((3, 3, 979, 1306))
# the four app icons, sharper artwork from the Budget original: Apple Calendar, Reminders, Google Calendar, ChatGPT
icon_acal = icons_src.crop((778, 478, 870, 570))
icon_rem = icons_src.crop((895, 478, 987, 570))
icon_gcal = icons_src.crop((1013, 478, 1105, 570))
icon_gpt = icons_src.crop((1131, 478, 1223, 570))

F = "/usr/share/fonts/truetype/"
serif = F + "liberation/LiberationSerif-Regular.ttf"
sans = F + "dejavu/DejaVuSans.ttf"

# --- background: warm taupe wall lit by one broad, soft beam of window light.
#     Shade on the left (darker toward the bottom), lighter shade on the right, a soft window-frame shadow in the top-right corner.
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)


def _smooth(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


_xl = 545 - 0.045 * _yy                      # left edge of the beam (leans a little as the light falls)
_xr = 1880 - 0.035 * _yy                     # right edge
_beam = _smooth(_xl - 85, _xl + 130, _xx) * (1 - _smooth(_xr - 80, _xr + 80, _xx))
_beam *= 1 - 0.26 * (_yy / H) ** 1.4         # the light weakens toward the bottom
_bloom = np.exp(-(((_xx - 1180) / 760) ** 2 + ((_yy - 230) / 520) ** 2))     # a little warm glow up where the title sits
_lit = np.array([226, 213, 201], np.float32)
_shade_l = np.array([156, 142, 132], np.float32)
_shade_r = np.array([184, 170, 159], np.float32)
_sx = _smooth(900, 2400, _xx)[..., None]
_shade = _shade_l * (1 - _sx) + _shade_r * _sx
_shade = _shade * (1 - 0.20 * (_yy / H)[..., None] * (1 - _beam[..., None]))     # shade sinks toward the bottom corners
_bg = _shade + (_lit - _shade) * _beam[..., None]
_bg = _bg * (1 + 0.05 * _bloom[..., None] * _beam[..., None])
_corner = Image.new("L", (W, H), 0)
ImageDraw.Draw(_corner).polygon([(1930, 0), (W, 0), (W, 430)], fill=255)
_corner = np.asarray(_corner.filter(ImageFilter.GaussianBlur(75)), np.float32) / 255
_bg = _bg * (1 - 0.34 * _corner[..., None])
# faint "god-ray" streaks inside the beam: broad, very low contrast, same lean as the beam edges
_rays = np.zeros((H, W), np.float32)
for _cx, _wd, _a in [(760, 110, 1.0), (1090, 70, 0.7), (1400, 140, 0.9), (1690, 60, 0.6)]:
    _c = _cx - 0.04 * _yy
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
    # the source mockup is only ~1x resolution: keep the paper natural and sharpen gently (an edge-aware crisp pass makes it look painted)
    scr = scr.filter(ImageFilter.UnsharpMask(radius=1.5, percent=70, threshold=2))
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


place(tablet(cover, 1130, 4), 735, 1075)
place(tablet(daily, 1320, -1.5), 1625, 1035)

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
drop(pen.split()[3], 95, 690, [(36, 52, 22, .30), (14, 20, 7, .45), (4, 6, 3, .6)])
canvas.alpha_composite(pen, (95, 690))


# ---------- stickers ("+1000 stickers")
import math
SS = 4


def sticker(draw_fn, size, cx, cy, angle=0, border=14):
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), size * SS)
    a = big.split()[3].filter(ImageFilter.MaxFilter(border * SS // 2 * 2 + 1))
    white = Image.new("RGBA", big.size, (255, 255, 255, 255)); white.putalpha(a)
    white.alpha_composite(big)
    st = white.resize((size, size), Image.LANCZOS).rotate(angle, expand=True, resample=Image.BICUBIC)
    global canvas
    x, y = int(cx - st.width / 2), int(cy - st.height / 2)
    shd = Image.new("RGBA", st.size, (50, 35, 25, 0)); shd.putalpha(st.split()[3].point(lambda v: int(v * .45)))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); lay.paste(shd, (x + 8, y + 14), shd)
    canvas = Image.alpha_composite(canvas, lay.filter(ImageFilter.GaussianBlur(9)))
    canvas.alpha_composite(st, (x, y))


def star(d, n, col=(201, 150, 90)):
    c = n / 2
    pts = [(c + (n * .46 if i % 2 == 0 else n * .2) * math.sin(math.pi * i / 5),
            c - (n * .46 if i % 2 == 0 else n * .2) * math.cos(math.pi * i / 5)) for i in range(10)]
    d.polygon(pts, fill=col)

def heart(d, n, col=(196, 120, 110)):
    r = n * .24
    d.ellipse((n * .1, n * .16, n * .1 + 2 * r, n * .16 + 2 * r), fill=col)
    d.ellipse((n * .9 - 2 * r, n * .16, n * .9, n * .16 + 2 * r), fill=col)
    d.polygon([(n * .12, n * .5), (n * .88, n * .5), (n * .5, n * .86)], fill=col)

def leaf(d, n, col=(139, 120, 84)):
    d.polygon([(n * .12, n * .88), (n * .2, n * .35), (n * .55, n * .1), (n * .88, n * .12), (n * .85, n * .5), (n * .6, n * .8)], fill=col)
    d.line([(n * .15, n * .85), (n * .75, n * .25)], fill=(236, 226, 205), width=int(n * .03))

def flower(d, n, col=(226, 178, 160)):
    c = n / 2
    for i in range(6):
        a = i * math.pi / 3
        px_, py_ = c + n * .22 * math.cos(a), c + n * .22 * math.sin(a)
        d.ellipse((px_ - n * .16, py_ - n * .16, px_ + n * .16, py_ + n * .16), fill=col)
    d.ellipse((c - n * .12, c - n * .12, c + n * .12, c + n * .12), fill=(214, 168, 84))

def sun(d, n, col=(214, 168, 84)):
    c = n / 2
    for i in range(12):
        a = i * math.pi / 6
        d.line([(c + n * .3 * math.cos(a), c + n * .3 * math.sin(a)), (c + n * .45 * math.cos(a), c + n * .45 * math.sin(a))], fill=col, width=int(n * .06))
    d.ellipse((c - n * .24, c - n * .24, c + n * .24, c + n * .24), fill=col)

def check(d, n, col=(139, 120, 84)):
    d.ellipse((n * .06, n * .06, n * .94, n * .94), fill=col)
    d.line([(n * .28, n * .52), (n * .44, n * .68), (n * .74, n * .34)], fill=(255, 250, 240), width=int(n * .09), joint="curve")

def cup(d, n, col=(150, 106, 84)):
    d.rounded_rectangle((n * .14, n * .3, n * .68, n * .8), radius=int(n * .12), fill=col)
    d.arc((n * .5, n * .36, n * .88, n * .68), -90, 90, fill=col, width=int(n * .07))
    d.rounded_rectangle((n * .1, n * .8, n * .72, n * .86), radius=int(n * .03), fill=(120, 84, 66))
    for xx in (.3, .44):
        d.arc((n * xx, n * .08, n * (xx + .1), n * .26), 90, 270, fill=(190, 170, 150), width=int(n * .03))

def moon(d, n, col=(120, 100, 88)):
    d.ellipse((n * .12, n * .1, n * .9, n * .88), fill=col)
    d.ellipse((n * .34, n * .02, n * 1.0, n * .7), fill=(0, 0, 0, 0))

def tape(d, n, col=(214, 196, 170)):
    d.rectangle((n * .04, n * .36, n * .96, n * .64), fill=col)
    for i in range(7):
        d.line([(n * (.1 + i * .13), n * .36), (n * (.02 + i * .13), n * .64)], fill=(236, 224, 204), width=int(n * .035))

def sparkle(d, n, col=(214, 168, 84)):
    c = n / 2
    d.polygon([(c, n * .04), (c + n * .1, c - n * .1), (n * .96, c), (c + n * .1, c + n * .1), (c, n * .96), (c - n * .1, c + n * .1), (n * .04, c), (c - n * .1, c - n * .1)], fill=col)


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
spaced(d, (W / 2, 94), "MANIFESTATION JOURNAL", ImageFont.truetype(serif, 100), ink, 14)
_sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
spaced(ImageDraw.Draw(_sub), (W / 2, 234), "VISION \u00b7 GOALS \u00b7 GRATITUDE \u00b7 GLOW", ImageFont.truetype(sans, 29), (74, 52, 42, 255), 8)
_sub = _sub.filter(ImageFilter.GaussianBlur(5)); _sub.putalpha(_sub.split()[3].point(lambda v: int(v * .75)))
canvas.alpha_composite(_sub)
d = ImageDraw.Draw(canvas)
spaced(d, (W / 2, 232), "VISION \u00b7 GOALS \u00b7 GRATITUDE \u00b7 GLOW", ImageFont.truetype(sans, 29), (252, 249, 245), 8)
# app icons (original artwork, cropped from the source mockup)
IC = 88
tiles = [i.resize((IC, IC), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 90, 0)) for i in (icon_acal, icon_rem, icon_gcal, icon_gpt)]
gap = 22
lab_font = ImageFont.truetype(sans, 19)
LEFT = 505                                         # left edge of the cover tablet
lab_w = sum(d.textlength(c, font=lab_font) for c in "WORKS WITH") + 5 * 9
x0 = 1527                                          # icons above the page iPad's top edge, as marked
mk_i = Image.new("L", (IC, IC), 0); ImageDraw.Draw(mk_i).rounded_rectangle((0, 0, IC - 1, IC - 1), radius=21, fill=255)
x = x0
for ic in tiles:
    tile = Image.new("RGBA", (IC, IC)); tile.paste(ic, (0, 0)); tile.putalpha(mk_i)
    drop(mk_i, x, 272, [(2, 4, 3, .35), (6, 12, 12, .25)])
    canvas.alpha_composite(tile, (x, 272))
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

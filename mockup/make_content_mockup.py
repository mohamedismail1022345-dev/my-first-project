"""Builds the CONTENT planner listing mockup (2400x1800), same style as make_college_mockup.py.
Usage: python3 make_content_mockup.py <content_source.jpg> <out.png>
<content_source> is the original 2000x2000 Content mockup: it supplies the cover, the daily page
and the four app icons. The stylus lying on it is painted out; the schedule's hidden time labels
(8, 9 and 10 AM), the clipped row rules and the top rule are re-drawn, so every entry stays visible."""
import sys
import math
import numpy as np
try:                                  # opencv-contrib (xphoto) gives patch-based inpainting; optional, falls back to a blur fill
    import cv2
    _ = cv2.xphoto.INPAINT_SHIFTMAP
except Exception:
    cv2 = None
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
W, H = 2400, 1800
src = Image.open(SRC).convert("RGB")
COVER_BOX = (175, 620, 912, 1736)    # visible part of the cover screen (the daily tablet's shadow starts at ~915)
COVER_FULL_W = 797                   # real cover width: centred on its text (x=572.5), left edge x=175
DAILY_BOX = (985, 642, 1825, 1758)   # daily page screen only
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


# paper grain, measured on a clean patch of the cover, re-used to texture everything we repaint there
_clean = np.asarray(src.crop((220, 650, 560, 900)).convert("L"), float)
_bl = np.asarray(Image.fromarray(_clean.astype("uint8")).filter(ImageFilter.GaussianBlur(3)), float)
GRAIN = _clean - _bl

# ---- cover: take the stylus off, extend the strip hidden behind the daily tablet, show the two-year range
cover = src.crop(COVER_BOX)
cmask = pencil_mask(src.size, COVER_BOX, hl=27, hr=21)
if cv2 is not None:
    # patch-based fill so the watercolour leaves under the stylus keep their texture instead of turning into a smear.
    # The centre text block is hidden from the patch search (so no letters get copied in) and pasted back afterwards.
    TXT = (170, 330, 640, 780)                                    # x0, y0, x1, y1 in cover-crop coordinates
    base = np.asarray(cover.convert("RGB")).copy()
    guard = np.zeros(base.shape[:2], np.uint8); guard[TXT[1]:TXT[3], TXT[0]:TXT[2]] = 255
    clean = cv2.inpaint(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), guard, 9, cv2.INPAINT_TELEA)
    m8 = cv2.dilate(cmask.astype("uint8") * 255, np.ones((3, 3), np.uint8))
    out = np.zeros_like(clean)
    cv2.xphoto.inpaint(clean, 255 - m8, out, cv2.xphoto.INPAINT_SHIFTMAP)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(float)
    # The patch fill has the right grain but its overall tone can drift (it prefers paper over leaf). So keep its fine
    # detail and take the broad tone from a straight bridge between the colours on either side of the stylus, row by row.
    Hh, Ww = m8.shape
    Lc = np.full((Hh, 3), np.nan); Rc = np.full((Hh, 3), np.nan); runs = {}
    for y in range(Hh):
        xs = np.flatnonzero(m8[y])
        if len(xs):
            a, b = xs.min(), xs.max(); runs[y] = (a, b)
            if a - 8 >= 0: Lc[y] = base[y, a - 8:a].mean(0)
            if b + 9 <= Ww: Rc[y] = base[y, b + 1:b + 9].mean(0)

    def _vsmooth(c, sg=5):
        ok = (~np.isnan(c[:, 0])).astype(float); v = np.nan_to_num(c)
        k = _k(sg)
        num = _conv(v * ok[:, None], k, 0); den = _conv(ok, k, 0)
        return np.where(den[:, None] > 1e-3, num / np.maximum(den, 1e-6)[:, None], np.nan), ok > 0
    Ls, Lok = _vsmooth(Lc); Rs, Rok = _vsmooth(Rc)
    interp = out.copy()
    for y, (a, b) in runs.items():
        xs = np.arange(a, b + 1); t = ((xs - a + 0.5) / (b - a + 1))[:, None]
        l = Ls[y] if not np.isnan(Ls[y, 0]) else None; r = Rs[y] if not np.isnan(Rs[y, 0]) else None
        if l is not None and r is not None: interp[y, a:b + 1] = l * (1 - t) + r * t
        elif l is not None: interp[y, a:b + 1] = l
        elif r is not None: interp[y, a:b + 1] = r
    hp = out - cv2.GaussianBlur(out.astype("float32"), (0, 0), 9)
    out = np.where(m8[..., None] > 0, interp + hp, out)
    soft = cv2.GaussianBlur(m8.astype("float32") / 255, (0, 0), 1.6)[..., None]
    comp = base.astype(float) * (1 - soft) + out * soft
    comp[TXT[1]:TXT[3], TXT[0]:TXT[2]] = base[TXT[1]:TXT[3], TXT[0]:TXT[2]]
    cover = Image.fromarray(np.clip(comp, 0, 255).astype("uint8"))
else:
    cover = fill_region(cover, cmask, 14, 14, tex=GRAIN)
# The right ~60 px of the real cover are hidden behind the daily tablet in the original. Instead of mirroring them
# (which duplicates the cut-off leaf), ease-stretch the last 200 px of artwork over 260 px: gentle at the left, widest at the edge.
BAND_SRC, BAND_OUT = 320, 320 + (COVER_FULL_W - cover.width)
x0 = cover.width - BAND_SRC
uu = (np.arange(BAND_OUT) + 0.5) / BAND_OUT
ss = BAND_SRC * (uu + 0.3 * uu * (1 - uu))                      # monotonic, ds/du: 1.0 -> 0.54
arr = np.asarray(cover, float)
band = np.empty((arr.shape[0], BAND_OUT, 3))
i0 = np.clip(np.floor(ss - 0.5).astype(int), 0, BAND_SRC - 1); i1 = np.clip(i0 + 1, 0, BAND_SRC - 1)
fr = (ss - 0.5 - i0)[None, :, None]
src_band = arr[:, x0:, :]
band = src_band[:, i0, :] * (1 - fr) + src_band[:, i1, :] * fr
cover_full = np.concatenate([arr[:, :x0, :], band], axis=1)
cover = Image.fromarray(np.clip(cover_full, 0, 255).astype("uint8"))
assert cover.width == COVER_FULL_W


def cover_year(img, text="2026 - 2027"):
    """Replace the single '2027' (and its two thin rules) on the cover with the two-year range."""
    ox, oy = COVER_BOX[0], COVER_BOX[1]
    cx = 572.5 - ox                                  # centre of the cover text column
    m = np.zeros((img.height, img.width), bool)
    m[1066 - oy:1170 - oy, int(cx) - 140:int(cx) + 141] = True
    img = fill_region(img, m, 10, 10, tex=GRAIN)
    k = 6
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for yy in (1079, 1155):                          # the two thin rules, widened to suit the longer text
        ld.rectangle(((cx - 118) * k, (yy - oy) * k, (cx + 118) * k, (yy - oy + 1.4) * k), fill=(204, 178, 150, 255))
    f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", int(25 * k))
    tr = 9 * k
    widths = [ld.textlength(c, font=f) for c in text]
    x = cx * k - (sum(widths) + tr * (len(text) - 1)) / 2
    for c, w in zip(text, widths):
        ld.text((x, (1131 - oy) * k), c, font=f, fill=(96, 76, 64, 255), anchor="ls")
        x += w + tr
    layer = layer.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")


cover = cover_year(cover)

# ---- daily page: take the stylus off, then re-typeset the schedule time labels it was hiding (5-9 AM)
daily = src.crop(DAILY_BOX)
dmask = pencil_mask(src.size, DAILY_BOX, hl=30, hr=24, ext=6)
dmask[:932 - DAILY_BOX[1], :] = False                                                            # keep the SCHEDULE heading intact
dmask[940 - DAILY_BOX[1]:1052 - DAILY_BOX[1], 1026 - DAILY_BOX[0]:1068 - DAILY_BOX[0]] = True   # label column, rows 8-10 AM
dmask[1052 - DAILY_BOX[1]:, 1030 - DAILY_BOX[0]:] = False                                         # leave the 11 AM label and below untouched
daily = fill_region(daily, dmask, 6, 0.9, ink=232)
_k8 = 8
_lay = Image.new("RGBA", (daily.width * _k8, daily.height * _k8), (0, 0, 0, 0))
_ld = ImageDraw.Draw(_lay)
_ld.rectangle(((1027 - DAILY_BOX[0]) * _k8, (936 - DAILY_BOX[1]) * _k8, (1082 - DAILY_BOX[0]) * _k8, (938 - DAILY_BOX[1]) * _k8 - 1),
              fill=(121, 116, 112, 255))                                                          # the schedule's top rule, 1px
for _ry in (972, 1005, 1038.6):                                                                   # the row rules the stylus covered
    _ld.rectangle(((1027 - DAILY_BOX[0]) * _k8, (_ry - DAILY_BOX[1]) * _k8, (1082 - DAILY_BOX[0]) * _k8, (_ry + 1 - DAILY_BOX[1]) * _k8 - 1),
                  fill=(228, 223, 217, 255))
_lf = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", int(8.2 * _k8))
for i, t in enumerate(["8 AM", "9 AM", "10 AM"]):
    xx = (1031.7 - DAILY_BOX[0]) * _k8
    for ch in t:
        _ld.text((xx, (964.9 + 33.35 * i - DAILY_BOX[1]) * _k8), ch, font=_lf, fill=(150, 144, 138, 235), anchor="ls")
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


place(tablet(cover, 1130, -4), 1690, 1085)     # cover on the right, behind
place(tablet(daily, 1300, 1.5), 815, 1050)       # daily page on the left, in front

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
drop(pen.split()[3], 2165, 690, [(36, 52, 22, .30), (14, 20, 7, .45), (4, 6, 3, .6)])
canvas.alpha_composite(pen, (2165, 690))


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
spaced(d, (W / 2, 94), "CONTENT PLANNER", ImageFont.truetype(serif, 108), ink, 16)
_sub = Image.new("RGBA", (W, H), (0, 0, 0, 0))
spaced(ImageDraw.Draw(_sub), (W / 2, 234), "A CALM SYSTEM FOR CREATORS", ImageFont.truetype(sans, 29), (74, 52, 42, 255), 8)
_sub = _sub.filter(ImageFilter.GaussianBlur(5)); _sub.putalpha(_sub.split()[3].point(lambda v: int(v * .75)))
canvas.alpha_composite(_sub)
d = ImageDraw.Draw(canvas)
spaced(d, (W / 2, 232), "A CALM SYSTEM FOR CREATORS", ImageFont.truetype(sans, 29), (252, 249, 245), 8)
# app icons (original artwork, cropped from the source mockup)
IC = 88
tiles = [i.resize((IC, IC), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 90, 0)) for i in (icon_acal, icon_rem, icon_gcal, icon_gpt)]
gap = 22
lab_font = ImageFont.truetype(sans, 19)
LEFT = 505                                         # left edge of the cover tablet
lab_w = sum(d.textlength(c, font=lab_font) for c in "WORKS WITH") + 5 * 9
x0 = 1830 - (IC * 4 + gap * 3)                   # icons sit above the cover tablet, on the right
mk_i = Image.new("L", (IC, IC), 0); ImageDraw.Draw(mk_i).rounded_rectangle((0, 0, IC - 1, IC - 1), radius=21, fill=255)
x = x0
for ic in tiles:
    tile = Image.new("RGBA", (IC, IC)); tile.paste(ic, (0, 0)); tile.putalpha(mk_i)
    drop(mk_i, x, 330, [(2, 4, 3, .35), (6, 12, 12, .25)])
    canvas.alpha_composite(tile, (x, 330))
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

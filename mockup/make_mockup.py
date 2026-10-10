"""Builds the listing mockup (2400x1800) from the existing planner page images.
Usage: python3 make_mockup.py <source_mockup.webp> <out.png>
The daily page and cover are cropped unchanged from the source mockup."""
import sys
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
W, H = 2400, 1800
src = Image.open(SRC).convert("RGB")
daily = src.crop((667, 398, 1143, 1070))   # screen only, bezel removed
cover = src.crop((111, 395, 598, 1080))


def cover_year(img, text="2026 - 2027"):
    """Replace the single '2027' on the cover with the two-year range, in the same quiet serif."""
    img = img.copy()
    cx, y0, y1 = 243, 276, 342                      # centre x and band that holds the divider lines + year
    pw = 150
    patch = img.crop((330, y0, 330 + pw, y1))        # clean paper from the same row, right of the text
    fm = Image.new("L", (pw, y1 - y0), 255).filter(ImageFilter.GaussianBlur(0))
    edge = Image.new("L", (pw, y1 - y0), 0)
    ImageDraw.Draw(edge).rectangle((6, 6, pw - 7, y1 - y0 - 7), fill=255)
    edge = edge.filter(ImageFilter.GaussianBlur(4))
    img.paste(patch, (cx - pw // 2, y0), edge)
    k = 6
    layer = Image.new("RGBA", (img.width * k, img.height * k), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    line_col = (196, 168, 138, 255)
    half = 40
    for yy in (286, 334):                             # divider lines, widened to suit the longer text
        ld.rectangle(((cx - half) * k, yy * k, (cx + half) * k, (yy + 1) * k), fill=line_col)
    f = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 19 * k)
    tr = 4.2 * k
    widths = [ld.textlength(c, font=f) for c in text]
    x = cx * k - (sum(widths) + tr * (len(text) - 1)) / 2
    for c, w in zip(text, widths):
        ld.text((x, 322 * k), c, font=f, fill=(78, 60, 50, 255), anchor="ls")
        x += w + tr
    layer = layer.resize(img.size, Image.LANCZOS)
    img = img.convert("RGBA"); img.alpha_composite(layer)
    return img.convert("RGB")


cover = cover_year(cover)
# the four app icons from the original mockup: Apple Calendar, Reminders, Google Calendar, ChatGPT
icon_acal = src.crop((485, 302, 547, 362))
icon_rem = src.crop((559, 302, 621, 362))
icon_gcal = src.crop((635, 302, 697, 362))
icon_gpt = src.crop((709, 302, 771, 362))

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
    # edge-aware sharpening: crisp text/lines, smooth paper (keeps JPEG noise from being amplified)
    flat = scr.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.5))
    crisp = scr.filter(ImageFilter.UnsharpMask(radius=1.8, percent=150, threshold=2))
    edge = scr.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
    edge = edge.point(lambda v: min(255, v * 5))
    scr = Image.composite(crisp, flat, edge)
    scr = ImageEnhance.Contrast(scr).enhance(1.08)
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


place(tablet(cover, 1130, 4), 770, 1075)
place(tablet(daily, 1320, -1.5), 1590, 1035)

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
pen = pen.rotate(-80, expand=True, resample=Image.BICUBIC)
drop(pen.split()[3], 380, 720, [(36, 52, 22, .30), (14, 20, 7, .45), (4, 6, 3, .6)])
canvas.alpha_composite(pen, (380, 720))


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
sx, sy, sr = 222, 585, 97
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
spaced(d, (W / 2, 94), "ADHD PLANNER", ImageFont.truetype(serif, 108), ink, 16)
spaced(d, (W / 2, 232), "PLANNER FOR ADHD & NEURODIVERGENT BRAINS", ImageFont.truetype(sans, 29), (252, 249, 245), 8)
# app icons (original artwork, cropped from the source mockup)
IC = 88
tiles = [i.resize((IC, IC), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1.2, 90, 0)) for i in (icon_acal, icon_rem, icon_gcal, icon_gpt)]
gap = 22
lab_font = ImageFont.truetype(sans, 19)
LEFT = 450                                         # left edge of the cover tablet
lab_w = sum(d.textlength(c, font=lab_font) for c in "WORKS WITH") + 5 * 9
x0 = int(LEFT + lab_w + 28)
mk_i = Image.new("L", (IC, IC), 0); ImageDraw.Draw(mk_i).rounded_rectangle((0, 0, IC - 1, IC - 1), radius=21, fill=255)
spaced(d, (LEFT, 306 + 22), "WORKS WITH", lab_font, ink, 5, anchor_center=False)
x = x0
for ic in tiles:
    tile = Image.new("RGBA", (IC, IC)); tile.paste(ic, (0, 0)); tile.putalpha(mk_i)
    drop(mk_i, x, 306, [(2, 4, 3, .35), (6, 12, 12, .25)])
    canvas.alpha_composite(tile, (x, 306))
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

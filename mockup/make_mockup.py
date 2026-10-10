"""Builds the listing mockup (2400x1800) from the existing planner page images.
Usage: python3 make_mockup.py <source_mockup.webp> <out.png>
The daily page and cover are cropped unchanged from the source mockup."""
import sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
W, H = 2400, 1800
src = Image.open(SRC).convert("RGB")
daily = src.crop((667, 398, 1143, 1070))   # screen only, bezel removed
cover = src.crop((111, 395, 598, 1080))
icons = [src.crop(b) for b in [(485, 302, 547, 362), (559, 302, 621, 362), (635, 302, 697, 362), (709, 302, 771, 362)]]

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
d.polygon([(0, 0), (520, 0), (180, H)], fill=60)
d.polygon([(1700, 0), (W, 0), (W, 500)], fill=45)
sh = sh.filter(ImageFilter.GaussianBlur(120))
bg = Image.composite(Image.new("RGB", (W, H), (120, 105, 95)), bg, sh)
canvas = bg.convert("RGBA")


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
    w, h = sw + 2 * pad, height
    m = Image.new("L", (sw, sh_), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, sw - 1, sh_ - 1), radius=int(pad * 1.1), fill=255)
    dev = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(dev).rounded_rectangle((0, 0, w - 1, h - 1), radius=int(pad * 2.2), fill=(14, 14, 14, 255))
    ImageDraw.Draw(dev).rounded_rectangle((3, 3, w - 4, h - 4), radius=int(pad * 2.2), outline=(95, 92, 90, 255), width=3)
    dev.paste(scr, (pad, pad), m)
    gl = Image.new("RGBA", (sw, sh_), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl)
    gd.polygon([(sw * .55, 0), (sw, 0), (sw, sh_ * .35), (sw * .1, sh_)], fill=(255, 255, 255, 7))
    gd.polygon([(0, sh_ * .6), (sw * .45, 0), (sw * .6, 0), (0, sh_ * .8)], fill=(255, 255, 255, 4))
    dev.paste(gl, (pad, pad), Image.composite(gl.split()[3], Image.new("L", gl.size, 0), m))
    ImageDraw.Draw(dev).ellipse((w / 2 - 5, pad / 2 - 5, w / 2 + 5, pad / 2 + 5), fill=(38, 38, 42, 255))
    return dev.rotate(angle, resample=Image.BICUBIC, expand=True)


def place(dev, cx, cy):
    global canvas
    x, y = int(cx - dev.width / 2), int(cy - dev.height / 2)
    a = dev.split()[3]
    shadow = Image.new("RGBA", dev.size, (40, 28, 20, 0))
    shadow.putalpha(a.point(lambda v: int(v * 0.55)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.paste(shadow, (x + 30, y + 50), shadow)
    layer = layer.filter(ImageFilter.GaussianBlur(38))
    canvas = Image.alpha_composite(canvas, layer)
    canvas.alpha_composite(dev, (x, y))


place(tablet(cover, 1060, 4), 790, 1105)
place(tablet(daily, 1130, -3), 1530, 1125)

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
sh_p = Image.new("RGBA", pen.size, (30, 20, 14, 0)); sh_p.putalpha(pen.split()[3].point(lambda v: int(v * .85)))
lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); lay.paste(sh_p, (418, 768), sh_p)
canvas = Image.alpha_composite(canvas, lay.filter(ImageFilter.GaussianBlur(9)))
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

sticker(star, 150, 120, 960, -12)
sticker(flower, 190, 250, 1180, 10)
sticker(heart, 150, 100, 1400, -8)
sticker(cup, 170, 270, 1520, 8)
sticker(tape, 230, 190, 880, -35, border=8)
sticker(sun, 170, 2250, 640, 8)
sticker(leaf, 190, 2200, 840, -10)
sticker(check, 140, 2290, 1000, 6)
sticker(moon, 150, 2150, 1180, 18)
sticker(sparkle, 120, 2290, 1330, 0)
sticker(flower, 150, 2190, 1480, -14)
sticker(heart, 120, 2300, 1590, 12)
sticker(star, 110, 140, 1620, 14)

# "+1000 STICKERS" seal
sx, sy, sr = 260, 650, 150
seal = Image.new("RGBA", (sr * 2 + 40, sr * 2 + 40), (0, 0, 0, 0))
sd = ImageDraw.Draw(seal)
sc = sr + 20
for i in range(28):
    a = i * 2 * math.pi / 28
    sd.ellipse((sc + (sr - 8) * math.cos(a) - 16, sc + (sr - 8) * math.sin(a) - 16, sc + (sr - 8) * math.cos(a) + 16, sc + (sr - 8) * math.sin(a) + 16), fill=(122, 88, 72, 255))
sd.ellipse((sc - sr + 8, sc - sr + 8, sc + sr - 8, sc + sr - 8), fill=(122, 88, 72, 255))
sd.ellipse((sc - sr + 24, sc - sr + 24, sc + sr - 24, sc + sr - 24), outline=(236, 220, 200, 255), width=3)
tf = ImageFont.truetype(serif, 80); sf = ImageFont.truetype(sans, 25)
cream = (250, 242, 232, 255)
sd.text((sc, sc - 28), "+1000", font=tf, fill=cream, anchor="mm")
spaced(sd, (sc, sc + 28), "STICKERS", sf, cream, 5)
sd.line([(sc - 70, sc + 18), (sc + 70, sc + 18)], fill=cream, width=2)
seal = seal.rotate(10, expand=True, resample=Image.BICUBIC)
shd = Image.new("RGBA", seal.size, (50, 35, 25, 0)); shd.putalpha(seal.split()[3].point(lambda v: int(v * .45)))
lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); lay.paste(shd, (sx - seal.width // 2 + 8, sy - seal.height // 2 + 14), shd)
canvas = Image.alpha_composite(canvas, lay.filter(ImageFilter.GaussianBlur(10)))
canvas.alpha_composite(seal, (sx - seal.width // 2, sy - seal.height // 2))

d = ImageDraw.Draw(canvas)
ink = (96, 70, 58)
spaced(d, (W / 2, 70), "2026 - 2027   \u2022   +1000 STICKERS", ImageFont.truetype(serif, 52), ink, 10)
spaced(d, (W / 2, 140), "ADHD PLANNER", ImageFont.truetype(serif, 138), ink, 20)
spaced(d, (W / 2, 318), "PLANNER FOR ADHD & NEURODIVERGENT BRAINS", ImageFont.truetype(sans, 36), (252, 249, 245), 9)
# app icons, top right
x = 1855
for ic in icons:
    ic = ic.resize((100, 100), Image.LANCZOS)
    m = Image.new("L", ic.size, 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, 99, 99), radius=23, fill=255)
    canvas.paste(ic, (x, 405), m)
    x += 120
spaced(d, (W / 2, 1722), "dreamwy", ImageFont.truetype(serif, 54), ink, 12)
import random
out = canvas.convert("RGB")
vg = Image.new("L", (W, H), 0)
ImageDraw.Draw(vg).ellipse((-W * .25, -H * .25, W * 1.25, H * 1.25), fill=255)
vg = vg.filter(ImageFilter.GaussianBlur(260))
out = Image.composite(out, Image.new("RGB", (W, H), (150, 134, 122)), vg.point(lambda v: 150 + v * 105 // 255))
nz = Image.effect_noise((W, H), 14).convert("RGB")
out = Image.blend(out, nz, 0.025)
out.save(OUT)

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
pen = Image.new("RGBA", (900, 56), (0, 0, 0, 0))
pd = ImageDraw.Draw(pen)
body = Image.new("RGBA", (860, 56), (0, 0, 0, 0))
bd = ImageDraw.Draw(body)
for yy in range(56):
    k = 1 - abs(yy - 22) / 34
    c = int(212 + 43 * max(k, 0))
    bd.line([(0, yy), (860, yy)], fill=(c, c, int(c - 3), 255))
mk = Image.new("L", (860, 56), 0)
ImageDraw.Draw(mk).rounded_rectangle((0, 0, 859, 55), radius=26, fill=255)
pen.paste(body, (0, 0), mk)
pd.polygon([(850, 8), (900, 28), (850, 48)], fill=(232, 228, 222, 255))
pd.polygon([(886, 24), (900, 28), (886, 32)], fill=(60, 56, 54, 255))
pd.rectangle((120, 10, 720, 13), fill=(255, 255, 255, 255))
pen = pen.rotate(-80, expand=True, resample=Image.BICUBIC)
sh_p = Image.new("RGBA", pen.size, (30, 20, 14, 0)); sh_p.putalpha(pen.split()[3].point(lambda v: int(v * .5)))
lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); lay.paste(sh_p, (400, 740), sh_p)
canvas = Image.alpha_composite(canvas, lay.filter(ImageFilter.GaussianBlur(14)))
canvas.alpha_composite(pen, (380, 720))

d = ImageDraw.Draw(canvas)
ink = (96, 70, 58)
spaced(d, (W / 2, 70), "2026 - 2027", ImageFont.truetype(serif, 52), ink, 10)
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

---
tags: [dreamwy, etsy, mockup, process, python]
created: 2026-10-11
status: reference
---

# Dreamwy planner mockups: how they are made

Listing mockups for the Dreamwy iPad planners, built with Python scripts instead of a design app. One script per planner, all in the same style.

- **Where:** GitHub `mohamedismail1022345-dev/my-first-project`, branch `claude/affectionate-feynman-v8qsw4`, folder `mockup/`
- **Output:** `<name>_mockup.png` (2400x1800) plus `<name>_mockup_794x596.png/.webp` (same size as the old Etsy reference)

## 1. What goes in

| Input | What it is |
|---|---|
| `<name>_source.jpg` | The original 2000x2000 mockup (two iPads, app icons, a stylus lying across them). Gives the **daily page** and the **4 app icons** |
| `<name>_cover.png/.webp` | The **clean cover** (no stylus, nothing hidden). Sent separately when the original cover was damaged |
| `budget_source.jpg` | Also reused for sharper icons in the 1024px Business/Freelance originals |

Rule from the author: **the planner pages stay as they are.** Only repaint what the stylus covered, and say what was guessed.

## 2. The shared look (canvas 2400x1800)

1. **Background:** a warm wall lit by one broad beam of window light (numpy: smoothstep beam, shade left/right, soft shadow in one top corner, faint rays). Palette per planner (see table).
2. **Header text (Liberation Serif):**
   - top line `2026 - 2027   •   +1000 STICKERS` (size 40)
   - title `<NAME> PLANNER` (size 108, tracking 16)
   - subtitle, white DejaVu Sans 24 with a soft dark shadow under it. Taken from the cover tagline; if the cover has none, made up from the page tabs.
3. **Badge:** brown scalloped seal "+1000 STICKERS" at top-left (centre 545,300, radius 97).
4. **App icons:** Apple Calendar, Reminders, Google Calendar, ChatGPT, cropped from the original at 92px: `(778,478,870,570)`, `(895,...)`, `(1013,...)`, `(1131,...)`. Never drop them. Size 118, gap 28 in the newest scripts.
5. **iPads:** drawn in code (rounded black bezel, rim light top-left, camera dot, faint glare). Newest ones are **untilted**. Layered shadows `(dx,dy,blur,opacity)`: `(170,240,130,.38) (95,140,70,.42) (42,64,30,.48) (16,26,11,.58) (5,8,3,.65)`.
6. **Layout (Student/Teacher/That Girl):** page iPad height 1080 at (745,1105) behind; **cover iPad height 1220 at (1600,1085) in front and bigger**, overlapping the page's right edge.
7. **Stylus:** drawn white pencil with shading and its own shadow, lying beside an iPad (side differs per planner).
8. **Footer:** `dreamwy` in Liberation Serif at the bottom centre. Vignette + light film grain at the end.

Amber palette (the master, from the Reading planner): lit `(240,216,176)`, shade L `(166,132,104)`, shade R `(192,160,128)`, ink `(104,66,38)`.

## 3. How a build runs

1. **Crop the screens** from the original. Daily page = box `(985,642,1825,1758)`; the cover screen is partly hidden by the page iPad in the original, which is why a clean cover is better.
2. **Straighten** tilted screens with a perspective transform (Business 11 deg, Freelance 3 deg, Girl Boss b 4 deg, Running Girl 0.35 deg).
3. **Remove the stylus** from the page. The stylus line is `(707,1775.5) -> (1056.7,943.3)` in all 2000px originals.
   - plain margin: paint the page colour
   - thin rules and shading: copy the same row from a clear column
   - never paint over visible handwriting
4. **Rebuild what it hid:**
   - small time labels: fit font size/tracking against the clear labels on the same page, then re-type (DejaVu 7.4 for Teacher, Inter 6.0 for That Girl, Liberation Sans for early ones)
   - handwriting letters: re-type in **Caveat** (the planner's own font, OFL licence, `Caveat-Variable.ttf` in the repo), or copy the same glyph from another row (the "M" of "Morning walk", the "I" of "Invoice")
5. **Cover year:** the single `2027` becomes `2026 - 2027`. Fill the old year from nearby paper (+ measured grain on textured covers), redraw in the same colour/typeface, widen the thin rules if the cover has them.
6. **Compose** background, iPads, shadows, stylus, badge, icons, text. Render, then **zoom in on every repaired spot** before sending.
7. **Export** 2400x1800 PNG plus 794x596 (LANCZOS + unsharp radius 0.8, 60%).

## 4. Scripts per planner

Run from `mockup/`. Signature = what the script needs.

| Planner | Script | Inputs | Look |
|---|---|---|---|
| ADHD | `make_mockup.py` | `adhd_source.webp out.png` | taupe, cover left (tilted), pencil on cover |
| Budget | `make_budget_mockup.py` | `budget_source.jpg out.png` | taupe, page left / cover right, pencil right |
| Small Business | `make_business_mockup.py` | `business_source.jpg budget_source.jpg out.png` | 1024px source, straightened; 30% of the review page is hidden in the original |
| College | `make_college_mockup.py` | `college_source.jpg out.png` | cover left, hidden cover edge stretched |
| Content | `make_content_mockup.py` | `content_source.jpg content_cover.webp out.png` | clean cover; 8/9/10 AM labels re-typed |
| Freelance | `make_freelance_mockup.py` | `freelance_source.jpg budget_source.jpg out.png` | 3 deg straighten; bezel overlap on the page margin repainted |
| Girl Boss | `make_girlboss_mockup.py` | `a|b source_a source_b cover out.png` | two variants (two different daily pages) |
| Glow-Up | `make_glowup_mockup.py` | `glowup_source.jpg glowup_cover.png out.png` | blush palette, cover left |
| Language | `make_language_mockup.py` | `language_source.jpg out.png` | sage palette, **no clean cover**: stylus inpainted with OpenCV, hidden strip stretched; 3 table cells unknown |
| Reading | `make_reading_mockup.py` | `reading_source.jpg reading_cover.webp out.png` | amber (master palette), "Wh" re-typed in Caveat |
| Running Girl | `make_runninggirl_mockup.py` | `runninggirl_source.jpg runninggirl_cover.png out.png` | lavender, cover left, page straightened |
| Self-Care | `make_selfcare_mockup.py` | `selfcare_source.jpg selfcare_cover.webp out.png` | amber, "Sle" re-typed |
| Student | `make_student_mockup.py` | `student_source.jpg student_cover.webp out.png` | amber, **big cover in front**, strong shadows, big icons |
| Teacher | `make_teacher_mockup.py` | `teacher_source.jpg teacher_cover.webp out.png` | same as Student, untilted; period labels re-typed |
| That Girl | `make_thatgirl_mockup.py` | `thatgirl_source.jpg thatgirl_cover.png out.png` | same, stylus on the left, icons above the page iPad |

Example: `python3 make_teacher_mockup.py teacher_source.jpg teacher_cover.webp teacher_mockup.png`

## 5. Tools and gotchas

- Python 3.13 + **Pillow 12** + **numpy**. OpenCV is optional: `opencv-contrib-python-headless` (needs `cv2.xphoto` for patch inpainting). It is not preinstalled: `pip download` the wheel, then `pip install --no-deps --target <dir>` and use `PYTHONPATH`. Scripts fall back to a blur fill without it.
- Fonts: Liberation Serif/Sans, DejaVu Sans, Inter (system), **Caveat** (bundled). The planner's real label font looks like Josefin Sans, which is not installed, so labels are a best-fit match.
- Each build takes 20-50 s. Slow part = the numpy lighting + normalised-convolution fills.
- **Pasted images only count if they arrive as files.** If a cover only appears inline in the chat, ask for it again as a separate message.
- Check zoomed crops of: the repaired margin, re-typed labels, the cover year, and the page edge under the front iPad.
- The cover iPad in front hides about 10% of the page's right edge (month tabs). Tell the author.

## 6. Known limits

- 1024px originals (Business, Freelance, Girl Boss b) are soft at 2400px. Ask for a bigger export of the page.
- Language: the first three "new words" cells were fully hidden, left blank. Cover leaf cluster is rebuilt, not real.
- Student: one cell and one column heading in the lessons table were hidden, left blank.
- Subtitles for covers without a tagline (Glow-Up, That Girl) are invented from the planner sections. The author should confirm.
- The "+1000 STICKERS" badge and title line are assumed for every planner.

## 7. New planner checklist

1. Get the 2000px original and the clean cover (as files). Copy to `<name>_source.jpg` / `<name>_cover.*`.
2. Copy the closest script (`make_teacher_mockup.py` for the latest layout), set the name, subtitle and palette.
3. Find what the stylus hid: label rows, first letters, rules. Measure positions on the original.
4. Set the cover-year box and colour (measure the old year's bbox and darkest pixel).
5. Render, zoom, fix, export both sizes, commit and push.
6. Send the PNG and list what was repainted or guessed.

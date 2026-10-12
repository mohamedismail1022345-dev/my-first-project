# Meal Planner hero mockup

Etsy listing thumbnail (2000×2000) for the Meal Planner, in the same beige editorial
style as the Budget and ADHD planner mockups.

- `meal-planner-hero.html` — the mockup source (everything is HTML/CSS, no external requests)
- `meal-planner-hero.png` — rendered output
- `assets/` — app icons and the OFL-licensed web fonts the page uses
- `render.mjs` — renders any of the HTML files to PNG with Playwright

## Manifestation Planner hero (landscape 4:3)

Same layout as the ADHD planner mockup: kicker line, big title, subtitle, app icons,
sticker badge, cover iPad + daily page iPad, pencil resting on the cover, wordmark.

- `manifestation-planner-hero.html` — source (designed at 1269×952, rendered at 2× → 2538×1904)
- `manifestation-planner-hero.png` — beige backdrop (ADHD-planner look)
- `manifestation-planner-hero-rose.png` — dusty-rose backdrop (`?theme=rose`)

The "2026 - 2027 • +1000 STICKERS" line and badge are copied from the existing listing
images — remove them from the HTML if this product does not ship the stickers.

## Re-render

```sh
npm i playwright-core            # once, anywhere
node render.mjs <path-to>/node_modules/playwright-core meal-planner-hero.html meal-planner-hero.png
node render.mjs <path-to>/node_modules/playwright-core manifestation-planner-hero.html manifestation-planner-hero.png "" 2
node render.mjs <path-to>/node_modules/playwright-core manifestation-planner-hero.html manifestation-planner-hero-rose.png "theme=rose" 2
```

Arguments: `<playwright-core dir> <html> <out.png> [query] [scale]`.
`render.mjs` expects Chromium at `/opt/pw-browsers/chromium`; edit `executablePath` if yours differs.

## Reuse for another planner

Edit the text in the HTML. For the meal planner the page content comes from the arrays at the
bottom of the file; for the manifestation planner see the `SCHED`, `ACT` and row helpers in the
script at the bottom.

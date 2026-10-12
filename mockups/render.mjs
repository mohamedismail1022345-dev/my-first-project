// Renders a mockup HTML file to PNG with Playwright.
// usage: node render.mjs <playwright-core dir> <file.html> <out.png> [query] [scale]
//   e.g. node render.mjs ./node_modules/playwright-core meal-planner-hero.html meal.png
//        node render.mjs ./node_modules/playwright-core manifestation-planner-hero.html out.png theme=rose 2
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const [pwDir, htmlFile, out, query = '', scale = '1'] = process.argv.slice(2);
const pw = await import(pathToFileURL(path.join(pwDir, 'index.js')).href);
const chromium = pw.chromium ?? pw.default.chromium;
const url = pathToFileURL(path.resolve(htmlFile)).href + (query ? '?' + query : '');
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const probe = await browser.newPage({ viewport: { width: 4000, height: 4000 } });
await probe.goto(url);
const { w, h } = await probe.evaluate(() => {
  const r = document.querySelector('.canvas').getBoundingClientRect();
  return { w: Math.round(r.width), h: Math.round(r.height) };
});
await probe.close();
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: Number(scale) });
page.on('console', m => console.log('console:', m.text()));
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto(url);
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(400);
await page.screenshot({ path: out, clip: { x: 0, y: 0, width: w, height: h } });
await browser.close();
console.log('wrote', out, `${w * Number(scale)}x${h * Number(scale)}`);

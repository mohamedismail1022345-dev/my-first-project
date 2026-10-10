// Renders meal-planner-hero.html to a 2000x2000 PNG.
// usage: node render.mjs <playwright-core dir> [out.png]
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const pw = await import(pathToFileURL(path.join(process.argv[2], 'index.js')).href);
const chromium = pw.chromium ?? pw.default.chromium;
const out = process.argv[3] ?? 'meal-planner-hero.png';
const html = pathToFileURL(path.resolve('meal-planner-hero.html')).href;
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--force-device-scale-factor=1'] });
const page = await browser.newPage({ viewport: { width: 2000, height: 2000 }, deviceScaleFactor: 1 });
page.on('console', m => console.log('console:', m.text()));
page.on('pageerror', e => console.log('pageerror:', e.message));
await page.goto(html);
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(400);
await page.screenshot({ path: out, clip: { x: 0, y: 0, width: 2000, height: 2000 } });
await browser.close();
console.log('wrote', out);

// Draws the OCR Arena Space's thumbnail (space/thumbnail.png, 1200×630) in the homepage style:
// real pages around the headline, and the engines it compares.   node scripts/make-arena-thumb.mjs
import { chromium } from "playwright-core";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const space = path.join(root, "space");
// the logo, font and sample pages come from runemic.com (runemic/website keeps the only copy)
const A = "https://runemic.com/assets/";
const get = async (p) => { const r = await fetch(A + p); if (!r.ok) throw new Error(`${A + p}: HTTP ${r.status}`); return Buffer.from(await r.arrayBuffer()); };
const logo = (await get("brand/logo.svg")).toString("utf8");
const data = {};
for (const f of ["shahnameh.jpg", "japanese.png", "arabic-1897.jpg", "grimm.jpg"]) data[f] = (await get("samples/" + f)).toString("base64");
const img = (f) => `data:image/${f.endsWith(".png") ? "png" : "jpeg"};base64,` + data[f];
const geist = `@font-face{font-family:"Geist";src:url("data:font/woff2;base64,${(await get("fonts/geist-latin-wght-normal.woff2")).toString("base64")}") format("woff2");font-weight:100 900}`;
const cards = [["shahnameh.jpg", "فارسی", "left:52px;top:34px;width:204px;--r:-5deg"], ["japanese.png", "日本語", "left:-6px;top:338px;width:214px;--r:4deg"],
  ["arabic-1897.jpg", "العربية", "right:46px;top:26px;width:208px;--r:5deg"], ["grimm.jpg", "Fraktur", "right:-4px;top:336px;width:218px;--r:-4deg"]]
  .map(([f, t, s]) => `<div class="card" style="${s}"><img src="${img(f)}"><span class="tag">${t}</span></div>`).join("");
const chips = [["PaddleOCR", "gpu"], ["EasyOCR", "gpu"], ["Tesseract", "cpu"], ["Runemic", "api"], ["PDF text", "pdf"]]
  .map(([n, k]) => `<span class="chip chip--${k}">${n}</span>`).join("");
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || "/opt/pw-browsers/chromium" });
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>${geist}
  *{box-sizing:border-box}html,body{margin:0}body{width:1200px;height:630px;position:relative;overflow:hidden;background:#fff;color:#0a0a0a;font-family:"Geist","Noto Sans CJK JP",sans-serif}
  body::before{content:"";position:absolute;inset:0;background-image:radial-gradient(#e5e5e5 1.3px,transparent 1.6px);background-size:22px 22px;-webkit-mask-image:radial-gradient(ellipse 62% 60% at 50% 50%,#000 25%,transparent 78%)}
  .card{position:absolute;transform:rotate(var(--r));border-radius:16px;overflow:hidden;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.06),0 24px 50px -22px rgba(0,0,0,.38)}
  .card img{display:block;width:100%;aspect-ratio:3/4;object-fit:cover;object-position:top}
  .tag{position:absolute;left:10px;bottom:10px;padding:4px 11px;border-radius:999px;background:rgba(10,10,10,.8);color:#fff;font-size:15px;font-weight:500}
  .copy{position:absolute;left:290px;right:290px;top:0;bottom:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:26px}
  .brand{display:flex;align-items:center;gap:10px;font-size:24px;font-weight:600}.brand svg{width:38px;height:38px}
  h1{margin:0;font-size:84px;line-height:1;font-weight:650;letter-spacing:-.05em}
  p{margin:0;font-size:24px;color:#525252;letter-spacing:-.01em}
  .chips{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;max-width:680px}
  .chip{height:34px;display:inline-flex;align-items:center;padding:0 14px;border-radius:999px;font-size:17px;font-weight:600;border:1px solid}
  .chip--cpu{background:#e7f6ec;color:#166534;border-color:#c4ead0}.chip--gpu{background:#e8effd;color:#1e40af;border-color:#c9d8f8}
  .chip--api{background:#e6f0f1;color:#0f5257;border-color:#c8dfe1}.chip--pdf{background:#f5f5f5;color:#404040;border-color:#e5e5e5}
  </style></head><body>${cards}<div class="copy"><span class="brand">${logo}<span>Runemic</span></span><h1>OCR Arena</h1>
  <p>Compare OCR engines on images and PDFs, in your browser</p><div class="chips">${chips}</div></div></body></html>`);
await page.evaluate(() => document.fonts.ready);
await page.waitForFunction(() => [...document.images].every((i) => i.complete));
await page.screenshot({ path: path.join(root, "space/thumbnail.png") });
await browser.close();
console.log("space/thumbnail.png");

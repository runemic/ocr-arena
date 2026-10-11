// Browser test for the OCR Arena Space (space/, a static Space: the engines run in the page).
// Opens the served Space in Chromium and reads the example pages with PaddleOCR, EasyOCR and Tesseract (every script).
// Fails if an engine errors or returns nothing, or if PaddleOCR misses a word it should find.
//   SITE=http://127.0.0.1:8813 CHROMIUM=/path/to/chrome node tests/ocr_arena_browser.mjs
// MIRROR=<dir> serves the CDN and Hugging Face files from local copies (for machines without that network).
import { chromium } from "playwright-core";
import { readFileSync, existsSync } from "node:fs";

const SITE = process.env.SITE || "http://127.0.0.1:8813";
const MIRROR = process.env.MIRROR;
const CASES = (process.env.CASES || "letter.png:de:Kirchenbücher,japanese.png:ja:清掃,chinese.png:zh:档案,persian.png:fa:کتابخانه,arabic.png:ar:مكتبة,hindi.png:hi:पुस्तकालय,russian.png:ru:музей,korean.png:ko:도서관")
  .split(",").map((c) => c.split(":"));
const ENGINES = (process.env.ENGINES || "paddleocr,easyocr,tesseract").split(",");

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || "/opt/pw-browsers/chromium" });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
if (MIRROR) {
  const NM = MIRROR + "/node_modules/", M = MIRROR + "/models/";
  await ctx.route("https://cdn.jsdelivr.net/npm/**", (route) => {
    const u = new URL(route.request().url()).pathname.replace(/^\/npm\//, ""), m = u.match(/^((?:@[^/]+\/)?[^@/]+)(?:@[^/]+)?\/(.*)$/);
    let f = NM + m[1] + "/" + m[2];
    if (!existsSync(f) && existsSync(f + ".gz")) f += ".gz";
    if (!existsSync(f)) return route.fulfill({ status: 404, body: "" });
    route.fulfill({ status: 200, body: readFileSync(f), headers: { "content-type": /\.m?js$/.test(f) ? "text/javascript" : f.endsWith(".wasm") ? "application/wasm" : "application/octet-stream", "access-control-allow-origin": "*" } });
  });
  await ctx.route("https://huggingface.co/**", (route) => {
    const u = route.request().url(), name = u.includes("/easyocr-onnx/") ? "easyocr/" + u.split("/").pop()
      : u.includes("/PaddlePaddle/") ? u.split("/PaddlePaddle/")[1].split("/")[0] + "." + u.split(".").pop() : "none";
    if (!existsSync(M + name)) return route.fulfill({ status: 404, body: "" });
    route.fulfill({ status: 200, body: readFileSync(M + name), headers: { "access-control-allow-origin": "*" } });
  });
}
const page = await ctx.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(SITE + "/", { waitUntil: "networkidle" });

// pick exactly these engines
async function pick(ids) { // one click at a time: each click re-renders the cards
  for (let i = 0; i < 20; i++) {
    const done = await page.$$eval(".card input", (inputs, ids) => {
      const wrong = inputs.find((i) => !i.disabled && i.checked !== ids.includes(i.closest(".card").querySelector(".card__name").textContent.toLowerCase().replace(/\s.*$/, "").replace("pdf", "pdf-text")));
      if (wrong) wrong.click();
      return !wrong;
    }, ids);
    if (done) return;
  }
}

const failed = [], score = {};
async function readCase(file, lang, expect, engines, timeout, fatal) {
  const idx = await page.$$eval(".ex img", (imgs, f) => imgs.findIndex((i) => i.getAttribute("src").endsWith(f)), file);
  await page.click(`.ex >> nth=${idx}`);
  await page.waitForTimeout(800);
  await page.selectOption("#lang", lang);
  await pick(engines);
  await page.click("#run");
  await page.waitForFunction((n) => document.querySelectorAll(".col .page__text, .col .page__err").length >= n, engines.length, { timeout }).catch(() => {});
  const cols = await page.$$eval(".col", (cs) => cs.map((c) => ({ name: c.querySelector(".col__name").textContent, meta: c.querySelector(".col__meta").textContent,
    text: (c.querySelector(".page__text") || {}).textContent || "", err: (c.querySelector(".page__err") || {}).textContent || "" })));
  for (const c of cols) {
    const hit = c.text.includes(expect);
    console.log(`${c.name.padEnd(10)} ${file.padEnd(13)} ${lang}  ${c.meta.padEnd(8)} ${c.err ? "ERROR " + c.err : hit ? "found" : "MISSED"}  ${JSON.stringify(c.text.slice(0, 60))}`);
    score[c.name] = (score[c.name] || []).concat(`${lang}:${c.err ? "error" : hit ? "found" : "missed"} ${c.meta}`);
    if (!fatal) { if (c.err || !c.text.trim()) console.log(`::warning title=${c.name}::${file}: ${c.err || "no text"}`); continue; }
    if (c.err || !c.text.trim()) failed.push(`${c.name} ${file}: ${c.err || "no text"}`);
    else if (!hit && c.name === "PaddleOCR") failed.push(`${c.name} ${file}: didn't find "${expect}"`);
  }
}
for (const [file, lang, expect] of CASES) await readCase(file, lang, expect, ENGINES, 300000, true);

// the WebGPU-only vision-language models, run on the CPU (?cpu): slow, so one page each, and not fatal yet
const VLM_CASES = (process.env.VLM_CASES ?? "letter.png:de:Kirchenbücher:lightonocr,korean.png:ko:도서관:glm-ocr").split(",").filter(Boolean).map((c) => c.split(":"));
if (VLM_CASES.length) { // on the CPU (?cpu): a software GPU is far slower still
  await page.goto(SITE + "/?cpu", { waitUntil: "networkidle" });
  for (const [file, lang, expect, engine] of VLM_CASES) await readCase(file, lang, expect, [engine], 1000000, false);
}
if (process.env.GITHUB_ACTIONS) for (const [n, r] of Object.entries(score)) console.log(`::notice title=${n}::${r.join(", ")}`);
await browser.close();
if (errors.length) failed.push(...errors.map((e) => "page error: " + e));
if (failed.length) { console.log("\nFAILED:\n  " + failed.join("\n  ")); process.exit(1); }
console.log("\nall pages read");

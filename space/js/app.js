// OCR Arena: the page. Engines (engines.js) read pages in the browser; text is always set with textContent.
import { ENGINES, HAS_WEBGPU, ALLOW_CPU, GPU_SERVER } from "./engines.js";

const PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/";
const MAX = 3, MAX_PAGES = 20;
const LANGUAGES = [["en", "English", "English"], ["de", "German", "Deutsch"], ["fr", "French", "Français"], ["es", "Spanish", "Español"],
  ["fa", "Persian", "فارسی"], ["ar", "Arabic", "العربية"], ["ur", "Urdu", "اردو"], ["hi", "Hindi", "हिन्दी"], ["ru", "Russian", "Русский"],
  ["zh", "Chinese", "中文"], ["ja", "Japanese", "日本語"], ["ko", "Korean", "한국어"]];
const EXAMPLES = [["persian.png", "فارسی", "fa"], ["arabic.png", "العربية", "ar"], ["urdu.png", "اردو", "ur"], ["hindi.png", "हिन्दी", "hi"],
  ["japanese.png", "日本語", "ja"], ["korean.png", "한국어", "ko"], ["chinese.png", "中文", "zh"], ["russian.png", "Русский", "ru"],
  ["letter.png", "Deutsch", "de"], ["fraktur.png", "Fraktur", "de"], ["invoice-1900.jpg", "Invoice", "en"], ["table.png", "Table", "en"]];
const WHERE = { cpu: ["Your CPU", "cpu"], gpu: ["Your GPU · WebGPU", "gpu"], webgpu: ["Your GPU only · WebGPU", "gpu"], server: ["GPU server · soon", "server"], api: ["API", "api"], instant: ["Instant", "pdf"] };

const $ = (id) => document.getElementById(id);
let file = null, pdf = null, pageTotal = 0, running = false, pdfjs = null;
const picked = new Set(["paddleocr", "tesseract"]);

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 2600); }
function badge(where) { const [label, kind] = WHERE[where]; return el("span", "badge badge--" + kind, label); }
const lang = () => $("lang").value;
const isPdf = () => !!pdf;

function blocked(e) {
  if (e.where === "server" && !GPU_SERVER) return "Coming soon: needs a GPU server.";
  if (e.where === "webgpu" && !HAS_WEBGPU && !ALLOW_CPU) return "Needs WebGPU: use Chrome or Edge on a computer.";
  if (e.pdfOnly && !isPdf()) return "Reads PDFs only.";
  if (e.languages && !e.languages.includes(lang())) return "Doesn't read this language.";
  return "";
}

function renderCards() {
  const box = $("cards"), soon = $("cards-soon"); box.textContent = ""; soon.textContent = "";
  let nSoon = 0;
  for (const e of ENGINES) {
    const later = e.where === "server" && !GPU_SERVER; nSoon += later;
    const why = blocked(e);
    if (why) picked.delete(e.id);
    const card = el("label", "card" + (picked.has(e.id) ? " is-on" : "") + (why ? " is-off" : ""));
    const input = el("input"); input.type = "checkbox"; input.checked = picked.has(e.id); input.disabled = !!why; input.setAttribute("aria-label", e.name);
    input.addEventListener("change", () => {
      if (input.checked && picked.size >= MAX) { input.checked = false; toast(`Pick up to ${MAX} engines at a time.`); return; }
      input.checked ? picked.add(e.id) : picked.delete(e.id);
      renderCards();
    });
    const head = el("span", "card__head"); head.append(el("span", "card__name", e.name), el("span", "card__tick"));
    const badges = el("span", "badges"); badges.append(badge(e.where));
    if (e.version) badges.append(el("span", "badge badge--size", e.version));
    if (e.pdfOnly) badges.append(el("span", "badge badge--size", "PDF"));
    card.append(input, head, el("span", "card__by", e.by), badges, el("span", "card__note", e.note));
    if (why) card.append(el("span", "card__why", why));
    (later ? soon : box).append(card);
  }
  $("soon").hidden = !nSoon; $("soon-count").textContent = nSoon ? `· ${nSoon} models` : "";
  $("pick-count").textContent = picked.size ? `${picked.size} of ${MAX} picked` : `pick up to ${MAX}`;
  refresh();
}

function refresh() {
  $("run").disabled = !(file && picked.size && !running);
  $("run-hint").textContent = running ? "Reading…" : !file ? "Choose a page first." : !picked.size ? "Pick at least one engine." :
    "The first run of an engine downloads its model once; after that it's cached.";
}

function renderTable() {
  const tb = document.querySelector("#engine-table tbody"); tb.textContent = "";
  for (const e of ENGINES) {
    const tr = el("tr"), a = el("a", "", e.name); a.href = e.link; a.target = "_blank"; a.rel = "noopener";
    const t1 = el("td"); t1.append(a); const t3 = el("td"); t3.append(badge(e.where));
    tr.append(t1, el("td", "", e.by), t3, el("td", "", e.size || "–"), el("td", "", e.license));
    tb.append(tr);
  }
}

// ── the page
async function pdfLib() {
  if (!pdfjs) { pdfjs = await import(PDFJS + "pdf.min.mjs"); pdfjs.GlobalWorkerOptions.workerSrc = PDFJS + "pdf.worker.min.mjs"; }
  return pdfjs;
}

async function useFile(f, langCode) {
  if (!f) return;
  if (!/^image\/(png|jpeg|webp|gif|bmp)$/.test(f.type) && f.type !== "application/pdf") { toast("Use an image (PNG, JPEG, WebP) or a PDF."); return; }
  if (f.size > 50 * 1048576) { toast("That file is larger than 50 MB."); return; }
  file = f; pdf = null; pageTotal = 0;
  if (langCode) $("lang").value = langCode;
  $("drop-empty").hidden = true; $("drop-full").hidden = false; $("file-name").textContent = f.name;
  $("preview").hidden = f.type === "application/pdf"; $("pdf-preview").hidden = f.type !== "application/pdf";
  if (f.type === "application/pdf") {
    try {
      const lib = await pdfLib();
      pdf = await lib.getDocument({ data: await f.arrayBuffer() }).promise;
      pageTotal = pdf.numPages;
      const c = await renderPdfPage(1, 1.1); const pc = $("pdf-preview"); pc.width = c.width; pc.height = c.height; pc.getContext("2d").drawImage(c, 0, 0);
      $("page-total").textContent = `of ${pageTotal}` + (pageTotal > MAX_PAGES ? ` · up to ${MAX_PAGES} at a time` : "");
      $("pages").placeholder = pageTotal > 1 ? `1-${Math.min(pageTotal, 3)}` : "1";
    } catch (err) { toast("Couldn't open that PDF: " + err.message); file = null; }
  } else {
    $("preview").src = URL.createObjectURL(f);
  }
  $("pdfbar").hidden = !pdf;
  renderCards();
}

async function renderPdfPage(n, scale) {
  const page = await pdf.getPage(n), vp = page.getViewport({ scale });
  const c = document.createElement("canvas"); c.width = Math.round(vp.width); c.height = Math.round(vp.height);
  const ctx = c.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
  await page.render({ canvasContext: ctx, viewport: vp }).promise;
  return c;
}

async function pageCanvas(n) {
  if (pdf) { // about 1800 px on the long side, whatever the paper size
    const vp = (await pdf.getPage(n)).getViewport({ scale: 1 });
    return renderPdfPage(n, Math.min(3, 1800 / Math.max(vp.width, vp.height)));
  }
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, 2400 / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas"); c.width = Math.round(bmp.width * scale); c.height = Math.round(bmp.height * scale);
  const ctx = c.getContext("2d"); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height); ctx.drawImage(bmp, 0, 0, c.width, c.height);
  return c;
}

function pagesToRead() {
  if (!pdf) return [0];
  const spec = $("pages").value.trim() || (pageTotal > 1 ? `1-${Math.min(pageTotal, 3)}` : "1"), out = [];
  for (const part of spec.split(",").map((s) => s.trim()).filter(Boolean)) {
    const m = part.match(/^(\d*)\s*-\s*(\d*)$/);
    if (m) { const a = +(m[1] || 1), b = +(m[2] || pageTotal); for (let p = a; p <= b; p++) out.push(p); }
    else if (/^\d+$/.test(part)) out.push(+part);
  }
  return [...new Set(out)].filter((p) => p >= 1 && p <= pageTotal).slice(0, MAX_PAGES);
}

// ── reading: one column per engine; pages are rendered once and shared; engines run side by side
async function run() {
  const pages = pagesToRead();
  if (!pages.length) { toast("Those pages aren't in this PDF."); return; }
  running = true; refresh();
  const ids = ENGINES.filter((e) => picked.has(e.id)), cols = $("cols");
  cols.textContent = ""; cols.style.setProperty("--n", ids.length);
  $("results").hidden = false; $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  const canvases = {};
  const getPage = (n) => (canvases[n] ||= pageCanvas(n || 1).then((canvas) => ({ canvas, pdf: pdf ? { doc: pdf, index: n - 1 } : null })));
  await Promise.all(ids.map((e) => runEngine(e, pages, getPage, cols)));
  running = false; refresh();
}

async function runEngine(e, pages, getPage, cols) {
  const col = el("article", "col"), head = el("div", "col__head"), meta = el("span", "col__meta", "starting"), body = el("div", "col__body");
  head.append(el("span", "col__name", e.name), badge(e.where), meta);
  const foot = el("div", "col__foot"), copy = el("button", "btn btn--sm", "Copy"), save = el("button", "btn btn--sm", "Download .md");
  copy.type = save.type = "button"; copy.disabled = save.disabled = true;
  foot.append(copy, save); col.append(head, body, foot); cols.append(col);
  const texts = []; let total = 0;
  for (const n of pages) {
    const box = el("section", "page");
    if (pdf) box.append(el("span", "page__label", `Page ${n}`));
    const wait = el("div", "page__wait"), label = el("span", "", "Reading…"); wait.append(el("span", "spin"), label);
    box.append(wait); body.append(box);
    const t0 = performance.now(), tick = setInterval(() => { meta.textContent = ((performance.now() - t0) / 1000).toFixed(0) + " s"; }, 500);
    try {
      const page = await getPage(n);
      const limit = e.limit || (e.where === "api" ? 200 : 300); // seconds; also covers a model that never finishes downloading
      const text = await Promise.race([e.read(page, lang(), (p, what = "Reading") => { label.textContent = `${what}… ${Math.round(p * 100)}%`; }),
        new Promise((_, no) => setTimeout(() => no(new Error(`No answer after ${limit / 60} minutes. Check your connection and try again.`)), limit * 1000))]);
      total += performance.now() - t0;
      const pre = el("pre", "page__text", text || "(No text found.)"); pre.dir = "auto";
      wait.replaceWith(pre); texts.push((pdf ? `<!-- page ${n} -->\n` : "") + text);
    } catch (err) {
      wait.replaceWith(el("div", "page__err", (err && err.message) || String(err)));
    } finally { clearInterval(tick); }
  }
  meta.textContent = (total / 1000).toFixed(1) + " s" + (pages.length > 1 ? ` · ${pages.length} pages` : "") +
    (e.lastModel ? ` · ${e.lastModel}` : "") + (e.id === "runemic" && e.left != null ? ` · ${e.left} free today` : "");
  if (texts.length) {
    const all = texts.join("\n\n");
    copy.disabled = save.disabled = false;
    copy.onclick = () => navigator.clipboard.writeText(all).then(() => toast("Copied"));
    save.onclick = () => { const a = el("a"); a.href = URL.createObjectURL(new Blob([all], { type: "text/markdown" })); a.download = `${file.name.replace(/\.[^.]+$/, "")}-${e.id}.md`; a.click(); };
  }
}

// ── start
const sel = $("lang");
for (const [code, name, native] of LANGUAGES) { const o = el("option", "", name === native ? name : `${name} · ${native}`); o.value = code; sel.append(o); }
sel.addEventListener("change", renderCards);
// the example pages, logo and fonts are runemic.com's (runemic/website keeps the only copy)
const SAMPLES = "https://runemic.com/assets/samples/";
for (const [name, label, code] of EXAMPLES) {
  const b = el("button", "ex"); b.type = "button";
  const img = el("img"); img.src = SAMPLES + name; img.alt = ""; img.loading = "lazy";
  b.append(img, el("span", "", label));
  b.addEventListener("click", async () => {
    const blob = await (await fetch(SAMPLES + name)).blob();
    useFile(new File([blob], name, { type: blob.type || (name.endsWith(".jpg") ? "image/jpeg" : "image/png") }), code);
  });
  $("examples").append(b);
}
$("gpu-note").textContent = HAS_WEBGPU ? "Your browser has WebGPU, so PaddleOCR and EasyOCR run on your GPU, and so can GLM-OCR and LightOnOCR." :
  "Your browser has no WebGPU, so PaddleOCR and EasyOCR run on your CPU, and GLM-OCR and LightOnOCR can't run here (try Chrome or Edge on a computer).";
renderCards(); renderTable();
$("file").addEventListener("change", () => useFile($("file").files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (ev) => { ev.preventDefault(); drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("is-over")));
drop.addEventListener("drop", (ev) => { ev.preventDefault(); useFile(ev.dataTransfer.files[0]); });
document.addEventListener("paste", (ev) => { const f = [...(ev.clipboardData?.files || [])][0]; if (f) useFile(f); });
$("run").addEventListener("click", run);
refresh();

// The OCR engines of OCR Arena. Everything except Runemic runs in the visitor's browser: nothing is uploaded.
// Each engine: id, name, by, link, license, where ("cpu" | "gpu" | "webgpu" | "server" | "api" | "instant"), languages (null = most),
// note, pdfOnly, limit (seconds a page), and read(page, lang, progress) -> text. A page is {canvas, pdf?: {doc, index}}.
// where "webgpu": needs the visitor's GPU (WebGPU). where "server": needs GPU_SERVER, a ZeroGPU Space (not there yet).
import { PaddleOCR } from "./paddle.js";
import { EasyOCR } from "./easyocr.js";
import { VLM } from "./vlm.js";

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const TESSERACT = "https://cdn.jsdelivr.net/npm/tesseract.js@6.0.1/dist/tesseract.esm.min.js";
const HF = "https://huggingface.co/PaddlePaddle/";
const HF_EASY = "https://huggingface.co/runemic-ai/easyocr-onnx/resolve/main/";
const RUNEMIC_TRY = "https://api.runemic.com/v1/try";
// navigator.gpu can exist without a usable GPU (Chrome on many Linux machines), so ask for an adapter
const ADAPTER = typeof navigator !== "undefined" && navigator.gpu ? await navigator.gpu.requestAdapter().catch(() => null) : null;
export const HAS_WEBGPU = !!ADAPTER;
const HAS_F16 = !!ADAPTER && ADAPTER.features.has("shader-f16"); // half-precision maths on the GPU (most, not all)
// ?cpu runs the WebGPU-only models on the CPU instead (very slow; for testing)
export const ALLOW_CPU = typeof location !== "undefined" && new URLSearchParams(location.search).has("cpu");
// The ZeroGPU Space that will run the large models (needs Hugging Face PRO). Empty: those engines show as coming soon.
// Contract: a Gradio Space with an endpoint /read(image, model, lang) -> text.
export const GPU_SERVER = "";
const GRADIO = "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.1/dist/browser.js";
const ALL = ["en", "de", "fr", "es", "fa", "ar", "ur", "hi", "ru", "zh", "ja", "ko"];
const RTL = new Set(["fa", "ar", "ur"]);

function imageData(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { data: d.data, width: d.width, height: d.height };
}

// ── PaddleOCR: PP-OCRv6 small for Chinese, Japanese, English and Latin scripts; PP-OCRv5 per script otherwise
const PADDLE_REC = { multi: "PP-OCRv6_small_rec_onnx", ar: "arabic_PP-OCRv5_mobile_rec_onnx", hi: "devanagari_PP-OCRv5_mobile_rec_onnx",
  ru: "cyrillic_PP-OCRv5_mobile_rec_onnx", ko: "korean_PP-OCRv5_mobile_rec_onnx" };
export const PADDLE_MODELS = ["PP-OCRv6_small_det_onnx", ...Object.values(PADDLE_REC)].map((m) => "PaddlePaddle/" + m);
let ortModule = null;
async function ort() {
  if (!ortModule) {
    ortModule = await import(ORT + (HAS_WEBGPU ? "ort.webgpu.min.mjs" : "ort.min.mjs"));
    ortModule.env.wasm.wasmPaths = ORT;
    ortModule.env.wasm.numThreads = 1; // threads need cross-origin isolation, which a Space can't rely on
  }
  return ortModule;
}
const paddles = {};
async function paddleFor(lang) {
  const group = RTL.has(lang) ? "ar" : ["hi", "ru", "ko"].includes(lang) ? lang : "multi";
  if (!paddles[group]) {
    paddles[group] = (async () => {
      const [o, yml] = await Promise.all([ort(), fetch(HF + PADDLE_REC[group] + "/resolve/main/inference.yml").then((r) => r.text())]);
      const { dictFromYaml } = await import("./paddle.js");
      const p = new PaddleOCR({ ort: o, det: HF + "PP-OCRv6_small_det_onnx/resolve/main/inference.onnx",
        rec: HF + PADDLE_REC[group] + "/resolve/main/inference.onnx", dict: dictFromYaml(yml), providers: HAS_WEBGPU ? ["webgpu", "wasm"] : ["wasm"] });
      await p.load();
      return p;
    })();
    paddles[group].catch(() => delete paddles[group]);
  }
  return paddles[group];
}

// ── EasyOCR: Jaided AI's CRAFT detector and one CRNN recognizer per script (runemic-ai/easyocr-onnx)
const EASY_GROUP = { en: "english", de: "latin", fr: "latin", es: "latin", fa: "arabic", ar: "arabic", ur: "arabic",
  hi: "devanagari", ru: "cyrillic", zh: "zh_sim", ja: "japanese", ko: "korean" };
const easies = {};
async function easyFor(lang) {
  const group = EASY_GROUP[lang];
  if (!easies[group]) {
    easies[group] = (async () => {
      const [o, meta] = await Promise.all([ort(), fetch(HF_EASY + "rec_" + group + ".json").then((r) => {
        if (!r.ok) throw new Error(`EasyOCR model list answered ${r.status}`);
        return r.json();
      })]);
      const e = new EasyOCR({ ort: o, det: HF_EASY + "craft.onnx", rec: HF_EASY + "rec_" + group + ".onnx", characters: meta.characters,
        providers: HAS_WEBGPU ? ["webgpu", "wasm"] : ["wasm"] });
      await e.load();
      return e;
    })();
    easies[group].catch(() => delete easies[group]);
  }
  return easies[group];
}

// ── Vision-language OCR models on the visitor's GPU (Transformers.js)
const vlms = {
  "glm-ocr": new VLM({ repo: "onnx-community/GLM-OCR-ONNX", prompt: "Text Recognition:", gpu: HAS_WEBGPU, f16: HAS_F16 }),
  lightonocr: new VLM({ repo: "onnx-community/LightOnOCR-2-1B-ONNX", imagesFirst: true, maxPixels: 1.4e6, gpu: HAS_WEBGPU, f16: HAS_F16 }),
};

// ── the large models, on a GPU server (GPU_SERVER)
let gradio = null;
async function serverRead(model, page, lang) {
  if (!GPU_SERVER) throw new Error("Coming soon: this model needs a GPU server.");
  if (!gradio) gradio = import(GRADIO).then((m) => m.Client.connect(GPU_SERVER));
  const blob = await new Promise((ok) => page.canvas.toBlob(ok, "image/png"));
  const r = await (await gradio).predict("/read", { image: blob, model, lang });
  return String(r.data[0] || "").trim();
}
const server = (id, name, by, repo, license, note, languages = null) => ({
  id, name, version: "", by, link: "https://huggingface.co/" + repo, repo, license, where: "server", languages, size: "GPU server",
  note, limit: 240, read: (page, lang) => serverRead(id, page, lang),
});

// ── Tesseract.js: one worker per language setting, kept
const TESS_CODES = { en: "eng", de: "deu", fr: "fra", es: "spa", fa: "fas", ar: "ara", ur: "urd", hi: "hin", ru: "rus", zh: "chi_sim+chi_tra", ja: "jpn", ko: "kor" };
const workers = {};
let tessProgress = null;
async function tesseractFor(lang) {
  const codes = TESS_CODES[lang] + (lang === "en" ? "" : "+eng");
  if (!workers[codes]) {
    workers[codes] = (async () => {
      const mod = await import(TESSERACT), T = mod.createWorker ? mod : mod.default;
      return T.createWorker(codes.split("+"), 1, { logger: (m) => tessProgress && m.status === "recognizing text" && tessProgress(m.progress) });
    })();
    workers[codes].catch(() => delete workers[codes]);
  }
  return workers[codes];
}

export const ENGINES = [
  {
    id: "paddleocr", name: "PaddleOCR", version: "PP-OCRv6 / v5", by: "Baidu PaddlePaddle", link: "https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx",
    license: "Apache-2.0", where: HAS_WEBGPU ? "gpu" : "cpu", languages: ALL, size: "15–30 MB",
    note: "Baidu's PP-OCR text detection and recognition, run with ONNX Runtime in your browser. Fast and very strong on Chinese and Japanese.",
    async read(page, lang, progress) {
      const p = await paddleFor(lang);
      return p.read(imageData(page.canvas), { rtl: RTL.has(lang), onProgress: progress });
    },
  },
  {
    id: "easyocr", name: "EasyOCR", version: "1.7 (CRAFT + CRNN)", by: "Jaided AI", link: "https://github.com/JaidedAI/EasyOCR",
    license: "Apache-2.0", where: HAS_WEBGPU ? "gpu" : "cpu", languages: ALL, size: "95–300 MB (Arabic-script and Hindi are the large ones)",
    note: "The popular PyTorch OCR library: CRAFT finds the text, a CRNN reads each line. Its official models, converted to ONNX, run in your browser.",
    async read(page, lang, progress) {
      const e = await easyFor(lang);
      return e.read(imageData(page.canvas), { rtl: RTL.has(lang), onProgress: progress });
    },
  },
  {
    id: "glm-ocr", name: "GLM-OCR", version: "0.9B", by: "Zhipu AI (Z.ai)", link: "https://huggingface.co/zai-org/GLM-OCR",
    license: "MIT", where: "webgpu", languages: ["en", "de", "fr", "es", "ru", "zh", "ja", "ko"], size: "about 780 MB", limit: 900,
    note: "A small vision-language model that tops the OmniDocBench document benchmark. Reads the whole page into Markdown, tables and formulas included.",
    read: (page, lang, progress) => vlms["glm-ocr"].read(page.canvas, { onProgress: progress }),
  },
  {
    id: "lightonocr", name: "LightOnOCR", version: "2 · 1B", by: "LightOn", link: "https://huggingface.co/lightonai/LightOnOCR-2-1B",
    license: "Apache-2.0", where: "webgpu", languages: ["en", "de", "fr", "es", "zh", "ja"], size: "about 900 MB", limit: 900,
    note: "An end-to-end OCR vision-language model: page in, Markdown out, with tables as HTML. Strongest on European languages.",
    read: (page, lang, progress) => vlms.lightonocr.read(page.canvas, { onProgress: progress }),
  },
  {
    id: "tesseract", name: "Tesseract", version: "5 (Tesseract.js)", by: "open source, originally Google", link: "https://github.com/naptha/tesseract.js",
    license: "Apache-2.0", where: "cpu", languages: ALL, size: "5–20 MB a language",
    note: "The classic open-source OCR engine, compiled to WebAssembly. Dependable on clean printed pages in 100+ languages.",
    async read(page, lang, progress) {
      const w = await tesseractFor(lang);
      tessProgress = progress;
      const r = await w.recognize(page.canvas);
      tessProgress = null;
      let text = (r.data.text || "").replace(/\n{3,}/g, "\n\n").trim();
      if (lang === "zh" || lang === "ja") // Tesseract puts spaces between CJK characters; Chinese and Japanese don't use them
        text = text.replace(/(?<=[\u3000-\u30ff\u3400-\u9fff\uff00-\uffef])[ \t]+(?=[\u3000-\u30ff\u3400-\u9fff\uff00-\uffef0-9])|(?<=[0-9])[ \t]+(?=[\u3000-\u30ff\u3400-\u9fff])/g, "");
      return text;
    },
  },
  {
    id: "runemic", name: "Runemic", version: "preview models", by: "Runemic Labs", link: "https://runemic.com",
    license: "Hosted API", where: "api", languages: null, size: "",
    note: "Runemic's hosted OCR: vision models that read whole pages into Markdown, strong on Arabic-script, Indic and CJK. The page is sent to api.runemic.com; free to try.",
    async read(page) {
      const blob = await new Promise((ok) => page.canvas.toBlob(ok, "image/jpeg", 0.9));
      const form = new FormData();
      form.append("file", blob, "page.jpg");
      form.append("format", "markdown");
      const r = await fetch(RUNEMIC_TRY, { method: "POST", body: form, credentials: "include" });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error((body.error && body.error.message) || `Runemic answered ${r.status}`);
      this.lastModel = body.model;
      this.left = body.remaining;
      return (body.text || "").trim();
    },
  },
  {
    id: "pdf-text", name: "PDF text layer", version: "", by: "pdf.js (Mozilla)", link: "https://mozilla.github.io/pdf.js/",
    license: "Apache-2.0", where: "instant", languages: null, pdfOnly: true, size: "",
    note: "Not OCR: the text a digital PDF already contains. Instant, and it shows whether a PDF needs OCR at all.",
    async read(page) {
      if (!page.pdf) throw new Error("Reads PDFs only.");
      const p = await page.pdf.doc.getPage(page.pdf.index + 1);
      const c = await p.getTextContent();
      const text = c.items.map((i) => i.str + (i.hasEOL ? "\n" : "")).join("").replace(/\n{3,}/g, "\n\n").trim();
      return text || "(This page has no text layer: it's a scan or a picture, so it needs OCR.)";
    },
  },
  server("unlimited-ocr", "Unlimited-OCR", "Baidu", "baidu/Unlimited-OCR", "MIT",
    "Baidu's long-document parser: reads many pages in one pass, with layout, tables and formulas."),
  server("deepseek-ocr-2", "DeepSeek-OCR 2", "DeepSeek", "deepseek-ai/DeepSeek-OCR-2", "Apache-2.0",
    "Compresses a page into few vision tokens and decodes it to Markdown; fast and accurate on dense documents."),
  server("paddleocr-vl", "PaddleOCR-VL", "Baidu PaddlePaddle", "PaddlePaddle/PaddleOCR-VL-1.6", "Apache-2.0",
    "PaddleOCR's vision-language parser: layout, tables, formulas, charts and seals in 100+ languages."),
  server("dots-ocr", "dots.ocr", "rednote hi lab", "dots-studio/dots.ocr", "MIT",
    "Layout and text in a single vision-language model; strong on multilingual documents."),
  server("chandra-ocr", "Chandra OCR 2", "Datalab", "datalab-to/chandra-ocr-2", "OpenRAIL",
    "Datalab's OCR model (the makers of Surya and Marker): Markdown, HTML or JSON out, with layout kept."),
];

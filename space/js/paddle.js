// PaddleOCR in the browser: PP-OCR text detection (DB) and line recognition (CTC) with ONNX Runtime, following the
// models' own configs (inference.yml): detection resizes the long side to 960, normalises BGR with ImageNet
// mean/std, and keeps regions above 0.3 whose mean score is over 0.6, grown by an unclip ratio of 1.5;
// recognition resizes each line to height 48 and normalises to [-1, 1].
// Works on plain {data: RGBA bytes, width, height} images, so it also runs in Node for testing.

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

// ── image helpers (RGBA in, RGBA out)
export function resize(img, w, h) {
  const out = new Uint8ClampedArray(w * h * 4), sx = img.width / w, sy = img.height / h;
  for (let y = 0; y < h; y++) {
    const fy = Math.max(0, (y + 0.5) * sy - 0.5), y0 = Math.floor(fy), y1 = Math.min(img.height - 1, y0 + 1), dy = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.max(0, (x + 0.5) * sx - 0.5), x0 = Math.floor(fx), x1 = Math.min(img.width - 1, x0 + 1), dx = fx - x0;
      for (let c = 0; c < 3; c++) {
        const a = img.data[(y0 * img.width + x0) * 4 + c], b = img.data[(y0 * img.width + x1) * 4 + c];
        const d = img.data[(y1 * img.width + x0) * 4 + c], e = img.data[(y1 * img.width + x1) * 4 + c];
        out[(y * w + x) * 4 + c] = (a * (1 - dx) + b * dx) * (1 - dy) + (d * (1 - dx) + e * dx) * dy;
      }
      out[(y * w + x) * 4 + 3] = 255;
    }
  }
  return { data: out, width: w, height: h };
}

export function crop(img, x0, y0, x1, y1) {
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(img.width, Math.ceil(x1)); y1 = Math.min(img.height, Math.ceil(y1));
  const w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0), out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) out.set(img.data.subarray(((y0 + y) * img.width + x0) * 4, ((y0 + y) * img.width + x0 + w) * 4), y * w * 4);
  return { data: out, width: w, height: h };
}

// RGBA -> CHW float tensor data in BGR channel order
function toTensor(img, norm) {
  const n = img.width * img.height, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const r = img.data[i * 4], g = img.data[i * 4 + 1], b = img.data[i * 4 + 2];
    const bgr = [b, g, r];
    for (let c = 0; c < 3; c++) out[c * n + i] = norm(bgr[c], c);
  }
  return out;
}

// ── the character list from a model's inference.yml (PostProcess.character_dict)
export function dictFromYaml(text) {
  const lines = text.split("\n"), out = [];
  let i = lines.findIndex((l) => /^\s*character_dict:\s*$/.test(l));
  if (i < 0) throw new Error("no character_dict in inference.yml");
  const indent = lines[i + 1].match(/^(\s*)- /)?.[1] ?? "  ";
  for (i += 1; i < lines.length; i++) {
    const l = lines[i];
    if (!l.startsWith(indent + "- ") && !l.startsWith(indent + "-\t")) break;
    let v = l.slice(indent.length + 2);
    if (v.startsWith("'")) v = v.slice(1, v.lastIndexOf("'")).replace(/''/g, "'");
    else if (v.startsWith('"')) v = JSON.parse(v);
    out.push(v);
  }
  return out;
}

// ── right-to-left scripts: the recogniser returns characters left to right as they look; put them in reading
// order, keeping runs of digits and Latin letters as they are
const RTL = /[֐-ࣿיִ-﷿ﹰ-﻿]/;
export function visualToLogical(s) {
  if (!RTL.test(s)) return s;
  const runs = s.match(/[0-9٠-٩۰-۹A-Za-z.,:%/+\-]+|[^0-9٠-٩۰-۹A-Za-z.,:%/+\-]+/g) || [];
  return runs.reverse().map((r) => (/^[0-9٠-٩۰-۹A-Za-z.,:%/+\-]+$/.test(r) ? r : [...r].reverse().join(""))).join("");
}

export class PaddleOCR {
  constructor({ ort, det, rec, dict, providers = ["wasm"] }) {
    Object.assign(this, { ort, detSrc: det, recSrc: rec, dict, providers });
  }

  async load() {
    if (this.det) return;
    const opts = { executionProviders: this.providers, graphOptimizationLevel: "all" };
    [this.det, this.rec] = await Promise.all([this.ort.InferenceSession.create(this.detSrc, opts), this.ort.InferenceSession.create(this.recSrc, opts)]);
    this.provider = this.providers[0];
  }

  async detect(img) {
    const ratio = Math.min(1, 960 / Math.max(img.width, img.height));
    const w = Math.max(32, Math.round((img.width * ratio) / 32) * 32), h = Math.max(32, Math.round((img.height * ratio) / 32) * 32);
    const small = resize(img, w, h);
    const x = new this.ort.Tensor("float32", toTensor(small, (v, c) => (v / 255 - MEAN[c]) / STD[c]), [1, 3, h, w]);
    const out = await this.det.run({ [this.det.inputNames[0]]: x });
    const prob = out[this.det.outputNames[0]].data;
    // connected regions of the thresholded map
    const seen = new Uint8Array(w * h), boxes = [], stack = [];
    for (let i = 0; i < w * h; i++) {
      if (seen[i] || prob[i] <= 0.3) continue;
      let x0 = w, y0 = h, x1 = 0, y1 = 0, sum = 0, n = 0;
      stack.push(i); seen[i] = 1;
      while (stack.length) {
        const p = stack.pop(), px = p % w, py = (p - px) / w;
        sum += prob[p]; n++;
        if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
        for (const q of [p - 1, p + 1, p - w, p + w]) {
          if (q < 0 || q >= w * h || seen[q] || prob[q] <= 0.3) continue;
          if ((q === p - 1 && px === 0) || (q === p + 1 && px === w - 1)) continue;
          seen[q] = 1; stack.push(q);
        }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
      if (n < 4 || Math.min(bw, bh) < 3 || sum / n < 0.6) continue;
      const d = (bw * bh * 1.5) / (2 * (bw + bh)); // DB unclip distance for a rectangle
      const sx = img.width / w, sy = img.height / h;
      boxes.push({ x0: (x0 - d) * sx, y0: (y0 - d) * sy, x1: (x1 + 1 + d) * sx, y1: (y1 + 1 + d) * sy });
    }
    return boxes;
  }

  async recognise(line) {
    const h = 48, w = Math.min(3200, Math.max(16, Math.ceil((line.width * h) / line.height)));
    const img = resize(line, w, h);
    const x = new this.ort.Tensor("float32", toTensor(img, (v) => (v / 255 - 0.5) / 0.5), [1, 3, h, w]);
    const out = await this.rec.run({ [this.rec.inputNames[0]]: x });
    const t = out[this.rec.outputNames[0]], [, steps, classes] = t.dims, p = t.data;
    let text = "", prev = -1, conf = 0, n = 0;
    for (let s = 0; s < steps; s++) {
      let best = 0, bv = -Infinity;
      for (let c = 0; c < classes; c++) { const v = p[s * classes + c]; if (v > bv) { bv = v; best = c; } }
      if (best !== 0 && best !== prev) { text += best - 1 < this.dict.length ? this.dict[best - 1] : " "; conf += bv; n++; }
      prev = best;
    }
    return { text: visualToLogical(text.trim()), conf: n ? conf / n : 0 };
  }

  // the page's text, lines in reading order, a blank line between paragraphs
  async read(img, { rtl = false, onProgress } = {}) {
    await this.load();
    const boxes = await this.detect(img);
    const lines = [];
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i], r = await this.recognise(crop(img, b.x0, b.y0, b.x1, b.y1));
      if (r.text && r.conf > 0.3) lines.push({ ...b, text: r.text });
      onProgress && onProgress((i + 1) / boxes.length);
      if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0)); // let the page repaint between lines
    }
    return layout(lines, rtl);
  }
}

// rows of boxes (by vertical overlap), each row read left to right (or right to left), paragraphs by gaps
export function layout(lines, rtl) {
  lines.sort((a, b) => a.y0 - b.y0);
  const rows = [];
  for (const l of lines) {
    const row = rows.find((r) => Math.min(r.y1, l.y1) - Math.max(r.y0, l.y0) > 0.5 * Math.min(r.y1 - r.y0, l.y1 - l.y0));
    if (row) { row.items.push(l); row.y0 = Math.min(row.y0, l.y0); row.y1 = Math.max(row.y1, l.y1); }
    else rows.push({ y0: l.y0, y1: l.y1, items: [l] });
  }
  rows.sort((a, b) => a.y0 - b.y0);
  const heights = rows.map((r) => r.y1 - r.y0).sort((a, b) => a - b), mid = heights[heights.length >> 1] || 1;
  let out = "", prev = null;
  for (const r of rows) {
    r.items.sort((a, b) => (rtl ? b.x0 - a.x0 : a.x0 - b.x0));
    if (prev && r.y0 - prev.y1 > 0.9 * mid) out += "\n";
    out += (out ? "\n" : "") + r.items.map((i) => i.text).join(rtl ? "  " : "   ");
    prev = r;
  }
  return out.replace(/\n{3,}/g, "\n\n").trim();
}

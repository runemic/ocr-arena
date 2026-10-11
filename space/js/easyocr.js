// EasyOCR in the browser: Jaided AI's CRAFT detector and CRNN recognizers (exported unchanged to ONNX in
// runemic-ai/easyocr-onnx), with EasyOCR's own pre- and post-processing:
//   detection    resize the long side to at most 2560 (multiple of 32), RGB normalised with ImageNet mean/std;
//                regions where text > 0.4 or link > 0.4, kept if their peak text score is over 0.7, grown by
//                EasyOCR's dilation; boxes on one line are merged when the gap is under half the line height
//   recognition  greyscale, height 64, x/255 scaled to [-1, 1]; greedy CTC (index 0 is the blank)
import { resize, crop, layout, visualToLogical } from "./paddle.js";

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

export class EasyOCR {
  constructor({ ort, det, rec, characters, providers = ["wasm"] }) {
    Object.assign(this, { ort, detSrc: det, recSrc: rec, chars: characters, providers });
  }

  async load() {
    if (this.det) return;
    const opts = { executionProviders: this.providers, graphOptimizationLevel: "all" };
    [this.det, this.rec] = await Promise.all([this.ort.InferenceSession.create(this.detSrc, opts), this.ort.InferenceSession.create(this.recSrc, opts)]);
  }

  async detect(img) {
    const ratio = Math.min(1, 2560 / Math.max(img.width, img.height));
    const tw = Math.round(img.width * ratio), th = Math.round(img.height * ratio);
    const w = Math.ceil(tw / 32) * 32, h = Math.ceil(th / 32) * 32;
    const small = resize(img, tw, th), x = new Float32Array(3 * w * h);
    for (let c = 0; c < 3; c++) x.fill(-MEAN[c] / STD[c], c * w * h, (c + 1) * w * h); // black padding right and bottom, as EasyOCR
    for (let yy = 0; yy < th; yy++) for (let xx = 0; xx < tw; xx++) {
      const i = (yy * tw + xx) * 4, o = yy * w + xx;
      for (let c = 0; c < 3; c++) x[c * w * h + o] = (small.data[i + c] - MEAN[c] * 255) / (STD[c] * 255);
    }
    const out = await this.det.run({ [this.det.inputNames[0]]: new this.ort.Tensor("float32", x, [1, 3, h, w]) });
    const t = out[this.det.outputNames[0]], [, mh, mw] = t.dims, s = t.data; // (1, h/2, w/2, 2)
    const text = new Float32Array(mh * mw), link = new Float32Array(mh * mw);
    for (let i = 0; i < mh * mw; i++) { text[i] = s[i * 2]; link[i] = s[i * 2 + 1]; }
    const seen = new Uint8Array(mh * mw), boxes = [], stack = [];
    const on = (i) => text[i] > 0.4 || link[i] > 0.4;
    for (let i = 0; i < mh * mw; i++) {
      if (seen[i] || !on(i)) continue;
      let x0 = mw, y0 = mh, x1 = 0, y1 = 0, n = 0, peak = 0, tx0 = mw, ty0 = mh, tx1 = 0, ty1 = 0;
      stack.push(i); seen[i] = 1;
      while (stack.length) {
        const p = stack.pop(), px = p % mw, py = (p - px) / mw;
        n++; if (text[p] > peak) peak = text[p];
        if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
        if (text[p] > 0.4) { if (px < tx0) tx0 = px; if (px > tx1) tx1 = px; if (py < ty0) ty0 = py; if (py > ty1) ty1 = py; }
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { // 8-connected, as cv2.connectedComponents
          const qx = px + dx, qy = py + dy;
          if ((dx || dy) && qx >= 0 && qy >= 0 && qx < mw && qy < mh) { const q = qy * mw + qx; if (!seen[q] && on(q)) { seen[q] = 1; stack.push(q); } }
        }
      }
      if (n < 10 || peak < 0.7 || tx1 < tx0) continue;
      // EasyOCR dilates the text pixels with a (1 + niter) square kernel: about niter / 2 on each side
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1, grow = Math.floor(Math.sqrt((n * Math.min(bw, bh)) / (bw * bh)) * 2) / 2;
      const k = 2 / ratio; // map pixels (half size) back to the page
      boxes.push({ x0: Math.max(0, (tx0 - grow) * k), y0: Math.max(0, (ty0 - grow) * k),
        x1: Math.min(img.width, (tx1 + 1 + grow) * k), y1: Math.min(img.height, (ty1 + 1 + grow) * k) });
    }
    return mergeLines(boxes);
  }

  async recognise(line) {
    const h = 64, w = Math.min(4000, Math.max(16, Math.ceil((line.width * h) / line.height)));
    const img = resize(line, w, h), x = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) x[i] = ((0.299 * img.data[i * 4] + 0.587 * img.data[i * 4 + 1] + 0.114 * img.data[i * 4 + 2]) / 255 - 0.5) / 0.5;
    const out = await this.rec.run({ [this.rec.inputNames[0]]: new this.ort.Tensor("float32", x, [1, 1, h, w]) });
    const t = out[this.rec.outputNames[0]], [, steps, classes] = t.dims, p = t.data;
    let text = "", prev = -1, conf = 0, n = 0;
    for (let s = 0; s < steps; s++) {
      let best = 0, bv = -Infinity, sum = 0;
      for (let c = 0; c < classes; c++) { const v = p[s * classes + c]; if (v > bv) { bv = v; best = c; } }
      for (let c = 0; c < classes; c++) sum += Math.exp(p[s * classes + c] - bv);
      if (best !== 0 && best !== prev) { text += this.chars[best] ?? ""; conf += 1 / sum; n++; }
      prev = best;
    }
    return { text: visualToLogical(text.trim()), conf: n ? conf / n : 0 };
  }

  async read(img, { rtl = false, onProgress } = {}) {
    await this.load();
    const boxes = await this.detect(img), lines = [];
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i], r = await this.recognise(crop(img, b.x0, b.y0, b.x1, b.y1));
      if (r.text && r.conf > 0.1) lines.push({ ...b, text: r.text });
      onProgress && onProgress((i + 1) / boxes.length);
      if (i % 4 === 3) await new Promise((r) => setTimeout(r, 0));
    }
    return layout(lines, rtl);
  }
}

// EasyOCR's group_text_box, simplified: boxes whose vertical centres are within half a line height join a line, and
// neighbours on that line join when the gap is under half the line height; then a 10% margin
function mergeLines(boxes) {
  boxes.sort((a, b) => (a.y0 + a.y1) / 2 - (b.y0 + b.y1) / 2);
  const rows = [];
  for (const b of boxes) {
    const cy = (b.y0 + b.y1) / 2, hh = b.y1 - b.y0;
    const row = rows.find((r) => Math.abs(r.cy - cy) < 0.5 * Math.min(r.h, hh) && Math.abs(r.h - hh) < 0.5 * Math.max(r.h, hh));
    if (row) { row.items.push(b); row.cy = (row.cy * (row.items.length - 1) + cy) / row.items.length; } else rows.push({ cy, h: hh, items: [b] });
  }
  const out = [];
  for (const r of rows) {
    r.items.sort((a, b) => a.x0 - b.x0);
    let cur = null;
    for (const b of r.items) {
      if (cur && b.x0 - cur.x1 < 0.5 * (cur.y1 - cur.y0)) { cur.x1 = Math.max(cur.x1, b.x1); cur.y0 = Math.min(cur.y0, b.y0); cur.y1 = Math.max(cur.y1, b.y1); }
      else { if (cur) out.push(cur); cur = { ...b }; }
    }
    if (cur) out.push(cur);
  }
  return out.map((b) => { const m = 0.1 * Math.min(b.x1 - b.x0, b.y1 - b.y0); return { x0: b.x0 - m, y0: b.y0 - m, x1: b.x1 + m, y1: b.y1 + m }; });
}

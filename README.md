# OCR Arena

Compare OCR engines side by side on your own image or PDF, in the browser:
**[huggingface.co/spaces/runemic-ai/ocr-arena](https://huggingface.co/spaces/runemic-ai/ocr-arena)**

PaddleOCR, EasyOCR, Tesseract, GLM-OCR and LightOnOCR run on the visitor's own CPU or GPU (WebGPU), so pages never
leave the browser. The Runemic engine is the one exception: it sends the page to the Runemic API.

Made by [Runemic](https://runemic.com).

## What's here

| Path | What it is |
|---|---|
| `space/` | The Hugging Face Space (a static Space: `index.html`, `js/`, `static/`). Published as is. The example pages, logo and fonts load from https://runemic.com/assets/ (kept in runemic/website). |
| `space/js/engines.js` | The engines and where each one runs |
| `space/js/paddle.js`, `easyocr.js`, `vlm.js` | PaddleOCR (PP-OCR, ONNX Runtime Web), EasyOCR (CRAFT + CRNN, ONNX), and vision-language models (Transformers.js) |
| `tests/arena_browser.mjs` | Reads every example page with each engine in Chrome; fails if an engine errors |
| `scripts/make-thumb.mjs` | Draws `space/thumbnail.png` |
| `.github/workflows/publish.yml` | Test, then publish `space/` to Hugging Face (needs the `HF_TOKEN` secret) |

## Run it locally

```sh
npm install
npm run serve          # http://localhost:8813
SITE=http://127.0.0.1:8813 CHROMIUM=/path/to/chrome npm test
```

## Licence

Apache-2.0 for the code in this repository. Each engine and model keeps its own licence (listed on the Space).

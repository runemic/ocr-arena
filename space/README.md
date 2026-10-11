---
title: OCR Arena
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: Compare OCR engines on images & PDFs, in your browser
thumbnail: https://huggingface.co/spaces/runemic-ai/ocr-arena/resolve/main/thumbnail.png
tags:
  - ocr
  - pdf
  - pdf-ocr
  - image-to-text
  - document-parsing
  - text-recognition
  - multilingual
  - webgpu
  - onnx
  - in-browser
  - persian
  - arabic
  - urdu
  - hindi
  - chinese
  - japanese
  - korean
  - russian
  - german
  - tesseract
  - paddleocr
  - easyocr
  - glm-ocr
  - lightonocr
  - transformers.js
  - vision-language-model
models:
  - PaddlePaddle/PP-OCRv6_small_det_onnx
  - PaddlePaddle/PP-OCRv6_small_rec_onnx
  - PaddlePaddle/arabic_PP-OCRv5_mobile_rec_onnx
  - PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_onnx
  - PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx
  - PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx
  - runemic-ai/easyocr-onnx
  - onnx-community/GLM-OCR-ONNX
  - onnx-community/LightOnOCR-2-1B-ONNX
  - zai-org/GLM-OCR
  - lightonai/LightOnOCR-2-1B
---

# OCR Arena

Drop an image or a PDF, pick up to three OCR engines and compare what each one reads, side by side. Made by
[Runemic](https://runemic.com).

Everything runs **in your browser**, on your own CPU or GPU: there is no server, and your pages never leave your
device (except with the Runemic engine, which reads the page on api.runemic.com).

| Engine | Runs on | Notes |
|---|---|---|
| [PaddleOCR](https://huggingface.co/PaddlePaddle) PP-OCRv6 / v5 | Your GPU (WebGPU), or your CPU | Baidu's detection and recognition models with ONNX Runtime Web; per-script models for Arabic-script, Devanagari, Cyrillic and Korean |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) 1.7 | Your GPU (WebGPU), or your CPU | Jaided AI's CRAFT detector and per-script CRNN recognizers, [converted to ONNX](https://huggingface.co/runemic-ai/easyocr-onnx); the Arabic-script and Hindi models are large (about 300 MB) |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) 0.9B | Your GPU (WebGPU) | Zhipu AI's vision-language OCR model with [Transformers.js](https://github.com/huggingface/transformers.js); the whole page to Markdown; about 780 MB |
| [LightOnOCR-2](https://huggingface.co/lightonai/LightOnOCR-2-1B) 1B | Your GPU (WebGPU) | LightOn's end-to-end OCR model, the same way; about 900 MB |
| [Tesseract](https://github.com/naptha/tesseract.js) 5 | Your CPU (WebAssembly) | The classic engine, 100+ languages |
| [Runemic](https://runemic.com) | API | Hosted vision models; 10 free pages a day |
| PDF text layer ([pdf.js](https://mozilla.github.io/pdf.js/)) | Instant | Not OCR: the text a digital PDF already has |

**Coming soon, on a GPU server:** Unlimited-OCR (Baidu), DeepSeek-OCR 2, PaddleOCR-VL, dots.ocr and Chandra OCR 2.

**PDFs:** choose a page range (up to 20 pages a run); each engine reads the pages one by one.

Models download once into the browser and stay cached. Each engine keeps its own licence.

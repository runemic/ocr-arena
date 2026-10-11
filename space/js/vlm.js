// Small OCR vision-language models (GLM-OCR, LightOnOCR) in the browser with Transformers.js on WebGPU.
// They read the whole page in one go and write it out as Markdown, token by token.
const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";
let lib = null;
async function tjs() {
  if (!lib) lib = import(TJS).then((T) => { T.env.allowLocalModels = false; return T; });
  return lib;
}

export class VLM {
  // repo: an onnx-community model; prompt: the text after the image ("" for prompt-free models);
  // imagesFirst: the processor takes (images, text) rather than (text, images); maxPixels: the page is scaled down to this
  constructor({ repo, prompt = "", imagesFirst = false, maxPixels = 1.2e6, maxTokens = 2048, gpu = true, f16 = true }) {
    Object.assign(this, { repo, prompt, imagesFirst, maxPixels, maxTokens, gpu, f16 });
  }

  load(onProgress) {
    if (!this.ready) {
      this.ready = (async () => {
        const T = await tjs(), seen = {};
        const progress_callback = (p) => { // download progress over all files
          if (p.status !== "progress" || !p.total) return;
          seen[p.file] = [p.loaded, p.total];
          const v = Object.values(seen), got = v.reduce((a, [l]) => a + l, 0), all = v.reduce((a, [, t]) => a + t, 0);
          onProgress && onProgress(got / all, "Downloading");
        };
        this.processor = await T.AutoProcessor.from_pretrained(this.repo);
        this.model = await T.AutoModelForImageTextToText.from_pretrained(this.repo, {
          device: this.gpu ? "webgpu" : "wasm", progress_callback,
          use_external_data_format: true, // every file we load keeps its weights in one .onnx_data file (not all are declared)
          // 4-bit weights for the vision encoder and the decoder. Quantized embeddings use GatherBlockQuantized, which
          // WebGPU has and the CPU (WASM) backend doesn't, so on the CPU they stay fp32. Without fp16 on the GPU, the
          // 4-bit models that compute in fp32
          dtype: !this.gpu ? { embed_tokens: "fp32", vision_encoder: "q4", decoder_model_merged: "q4" }
            : this.f16 ? { embed_tokens: "fp16", vision_encoder: "q4f16", decoder_model_merged: "q4f16" }
            : { embed_tokens: "quantized", vision_encoder: "q4", decoder_model_merged: "q4" } });
        this.T = T;
      })();
      this.ready.catch(() => { this.ready = null; });
    }
    return this.ready;
  }

  async read(canvas, { onProgress } = {}) {
    await this.load(onProgress);
    const { T, processor, model } = this;
    const s = Math.min(1, Math.sqrt(this.maxPixels / (canvas.width * canvas.height)));
    let c = canvas;
    if (s < 1) {
      c = document.createElement("canvas"); c.width = Math.round(canvas.width * s); c.height = Math.round(canvas.height * s);
      const ctx = c.getContext("2d"); ctx.imageSmoothingQuality = "high"; ctx.drawImage(canvas, 0, 0, c.width, c.height);
    }
    const image = await T.RawImage.fromCanvas(c);
    const content = [{ type: "image" }];
    if (this.prompt) content.push({ type: "text", text: this.prompt });
    const text = processor.apply_chat_template([{ role: "user", content }], { add_generation_prompt: true });
    const inputs = this.imagesFirst ? await processor(image, text) : await processor(text, image);
    let n = 0;
    const streamer = new T.TextStreamer(processor.tokenizer, { skip_prompt: true, skip_special_tokens: true,
      callback_function: () => { n++; onProgress && onProgress(Math.min(0.99, n / 400), "Writing"); } });
    const out = await model.generate({ ...inputs, max_new_tokens: this.maxTokens, do_sample: false, streamer });
    const ids = out.slice(null, [inputs.input_ids.dims.at(-1), null]);
    return processor.batch_decode(ids, { skip_special_tokens: true })[0].trim();
  }
}

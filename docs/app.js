import {createModel, predictPixels} from "./inference.mjs";

const $ = (id) => document.getElementById(id);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
let busy = false;
let model;
let overview;

function status(message) { $("status").textContent = message || ""; }

function renderOverview(data) {
  $("test-count").textContent = `${(data.split.official_test / 1000).toFixed(0)}k`;
  for (const [key, prefix] of [["classical_ml", "ml"], ["deep_learning", "dl"]]) {
    const test = data.models[key].test;
    $(`${prefix}-accuracy`).textContent = percent(test.accuracy);
    $(`${prefix}-f1`).textContent = test.macro_f1.toFixed(3);
    $(`${prefix}-bar`).style.width = `${test.accuracy * 100}%`;
  }
  const samples = data.samples.filter((sample) => sample.file.endsWith("-0.png"));
  $("samples").innerHTML = samples.map((sample) => `<button class="sample" type="button" data-file="${esc(sample.file)}" aria-label="Analyze ${esc(sample.label)} sample"><img src="./samples/${esc(sample.file)}" alt=""><span>${esc(sample.label)}</span></button>`).join("");
  $("samples").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-file]");
    if (!button) return;
    document.querySelectorAll(".sample.selected").forEach((node) => node.classList.remove("selected"));
    button.classList.add("selected");
    predict({sample: button.dataset.file});
  });
}

function predictionCard(title, result, type) {
  const tied = result.score <= 0.101;
  const label = tied ? "No clear prediction" : result.label;
  const alternatives = tied ? "Scores tied across all 10 classes" :
    `Next: ${result.top_three.slice(1).map((item) => `${esc(item.label)} ${percent(item.score)}`).join(" · ")}`;
  return `<div class="prediction-card ${type}"><div class="pred-head">${title}</div><div class="prediction-label">${esc(label)}</div><div class="prediction-score">${percent(result.score)} model score</div><div class="mini-track"><span style="width:${Math.max(result.score * 100, 2)}%"></span></div><div class="pred-alt">${alternatives}</div></div>`;
}

function renderPrediction(data) {
  $("result-empty").hidden = true;
  $("result-content").hidden = false;
  $("processed-image").src = data.processed_image;
  $("expected").hidden = !data.expected_label;
  $("expected").textContent = data.expected_label ? `Known test label: ${data.expected_label}` : "";
  $("prediction-cards").innerHTML = predictionCard("CLASSICAL ML", data.ml, "ml") + predictionCard("DEEP LEARNING", data.dl, "dl");
  status("");
  $("result-content").scrollIntoView({behavior: "smooth", block: "nearest"});
}

function canvasPixels(bitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.drawImage(bitmap, 0, 0);
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let index = 0; index < gray.length; index++) {
    const offset = index * 4;
    gray[index] = Math.round(0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2]);
  }
  return {pixels: gray, width: canvas.width, height: canvas.height};
}

function median(values) {
  values.sort((left, right) => left - right);
  return values[Math.floor(values.length / 2)];
}

function preprocessUpload(bitmap) {
  let {pixels, width, height} = canvasPixels(bitmap);
  if (width < 8 || height < 8) throw new Error("Image is too small; use at least 8 × 8 pixels.");
  const border = [];
  for (let x = 0; x < width; x++) {border.push(pixels[x], pixels[(height - 1) * width + x]);}
  for (let y = 0; y < height; y++) {border.push(pixels[y * width], pixels[y * width + width - 1]);}
  if (median(border) > 127) pixels = pixels.map((value) => 255 - value);

  let low = 255, high = 0;
  for (const value of pixels) {low = Math.min(low, value); high = Math.max(high, value);}
  if (high > low) pixels = pixels.map((value) => Math.max(0, Math.min(255, Math.floor((value - low) * 255 / (high - low)))));

  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (pixels[y * width + x] <= 28) continue;
      left = Math.min(left, x); right = Math.max(right, x);
      top = Math.min(top, y); bottom = Math.max(bottom, y);
    }
  }
  if (right < left) throw new Error("No visible object was found. Try a clearer image.");
  const cropWidth = right - left + 1, cropHeight = bottom - top + 1;
  const scale = Math.min(24 / cropWidth, 24 / cropHeight, 1);
  const destWidth = Math.max(1, Math.round(cropWidth * scale));
  const destHeight = Math.max(1, Math.round(cropHeight * scale));

  const source = document.createElement("canvas");
  source.width = width; source.height = height;
  const sourceContext = source.getContext("2d");
  const imageData = sourceContext.createImageData(width, height);
  for (let index = 0; index < pixels.length; index++) {
    const offset = index * 4;
    imageData.data[offset] = imageData.data[offset + 1] = imageData.data[offset + 2] = pixels[index];
    imageData.data[offset + 3] = 255;
  }
  sourceContext.putImageData(imageData, 0, 0);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 28;
  const context = canvas.getContext("2d", {willReadFrequently: true});
  context.imageSmoothingQuality = "high";
  context.drawImage(source, left, top, cropWidth, cropHeight,
    Math.floor((28 - destWidth) / 2), Math.floor((28 - destHeight) / 2), destWidth, destHeight);
  const output = context.getImageData(0, 0, 28, 28).data;
  const normalized = new Uint8Array(784);
  for (let index = 0; index < 784; index++) normalized[index] = output[index * 4];
  return normalized;
}

function processedImageUrl(pixels) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 224;
  const context = canvas.getContext("2d");
  const image = context.createImageData(224, 224);
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const value = pixels[Math.floor(y / 8) * 28 + Math.floor(x / 8)];
      const offset = (y * 224 + x) * 4;
      image.data[offset] = image.data[offset + 1] = image.data[offset + 2] = value;
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

async function predict(payload) {
  if (busy || !model) return;
  busy = true;
  status("Analyzing image...");
  try {
    await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
    let pixels, expected = null;
    if (payload.sample) {
      const sample = overview.samples.find((item) => item.file === payload.sample);
      if (!sample) throw new Error("Sample was not found.");
      const response = await fetch(`./samples/${encodeURIComponent(sample.file)}`);
      if (!response.ok) throw new Error("Could not load the sample image.");
      const bitmap = await createImageBitmap(await response.blob());
      pixels = canvasPixels(bitmap).pixels;
      expected = sample.label;
      bitmap.close();
    } else {
      const bitmap = await createImageBitmap(payload.file);
      pixels = preprocessUpload(bitmap);
      bitmap.close();
    }
    renderPrediction({processed_image: processedImageUrl(pixels), expected_label: expected,
      ...predictPixels(pixels, model)});
  } catch (error) {
    status(error.message || "Could not analyze this image.");
  } finally {
    busy = false;
  }
}

function handleFile(file) {
  if (!file) return;
  document.querySelectorAll(".sample.selected").forEach((node) => node.classList.remove("selected"));
  if (file.size > 3000000) {status("Use an image smaller than 3 MB."); return;}
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {status("Use a PNG, JPEG, or WebP image."); return;}
  predict({file});
}

async function init() {
  const input = $("image-input");
  const zone = $("dropzone");
  input.addEventListener("change", () => handleFile(input.files[0]));
  zone.addEventListener("keydown", (event) => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); input.click();}});
  for (const name of ["dragenter", "dragover"]) zone.addEventListener(name, (event) => {event.preventDefault(); zone.classList.add("dragging");});
  for (const name of ["dragleave", "drop"]) zone.addEventListener(name, (event) => {event.preventDefault(); zone.classList.remove("dragging");});
  zone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
  status("Loading trained models...");
  try {
    const responses = await Promise.all(["overview.json", "model-manifest.json", "weights.bin"].map((file) => fetch(`./${file}`)));
    if (responses.some((response) => !response.ok)) throw new Error("Could not load trained models.");
    const [overviewData, manifest, binary] = await Promise.all([responses[0].json(), responses[1].json(), responses[2].arrayBuffer()]);
    model = createModel(manifest, binary);
    overview = overviewData;
    renderOverview(overview);
    status("");
  } catch (error) {status(error.message || "Could not load trained models.");}
}

init();

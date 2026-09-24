const $ = (id) => document.getElementById(id);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
let busy = false;

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
  $("samples").innerHTML = samples.map((sample) => `<button class="sample" type="button" data-file="${esc(sample.file)}" aria-label="Analyze ${esc(sample.label)} sample"><img src="/samples/${esc(sample.file)}" alt=""><span>${esc(sample.label)}</span></button>`).join("");
  $("samples").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-file]");
    if (!button) return;
    document.querySelectorAll(".sample.selected").forEach((node) => node.classList.remove("selected"));
    button.classList.add("selected");
    predict({sample: button.dataset.file});
  });
}

function predictionCard(title, result, type) {
  const alternatives = result.top_three.slice(1).map((item) => `${esc(item.label)} ${percent(item.score)}`).join(" · ");
  return `<div class="prediction-card ${type}"><div class="pred-head">${title}</div><div class="prediction-label">${esc(result.label)}</div><div class="prediction-score">${percent(result.score)} model score</div><div class="mini-track"><span style="width:${Math.max(result.score * 100, 2)}%"></span></div><div class="pred-alt">Next: ${alternatives}</div></div>`;
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

async function predict(payload) {
  if (busy) return;
  busy = true;
  status("Analyzing image...");
  try {
    const response = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    renderPrediction(data);
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
  const reader = new FileReader();
  reader.onload = () => predict({image: reader.result});
  reader.onerror = () => status("Could not read this image.");
  reader.readAsDataURL(file);
}

async function init() {
  const input = $("image-input");
  const zone = $("dropzone");
  input.addEventListener("change", () => handleFile(input.files[0]));
  zone.addEventListener("keydown", (event) => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); input.click();}});
  for (const name of ["dragenter", "dragover"]) zone.addEventListener(name, (event) => {event.preventDefault(); zone.classList.add("dragging");});
  for (const name of ["dragleave", "drop"]) zone.addEventListener(name, (event) => {event.preventDefault(); zone.classList.remove("dragging");});
  zone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
  try {
    const response = await fetch("/api/overview");
    if (!response.ok) throw new Error("Could not load model results.");
    renderOverview(await response.json());
  } catch (error) {status(error.message);}
}

init();

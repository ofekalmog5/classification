import "./style.css";

type ClassItem = {
  id: string;
  name: string;
  color: string;
};

type VectorLayer = {
  id: string;
  name: string;
  filePath: string;
  classId: string;
};

type FeatureFlags = {
  spectral: boolean;
  texture: boolean;
  indices: boolean;
};

type BandMap = {
  redBand: number;
  nirBand: number;
};

type AppState = {
  rasterPath: string;
  backendUrl: string;
  classes: ClassItem[];
  vectorLayers: VectorLayer[];
  smoothing: "superpixels" | "median";
  featureFlags: FeatureFlags;
  bandMap: BandMap;
};

const state: AppState = {
  rasterPath: "",
  backendUrl: "http://127.0.0.1:8000",
  classes: [],
  vectorLayers: [],
  smoothing: "superpixels",
  featureFlags: {
    spectral: true,
    texture: false,
    indices: true
  },
  bandMap: {
    redBand: 3,
    nirBand: 4
  }
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("App root not found");

app.innerHTML = `
  <div class="shell">
    <header class="hero">
      <div class="hero-copy">
        <p class="kicker">Ortho / Multispectral Materials</p>
        <h1>Material Classification Studio</h1>
        <p class="subtitle">Load imagery, attach vector layers, and run smooth classification.</p>
      </div>
      <div class="hero-panel">
        <div class="field">
          <label>Backend URL</label>
          <input id="backend-url" type="text" value="${state.backendUrl}" />
        </div>
        <div class="field">
          <label>Raster image</label>
          <input id="raster-file" type="file" />
          <p class="hint" id="raster-name">No raster selected</p>
        </div>
        <div class="field">
          <label>Smoothing</label>
          <select id="smoothing">
            <option value="superpixels">Superpixels + majority</option>
            <option value="median">Median filter</option>
          </select>
        </div>
        <div class="field">
          <label>Features</label>
          <div class="pill-group">
            <label class="pill"><input id="feat-spectral" type="checkbox" checked /> Spectral</label>
            <label class="pill"><input id="feat-texture" type="checkbox" /> Texture</label>
            <label class="pill"><input id="feat-indices" type="checkbox" checked /> Indices</label>
          </div>
        </div>
        <div class="field">
          <label>Band mapping (1-based)</label>
          <div class="row">
            <input id="band-red" type="number" min="1" value="${state.bandMap.redBand}" />
            <input id="band-nir" type="number" min="1" value="${state.bandMap.nirBand}" />
          </div>
          <p class="hint">Red band, NIR band for NDVI</p>
        </div>
      </div>
    </header>

    <section class="grid">
      <div class="card">
        <h2>Classes</h2>
        <div class="row">
          <input id="class-name" type="text" placeholder="Class name" />
          <input id="class-color" type="color" value="#1f7a8c" />
          <button id="add-class" class="primary">Add</button>
        </div>
        <ul id="class-list" class="list"></ul>
      </div>

      <div class="card">
        <h2>Vector layers</h2>
        <div class="row">
          <input id="vector-file" type="file" />
          <select id="vector-class"></select>
          <button id="add-vector" class="primary">Attach</button>
        </div>
        <ul id="vector-list" class="list"></ul>
      </div>
    </section>

    <section class="card output">
      <div class="row space">
        <div>
          <h2>Run</h2>
          <p class="subtitle">Generate a smooth material map using the selected settings.</p>
        </div>
        <button id="run" class="accent">Run classification</button>
      </div>
      <pre id="output" class="output-box">Waiting...</pre>
    </section>
  </div>
`;

const backendUrlInput = document.querySelector<HTMLInputElement>("#backend-url");
const rasterInput = document.querySelector<HTMLInputElement>("#raster-file");
const rasterName = document.querySelector<HTMLParagraphElement>("#raster-name");
const smoothingSelect = document.querySelector<HTMLSelectElement>("#smoothing");
const classNameInput = document.querySelector<HTMLInputElement>("#class-name");
const classColorInput = document.querySelector<HTMLInputElement>("#class-color");
const addClassButton = document.querySelector<HTMLButtonElement>("#add-class");
const classList = document.querySelector<HTMLUListElement>("#class-list");
const vectorFileInput = document.querySelector<HTMLInputElement>("#vector-file");
const vectorClassSelect = document.querySelector<HTMLSelectElement>("#vector-class");
const addVectorButton = document.querySelector<HTMLButtonElement>("#add-vector");
const vectorList = document.querySelector<HTMLUListElement>("#vector-list");
const runButton = document.querySelector<HTMLButtonElement>("#run");
const outputBox = document.querySelector<HTMLPreElement>("#output");

const featSpectral = document.querySelector<HTMLInputElement>("#feat-spectral");
const featTexture = document.querySelector<HTMLInputElement>("#feat-texture");
const featIndices = document.querySelector<HTMLInputElement>("#feat-indices");
const bandRedInput = document.querySelector<HTMLInputElement>("#band-red");
const bandNirInput = document.querySelector<HTMLInputElement>("#band-nir");

if (!backendUrlInput || !rasterInput || !rasterName || !smoothingSelect || !classNameInput) {
  throw new Error("UI elements missing");
}
if (!classColorInput || !addClassButton || !classList || !vectorFileInput) {
  throw new Error("UI elements missing");
}
if (!vectorClassSelect || !addVectorButton || !vectorList || !runButton || !outputBox) {
  throw new Error("UI elements missing");
}
if (!featSpectral || !featTexture || !featIndices) {
  throw new Error("UI elements missing");
}
if (!bandRedInput || !bandNirInput) {
  throw new Error("UI elements missing");
}

function renderClasses() {
  classList.innerHTML = "";
  vectorClassSelect.innerHTML = "";

  if (state.classes.length === 0) {
    classList.innerHTML = "<li class=\"empty\">No classes yet</li>";
    const emptyOption = document.createElement("option");
    emptyOption.textContent = "Add a class first";
    emptyOption.value = "";
    vectorClassSelect.appendChild(emptyOption);
    vectorClassSelect.disabled = true;
    return;
  }

  vectorClassSelect.disabled = false;
  state.classes.forEach((item) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class=\"swatch\" style=\"background:${item.color}\"></span>${item.name}`;
    classList.appendChild(li);

    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    vectorClassSelect.appendChild(option);
  });
}

function renderVectors() {
  vectorList.innerHTML = "";
  if (state.vectorLayers.length === 0) {
    vectorList.innerHTML = "<li class=\"empty\">No vector layers attached</li>";
    return;
  }

  state.vectorLayers.forEach((layer) => {
    const className = state.classes.find((c) => c.id === layer.classId)?.name ?? "Unknown";
    const li = document.createElement("li");
    li.textContent = `${layer.name} -> ${className}`;
    vectorList.appendChild(li);
  });
}

backendUrlInput.addEventListener("input", () => {
  state.backendUrl = backendUrlInput.value.trim();
});

rasterInput.addEventListener("change", () => {
  const file = rasterInput.files?.[0];
  const path = (file as unknown as { path?: string })?.path ?? file?.name ?? "";
  state.rasterPath = path;
  rasterName.textContent = path ? `Selected: ${path}` : "No raster selected";
});

smoothingSelect.addEventListener("change", () => {
  state.smoothing = smoothingSelect.value === "median" ? "median" : "superpixels";
});

featSpectral.addEventListener("change", () => {
  state.featureFlags.spectral = featSpectral.checked;
});
featTexture.addEventListener("change", () => {
  state.featureFlags.texture = featTexture.checked;
});
featIndices.addEventListener("change", () => {
  state.featureFlags.indices = featIndices.checked;
});

bandRedInput.addEventListener("input", () => {
  const value = Number.parseInt(bandRedInput.value, 10);
  state.bandMap.redBand = Number.isFinite(value) && value > 0 ? value : 1;
});

bandNirInput.addEventListener("input", () => {
  const value = Number.parseInt(bandNirInput.value, 10);
  state.bandMap.nirBand = Number.isFinite(value) && value > 0 ? value : 1;
});

addClassButton.addEventListener("click", () => {
  const name = classNameInput.value.trim();
  if (!name) return;
  const color = classColorInput.value;
  state.classes.push({
    id: crypto.randomUUID(),
    name,
    color
  });
  classNameInput.value = "";
  renderClasses();
});

addVectorButton.addEventListener("click", () => {
  const file = vectorFileInput.files?.[0];
  const classId = vectorClassSelect.value;
  if (!file || !classId) return;
  const path = (file as unknown as { path?: string })?.path ?? file.name;
  state.vectorLayers.push({
    id: crypto.randomUUID(),
    name: file.name,
    filePath: path,
    classId
  });
  vectorFileInput.value = "";
  renderVectors();
});

runButton.addEventListener("click", async () => {
  outputBox.textContent = "Running...";
  const payload = {
    rasterPath: state.rasterPath,
    classes: state.classes,
    vectorLayers: state.vectorLayers,
    smoothing: state.smoothing,
    featureFlags: state.featureFlags,
    bandMap: state.bandMap
  };

  try {
    const response = await fetch(`${state.backendUrl}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const json = await response.json();
    outputBox.textContent = JSON.stringify(json, null, 2);
  } catch (error) {
    outputBox.textContent = `Error: ${String(error)}`;
  }
});

renderClasses();
renderVectors();

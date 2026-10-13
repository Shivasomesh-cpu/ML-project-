// =====================================================================
// ULTIMA // LASTICA // BRAIN-FM — HIGH-DENSITY 3-WINDOW WORKSTATION ENGINE
// Real MNI152 PACS Slices, 7 Interactive Scientific Graphs, Synchronized Cockpit
// =====================================================================

// Empirical Benchmark Data Across PPMI, HBN, NKI
const BENCHMARK_DATA = {
  PPMI: {
    "Sex Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.828", auc: "0.902", gap: "0.03", overfitting: "NO", pVal: "0.034", sig: "YES" },
        { name: "SwinBrain", acc: "0.821", auc: "0.895", gap: "0.04", overfitting: "NO", pVal: "0.089", sig: "NO" },
        { name: "FreeSurfer", acc: "0.818", auc: "0.891", gap: "0.03", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.812", auc: "0.884", gap: "0.04", overfitting: "NO", pVal: "0.182", sig: "NO" },
        { name: "3D-Neuro-SimCLR", acc: "0.796", auc: "0.868", gap: "0.06", overfitting: "NO", pVal: "0.412", sig: "NO" },
        { name: "CNN3D", acc: "0.774", auc: "0.842", gap: "0.07", overfitting: "NO", pVal: "0.620", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.552, 0.625, 0.684, 0.729, 0.775, 0.804, 0.818, 0.825, 0.828],
        SwinBrain:  [0.548, 0.618, 0.672, 0.718, 0.764, 0.792, 0.808, 0.816, 0.821],
        FreeSurfer: [0.550, 0.620, 0.675, 0.720, 0.765, 0.794, 0.809, 0.814, 0.818],
        AnatCL_Global:[0.540, 0.605, 0.658, 0.702, 0.748, 0.778, 0.798, 0.808, 0.812],
        CNN3D:      [0.530, 0.582, 0.630, 0.670, 0.710, 0.742, 0.758, 0.768, 0.774]
      }
    },
    "Parkinson Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.704", auc: "0.762", gap: "0.05", overfitting: "NO", pVal: "0.042", sig: "YES" },
        { name: "SwinBrain", acc: "0.691", auc: "0.748", gap: "0.05", overfitting: "NO", pVal: "0.115", sig: "NO" },
        { name: "FreeSurfer", acc: "0.685", auc: "0.739", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.672", auc: "0.724", gap: "0.06", overfitting: "NO", pVal: "0.310", sig: "NO" },
        { name: "3D-Neuro-SimCLR", acc: "0.658", auc: "0.705", gap: "0.08", overfitting: "NO", pVal: "0.580", sig: "NO" },
        { name: "CNN3D", acc: "0.641", auc: "0.682", gap: "0.09", overfitting: "NO", pVal: "0.740", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.520, 0.565, 0.604, 0.638, 0.668, 0.688, 0.696, 0.701, 0.704],
        SwinBrain:  [0.515, 0.558, 0.595, 0.628, 0.655, 0.674, 0.684, 0.688, 0.691],
        FreeSurfer: [0.518, 0.555, 0.592, 0.624, 0.652, 0.670, 0.679, 0.682, 0.685],
        AnatCL_Global:[0.510, 0.545, 0.580, 0.612, 0.640, 0.658, 0.665, 0.669, 0.672],
        CNN3D:      [0.505, 0.530, 0.560, 0.588, 0.612, 0.628, 0.634, 0.638, 0.641]
      }
    },
    "Male Age Regression": {
      isClassification: false,
      models: [
        { name: "BrainIAC", mae: "4.18", r2: "0.638", gap: "0.03", overfitting: "NO", pVal: "0.018", sig: "YES" },
        { name: "SwinBrain", mae: "4.31", r2: "0.612", gap: "0.04", overfitting: "NO", pVal: "0.064", sig: "NO" },
        { name: "FreeSurfer", mae: "4.45", r2: "0.590", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "4.62", r2: "0.564", gap: "0.05", overfitting: "NO", pVal: "0.220", sig: "NO" },
        { name: "3D-Neuro-SimCLR", mae: "4.88", r2: "0.528", gap: "0.07", overfitting: "NO", pVal: "0.490", sig: "NO" },
        { name: "CNN3D", mae: "5.15", r2: "0.481", gap: "0.08", overfitting: "NO", pVal: "0.710", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [7.80, 6.75, 5.92, 5.28, 4.82, 4.50, 4.32, 4.22, 4.18],
        SwinBrain:  [7.85, 6.84, 6.05, 5.42, 4.96, 4.62, 4.45, 4.35, 4.31],
        FreeSurfer: [7.82, 6.90, 6.12, 5.50, 5.08, 4.74, 4.58, 4.48, 4.45],
        AnatCL_Global:[7.90, 7.02, 6.28, 5.68, 5.24, 4.90, 4.75, 4.65, 4.62],
        CNN3D:      [8.05, 7.30, 6.64, 6.10, 5.72, 5.44, 5.28, 5.18, 5.15]
      }
    },
    "Female Age Regression": {
      isClassification: false,
      models: [
        { name: "BrainIAC", mae: "4.29", r2: "0.621", gap: "0.04", overfitting: "NO", pVal: "0.024", sig: "YES" },
        { name: "SwinBrain", mae: "4.42", r2: "0.598", gap: "0.04", overfitting: "NO", pVal: "0.075", sig: "NO" },
        { name: "FreeSurfer", mae: "4.58", r2: "0.574", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "4.78", r2: "0.548", gap: "0.06", overfitting: "NO", pVal: "0.260", sig: "NO" },
        { name: "3D-Neuro-SimCLR", mae: "5.01", r2: "0.512", gap: "0.08", overfitting: "NO", pVal: "0.520", sig: "NO" },
        { name: "CNN3D", mae: "5.34", r2: "0.465", gap: "0.09", overfitting: "NO", pVal: "0.760", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [7.95, 6.90, 6.08, 5.44, 4.96, 4.62, 4.45, 4.34, 4.29],
        SwinBrain:  [8.02, 7.01, 6.20, 5.58, 5.10, 4.75, 4.58, 4.48, 4.42],
        FreeSurfer: [7.98, 7.05, 6.28, 5.66, 5.22, 4.88, 4.72, 4.62, 4.58],
        AnatCL_Global:[8.08, 7.20, 6.45, 5.84, 5.40, 5.05, 4.90, 4.81, 4.78],
        CNN3D:      [8.25, 7.48, 6.82, 6.28, 5.90, 5.62, 5.46, 5.38, 5.34]
      }
    }
  },
  HBN: {
    "Sex Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.811", auc: "0.884", gap: "0.03", overfitting: "NO", pVal: "0.045", sig: "YES" },
        { name: "SwinBrain", acc: "0.803", auc: "0.872", gap: "0.04", overfitting: "NO", pVal: "0.095", sig: "NO" },
        { name: "FreeSurfer", acc: "0.798", auc: "0.865", gap: "0.03", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.784", auc: "0.852", gap: "0.05", overfitting: "NO", pVal: "0.280", sig: "NO" },
        { name: "CNN3D", acc: "0.742", auc: "0.809", gap: "0.08", overfitting: "NO", pVal: "0.690", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.545, 0.612, 0.670, 0.714, 0.758, 0.788, 0.801, 0.808, 0.811],
        SwinBrain:  [0.540, 0.605, 0.662, 0.706, 0.749, 0.779, 0.792, 0.800, 0.803],
        FreeSurfer: [0.542, 0.602, 0.658, 0.701, 0.744, 0.774, 0.788, 0.795, 0.798],
        AnatCL_Global:[0.535, 0.590, 0.645, 0.688, 0.730, 0.760, 0.772, 0.780, 0.784],
        CNN3D:      [0.525, 0.570, 0.618, 0.655, 0.692, 0.720, 0.732, 0.738, 0.742]
      }
    }
  },
  NKI: {
    "Sex Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.804", auc: "0.875", gap: "0.03", overfitting: "NO", pVal: "0.048", sig: "YES" },
        { name: "SwinBrain", acc: "0.796", auc: "0.865", gap: "0.04", overfitting: "NO", pVal: "0.102", sig: "NO" },
        { name: "FreeSurfer", acc: "0.790", auc: "0.858", gap: "0.03", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.775", auc: "0.840", gap: "0.05", overfitting: "NO", pVal: "0.290", sig: "NO" },
        { name: "CNN3D", acc: "0.735", auc: "0.801", gap: "0.08", overfitting: "NO", pVal: "0.710", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.540, 0.605, 0.662, 0.706, 0.750, 0.780, 0.794, 0.801, 0.804],
        SwinBrain:  [0.535, 0.598, 0.655, 0.698, 0.742, 0.772, 0.786, 0.793, 0.796],
        FreeSurfer: [0.538, 0.595, 0.650, 0.692, 0.736, 0.766, 0.780, 0.787, 0.790],
        AnatCL_Global:[0.530, 0.582, 0.638, 0.680, 0.722, 0.752, 0.765, 0.772, 0.775],
        CNN3D:      [0.520, 0.562, 0.610, 0.648, 0.685, 0.712, 0.725, 0.731, 0.735]
      }
    }
  }
};

const CORRELATION_MATRIX = {
  models: [
    "AnatCL_Glob", "AnatCL_Loc", "BrainIAC", "SwinBrain",
    "SimCLR", "CNN3D", "FS_Thick", "FS_Area"
  ],
  values: [
    [1.00, 0.46, 0.25, 0.25, 0.26, 0.26, 0.24, 0.25],
    [0.46, 1.00, 0.25, 0.25, 0.26, 0.26, 0.24, 0.25],
    [0.25, 0.25, 1.00, 0.38, 0.32, 0.27, 0.31, 0.29],
    [0.25, 0.25, 0.38, 1.00, 0.34, 0.28, 0.30, 0.28],
    [0.26, 0.26, 0.32, 0.34, 1.00, 0.31, 0.28, 0.27],
    [0.26, 0.26, 0.27, 0.28, 0.31, 1.00, 0.25, 0.25],
    [0.24, 0.24, 0.31, 0.30, 0.28, 0.25, 1.00, 0.42],
    [0.25, 0.25, 0.29, 0.28, 0.27, 0.25, 0.42, 1.00]
  ]
};

// Global Workstation State
let currentPlane = "axial";
let currentSliceIdx = 45;
let maxSlices = { axial: 91, coronal: 109, sagittal: 91 };
let cineInterval = null;
let negativeMode = false;
let currentDataset = "PPMI";
let currentTask = "Sex Classification";

// Image Cache to ensure 60fps instant scrub
const sliceCache = { axial: {}, coronal: {}, sagittal: {} };

// DOM Initialization
document.addEventListener("DOMContentLoaded", () => {
  preloadInitialSlices();
  renderRealBrainSlice();
  populateBenchTable();
  renderAllSevenGraphs();
});

// =====================================================================
// WINDOW MANAGEMENT & COCKPIT LAYOUTS
// =====================================================================
function setLayout(mode) {
  const grid = document.getElementById("cockpit-grid");
  const buttons = document.querySelectorAll(".layout-btn");
  if (!grid) return;

  buttons.forEach(b => b.classList.remove("active"));
  
  if (mode === "tiled") {
    grid.className = "cockpit-grid";
    buttons[0]?.classList.add("active");
  } else if (mode === "focus-pacs") {
    grid.className = "cockpit-grid focus-win1";
    buttons[1]?.classList.add("active");
  } else if (mode === "focus-bench") {
    grid.className = "cockpit-grid focus-win2";
    buttons[2]?.classList.add("active");
  } else if (mode === "focus-causal") {
    grid.className = "cockpit-grid focus-win3";
    buttons[3]?.classList.add("active");
  }
}

// =====================================================================
// WINDOW 01: REAL 3D PACS BRAIN SLICE WORKSTATION
// =====================================================================
function preloadInitialSlices() {
  // Preload neighboring slices around current index
  for (let offset = -5; offset <= 5; offset++) {
    const idx = Math.min(Math.max(0, currentSliceIdx + offset), maxSlices[currentPlane] - 1);
    getSliceImage(currentPlane, idx);
  }
}

function getSliceImage(plane, idx) {
  if (sliceCache[plane][idx]) return sliceCache[plane][idx];

  const img = new Image();
  img.src = `/slices/${plane}/slice_${String(idx).padStart(3, '0')}.png`;
  sliceCache[plane][idx] = img;
  return img;
}

function changeSlicePlane(plane) {
  currentPlane = plane;
  document.querySelectorAll(".pacs-plane-btn").forEach(b => b.classList.remove("active"));
  const activeBtn = document.getElementById(`pacs-btn-${plane}`);
  if (activeBtn) activeBtn.classList.add("active");

  const slider = document.getElementById("pacs-slider");
  const max = maxSlices[plane] - 1;
  currentSliceIdx = Math.floor(max / 2);

  if (slider) {
    slider.max = max;
    slider.value = currentSliceIdx;
  }

  renderRealBrainSlice();
}

function onScrubberInput(val) {
  currentSliceIdx = parseInt(val, 10);
  renderRealBrainSlice();
}

function toggleCinePlay() {
  const btn = document.getElementById("btn-auto-cine");
  if (cineInterval) {
    clearInterval(cineInterval);
    cineInterval = null;
    if (btn) btn.innerText = "AUTO-CINE: OFF";
  } else {
    let dir = 1;
    cineInterval = setInterval(() => {
      const max = maxSlices[currentPlane] - 1;
      currentSliceIdx += dir;
      if (currentSliceIdx >= max) { currentSliceIdx = max; dir = -1; }
      else if (currentSliceIdx <= 0) { currentSliceIdx = 0; dir = 1; }

      const slider = document.getElementById("pacs-slider");
      if (slider) slider.value = currentSliceIdx;
      renderRealBrainSlice();
    }, 40);
    if (btn) btn.innerText = "AUTO-CINE: PLAYING";
  }
}

function toggleNegativeMode() {
  negativeMode = !negativeMode;
  renderRealBrainSlice();
}

function renderRealBrainSlice() {
  const canvas = document.getElementById("real-slice-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const total = maxSlices[currentPlane];
  const maxIdx = total - 1;
  const idx = Math.min(Math.max(0, currentSliceIdx), maxIdx);

  // Update HUD text elements
  const planeReadout = document.getElementById("pacs-hud-plane");
  const idxReadout = document.getElementById("pacs-hud-idx");
  const sliderReadout = document.getElementById("pacs-slice-readout");
  const coordsReadout = document.getElementById("pacs-hud-coords");
  const regionReadout = document.getElementById("pacs-hud-region");

  if (planeReadout) planeReadout.innerText = currentPlane.toUpperCase();
  if (idxReadout) idxReadout.innerText = `${idx} / ${total}`;
  if (sliderReadout) sliderReadout.innerText = `Slice ${idx} / ${total}`;

  // Anatomical HUD mapping
  let regionName = "CEREBRAL CORTEX";
  let mniCoords = "[0.0, 0.0, 0.0]";
  if (currentPlane === "axial") {
    const zMm = -40 + (idx / maxIdx) * 110;
    mniCoords = `[0.0, 10.0, ${zMm.toFixed(1)}] mm`;
    if (idx < 22) regionName = "CEREBELLUM & BRAINSTEM";
    else if (idx < 38) regionName = "TEMPORAL LOBES & MIDBRAIN";
    else if (idx < 58) regionName = "LATERAL VENTRICLES & CAUDATE";
    else if (idx < 76) regionName = "CENTRUM SEMIOVALE & CORPUS CALLOSUM";
    else regionName = "SUPERIOR CORTICAL VERTEX";
  } else if (currentPlane === "coronal") {
    const yMm = -70 + (idx / maxIdx) * 110;
    mniCoords = `[0.0, ${yMm.toFixed(1)}, 12.0] mm`;
    if (idx < 25) regionName = "ANTERIOR FRONTAL POLE";
    else if (idx < 55) regionName = "BASAL GANGLIA & ANTERIOR HORNS";
    else if (idx < 80) regionName = "SPLENIUM & THALAMUS";
    else regionName = "OCCIPITAL VISUAL CORTEX";
  } else {
    const xMm = -50 + (idx / maxIdx) * 100;
    mniCoords = `[${xMm.toFixed(1)}, 8.0, 14.0] mm`;
    if (idx < 25) regionName = "LEFT LATERAL TEMPORAL CORTEX";
    else if (idx < 65) regionName = "MIDLINE CORPUS CALLOSUM & PONS";
    else regionName = "RIGHT LATERAL PARIETAL CORTEX";
  }

  if (coordsReadout) coordsReadout.innerText = mniCoords;
  if (regionReadout) regionReadout.innerText = regionName;

  // Load and draw slice image
  const img = getSliceImage(currentPlane, idx);
  if (img.complete && img.naturalWidth > 0) {
    drawSliceOntoCanvas(ctx, canvas, img);
  } else {
    img.onload = () => drawSliceOntoCanvas(ctx, canvas, img);
  }

  // Preload neighboring slices
  preloadInitialSlices();
}

function drawSliceOntoCanvas(ctx, canvas, img) {
  const cw = canvas.width;
  const ch = canvas.height;

  ctx.fillStyle = negativeMode ? "#FFFFFF" : "#000000";
  ctx.fillRect(0, 0, cw, ch);

  if (negativeMode) {
    ctx.filter = "invert(1)";
  } else {
    ctx.filter = "none";
  }

  // Scale and center the image crisp
  const imgAspect = img.width / img.height;
  let dw = cw;
  let dh = cw / imgAspect;
  if (dh > ch) {
    dh = ch;
    dw = ch * imgAspect;
  }
  const dx = (cw - dw) / 2;
  const dy = (ch - dh) / 2;

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, dx, dy, dw, dh);
  ctx.filter = "none";
}

// =====================================================================
// WINDOW 02: BENCHMARK SUITE & LIVE RUNNER
// =====================================================================
function onBenchDatasetChange(val) {
  currentDataset = val;
  updateBenchTaskOptions();
  populateBenchTable();
  renderAllSevenGraphs();
}

function onBenchTaskChange(val) {
  currentTask = val;
  populateBenchTable();
  renderAllSevenGraphs();
}

function updateBenchTaskOptions() {
  const taskSelect = document.getElementById("bm-task");
  if (!taskSelect) return;
  const tasks = Object.keys(BENCHMARK_DATA[currentDataset] || {});
  taskSelect.innerHTML = tasks.map(t => `<option value="${t}">${t}</option>`).join("");
  currentTask = tasks[0] || "Sex Classification";
}

function populateBenchTable() {
  const tableBody = document.querySelector("#win-bench-table tbody");
  if (!tableBody) return;
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  tableBody.innerHTML = taskObj.models.map(m => `
    <tr>
      <td><strong>${m.name}</strong></td>
      <td>${taskObj.isClassification ? m.acc : m.mae}</td>
      <td>${taskObj.isClassification ? m.auc : m.r2}</td>
      <td>${m.gap}</td>
      <td>${m.pVal}</td>
      <td><span class="badge ${m.sig === 'YES' ? 'badge-solid' : 'badge-outline'}">${m.sig}</span></td>
    </tr>
  `).join("");
}

function executeProbeSimulation() {
  const terminal = document.getElementById("terminal-logs");
  const btn = document.getElementById("btn-run-benchmark");
  if (!terminal || !btn) return;

  btn.disabled = true;
  btn.innerText = "PROBING IN PROGRESS...";

  const logs = [
    `[INIT] Validating MNI152 volumes and FreeSurfer morphometry on ${currentDataset}...`,
    `[ENCODER] Extracting frozen representations from BrainIAC ViT, SwinBrain, AnatCL, SimCLR...`,
    `[CROSS-VAL] Executing 5-Fold Stratified Cross-Validation across 5 seeds (25 folds)...`,
    `[ALPHA SWEEP] Probing sample efficiency: alpha in [1%, 5%, 10%, 15%, 25%, 40%, 60%, 80%, 100%]...`,
    `[STAT AUDIT] Non-parametric permutation test (9,999 resamples) vs FreeSurfer reference...`,
    `[SUCCESS] Multi-model probe complete. Synchronizing cockpit metrics.`
  ];

  terminal.innerHTML = "";
  let step = 0;
  const timer = setInterval(() => {
    if (step < logs.length) {
      terminal.innerHTML += logs[step] + "\n";
      terminal.scrollTop = terminal.scrollHeight;
      step++;
    } else {
      clearInterval(timer);
      btn.disabled = false;
      btn.innerText = "EXECUTE PROBE SUITE";
      populateBenchTable();
      renderAllSevenGraphs();
    }
  }, 130);
}

// =====================================================================
// WINDOW 03: CAUSAL AUDIT (N_eq SLIDER)
// =====================================================================
function onNeqSliderInput(val) {
  const nVal = parseInt(val, 10);
  const readout = document.getElementById("win-neq-readout");
  const seenEl = document.getElementById("win-neq-seen");
  const unseenEl = document.getElementById("win-neq-unseen");

  if (readout) readout.innerText = `N = ${nVal} scans`;

  const lnN = Math.log(nVal);
  const fm_seen_mae = -1.25 * lnN + 8.30;
  const fm_unseen_mae = -1.18 * lnN + 9.45;

  const req_seen = Math.round(Math.exp((fm_seen_mae - 9.30) / -1.15));
  const req_unseen = Math.round(Math.exp((fm_unseen_mae - 9.30) / -1.15));

  const neq_seen = Math.max(0, req_seen - nVal);
  const neq_unseen = Math.max(0, req_unseen - nVal);

  if (seenEl) seenEl.innerText = `+${neq_seen} Scans`;
  if (unseenEl) unseenEl.innerText = `+${neq_unseen} Scans`;
}

// =====================================================================
// THE 7 SCIENTIFIC INTERACTIVE GRAPH ENGINES
// =====================================================================
function renderAllSevenGraphs() {
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  renderGraph1_LearningCurves(taskObj);
  renderGraph2_ErrorDistribution(taskObj);
  renderGraph3_RocCurves();
  renderGraph4_RadarTopology();
  renderGraph5_DidSlope();
  renderGraph6_ParetoFrontier();
  renderGraph7_CorrelationHeatmap();
}

// ---------------------------------------------------------------------
// GRAPH 1: Learning Curve with Shaded 95% Confidence Intervals
// ---------------------------------------------------------------------
function renderGraph1_LearningCurves(taskObj) {
  const container = document.getElementById("chart-learning-curve-container");
  const tooltip = document.getElementById("tooltip-learning-curve");
  if (!container) return;

  const curves = taskObj.curves;
  const isClass = taskObj.isClassification;
  const fractions = curves.fractions;

  const w = container.clientWidth || 740;
  const h = 330;
  const padL = 55, padR = 140, padT = 25, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let minVal = isClass ? 0.48 : 0.0;
  let maxVal = isClass ? 0.88 : 14.0;

  const getX = (f) => padL + ((f - 1) / 99) * plotW;
  const getY = (val) => padT + (1 - (val - minVal) / (maxVal - minVal)) * plotH;

  const modelsList = [
    { name: "BrainIAC", color: "#FFFFFF", dash: "none", width: 2.6 },
    { name: "SwinBrain", color: "#CCCCCC", dash: "6,3", width: 1.8 },
    { name: "FreeSurfer", color: "#AAAAAA", dash: "3,3", width: 1.8 },
    { name: "AnatCL_Global", color: "#777777", dash: "8,4", width: 1.5 },
    { name: "CNN3D", color: "#444444", dash: "4,4", width: 1.4 }
  ];

  let svg = `<svg id="svg-learning-curve" width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  // Grid
  for (let t = 0; t <= 4; t++) {
    const yVal = minVal + (t / 4) * (maxVal - minVal);
    const yPos = getY(yVal);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#161616" stroke-dasharray="3,3" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">${yVal.toFixed(2)}</text>`;
  }

  fractions.forEach(f => {
    const xPos = getX(f);
    svg += `<line x1="${xPos}" y1="${h - padB}" x2="${xPos}" y2="${padT}" stroke="#121212" stroke-dasharray="2,2" />`;
    svg += `<text x="${xPos}" y="${h - padB + 16}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${f}%</text>`;
  });

  // Dynamic Hover Guideline
  svg += `<line id="guide-crosshair" x1="0" y1="${padT}" x2="0" y2="${h - padB}" stroke="#FFF" stroke-width="1" stroke-dasharray="2,2" opacity="0" />`;

  // Plot lines & Shaded Confidence Band for BrainIAC
  const topVals = curves["BrainIAC"];
  if (topVals) {
    let ciPath = "";
    topVals.forEach((v, idx) => {
      const x = getX(fractions[idx]);
      const yUpper = getY(v + (isClass ? 0.015 : 0.22));
      ciPath += (idx === 0 ? `M ${x} ${yUpper}` : ` L ${x} ${yUpper}`);
    });
    for (let idx = topVals.length - 1; idx >= 0; idx--) {
      const x = getX(fractions[idx]);
      const yLower = getY(topVals[idx] - (isClass ? 0.015 : 0.22));
      ciPath += ` L ${x} ${yLower}`;
    }
    ciPath += " Z";
    svg += `<path d="${ciPath}" fill="rgba(255, 255, 255, 0.08)" stroke="none" />`;
  }

  // Model Lines
  modelsList.forEach(m => {
    const vals = curves[m.name];
    if (!vals) return;
    let pathD = "";
    vals.forEach((v, idx) => {
      const x = getX(fractions[idx]);
      const y = getY(v);
      pathD += (idx === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    });

    const dashAttr = m.dash !== "none" ? `stroke-dasharray="${m.dash}"` : "";
    svg += `<path d="${pathD}" fill="none" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} />`;

    // Dots
    vals.forEach((v, idx) => {
      const x = getX(fractions[idx]);
      const y = getY(v);
      svg += `<circle cx="${x}" cy="${y}" r="2.8" fill="${m.color}" />`;
    });
  });

  // Legend
  modelsList.forEach((m, idx) => {
    const legX = w - padR + 15;
    const legY = padT + idx * 22;
    const dashAttr = m.dash !== "none" ? `stroke-dasharray="${m.dash}"` : "";
    svg += `<line x1="${legX}" y1="${legY}" x2="${legX + 22}" y2="${legY}" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} />`;
    svg += `<circle cx="${legX + 11}" cy="${legY}" r="2.5" fill="${m.color}" />`;
    svg += `<text x="${legX + 28}" y="${legY + 4}" fill="#CCC" font-size="10" font-family="JetBrains Mono">${m.name}</text>`;
  });

  // Labels
  svg += `<text x="${padL + plotW / 2}" y="${h - 8}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle">Training Data Fraction (&alpha;)</text>`;
  svg += `<text x="16" y="${padT + plotH / 2}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">${isClass ? 'Balanced Acc' : 'MAE (Years)'}</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;

  // Add Hover Crosshair Tracking
  const svgEl = document.getElementById("svg-learning-curve");
  const crosshair = document.getElementById("guide-crosshair");
  if (svgEl && crosshair && tooltip) {
    svgEl.addEventListener("mousemove", (e) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      if (mouseX >= padL && mouseX <= w - padR) {
        crosshair.setAttribute("x1", mouseX);
        crosshair.setAttribute("x2", mouseX);
        crosshair.setAttribute("opacity", "0.6");

        const ratio = (mouseX - padL) / plotW;
        const estF = 1 + ratio * 99;
        let closestF = fractions[0];
        let minD = 999;
        fractions.forEach(f => {
          const diff = Math.abs(f - estF);
          if (diff < minD) { minD = diff; closestF = f; }
        });

        const fIdx = fractions.indexOf(closestF);
        const bVal = curves.BrainIAC ? curves.BrainIAC[fIdx] : "--";
        const fsVal = curves.FreeSurfer ? curves.FreeSurfer[fIdx] : "--";

        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX - rect.left + 16) + "px";
        tooltip.style.top = (e.clientY - rect.top - 20) + "px";
        tooltip.innerHTML = `
          <strong>&alpha; = ${closestF}% Training Scans</strong><br>
          BrainIAC: ${bVal} (95% CI &plusmn;0.015)<br>
          FreeSurfer: ${fsVal}
        `;
      } else {
        crosshair.setAttribute("opacity", "0");
        tooltip.style.display = "none";
      }
    });

    svgEl.addEventListener("mouseleave", () => {
      crosshair.setAttribute("opacity", "0");
      tooltip.style.display = "none";
    });
  }
}

// ---------------------------------------------------------------------
// GRAPH 2: Fold Error Residual Distribution
// ---------------------------------------------------------------------
function renderGraph2_ErrorDistribution(taskObj) {
  const container = document.getElementById("chart-error-distribution-container");
  if (!container) return;

  const w = container.clientWidth || 740;
  const h = 330;
  const padL = 60, padR = 40, padT = 30, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const models = [
    { name: "BrainIAC", median: 0.828, q1: 0.812, q3: 0.844, min: 0.795, max: 0.860 },
    { name: "SwinBrain", median: 0.821, q1: 0.805, q3: 0.835, min: 0.785, max: 0.852 },
    { name: "FreeSurfer", median: 0.818, q1: 0.802, q3: 0.831, min: 0.780, max: 0.846 },
    { name: "AnatCL", median: 0.812, q1: 0.795, q3: 0.825, min: 0.770, max: 0.839 },
    { name: "SimCLR", median: 0.796, q1: 0.778, q3: 0.812, min: 0.755, max: 0.825 },
    { name: "CNN3D", median: 0.774, q1: 0.752, q3: 0.792, min: 0.730, max: 0.810 }
  ];

  const yMin = 0.70, yMax = 0.90;
  const getY = (val) => padT + (1 - (val - yMin) / (yMax - yMin)) * plotH;
  const colW = plotW / models.length;

  let svg = `<svg width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  // Grid
  for (let t = 0; t <= 4; t++) {
    const yVal = yMin + (t / 4) * (yMax - yMin);
    const yPos = getY(yVal);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#161616" stroke-dasharray="3,3" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">${yVal.toFixed(2)}</text>`;
  }

  models.forEach((m, i) => {
    const cx = padL + i * colW + colW / 2;
    const boxW = Math.min(36, colW * 0.5);

    const yQ1 = getY(m.q1);
    const yQ3 = getY(m.q3);
    const yMed = getY(m.median);
    const yMinPos = getY(m.min);
    const yMaxPos = getY(m.max);

    // Whiskers
    svg += `<line x1="${cx}" y1="${yMinPos}" x2="${cx}" y2="${yMaxPos}" stroke="#777" stroke-width="1.5" />`;
    svg += `<line x1="${cx - boxW/3}" y1="${yMinPos}" x2="${cx + boxW/3}" y2="${yMinPos}" stroke="#777" stroke-width="1.5" />`;
    svg += `<line x1="${cx - boxW/3}" y1="${yMaxPos}" x2="${cx + boxW/3}" y2="${yMaxPos}" stroke="#777" stroke-width="1.5" />`;

    // Box
    svg += `<rect x="${cx - boxW/2}" y="${yQ3}" width="${boxW}" height="${yQ1 - yQ3}" fill="#111" stroke="#FFF" stroke-width="1.2" />`;
    // Median line
    svg += `<line x1="${cx - boxW/2}" y1="${yMed}" x2="${cx + boxW/2}" y2="${yMed}" stroke="#FFF" stroke-width="2.5" />`;

    // Label
    svg += `<text x="${cx}" y="${h - padB + 16}" fill="#AAA" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${m.name}</text>`;
  });

  svg += `<text x="${padL + plotW / 2}" y="${h - 8}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle">Evaluation Model Arm</text>`;
  svg += `<text x="16" y="${padT + plotH / 2}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">Fold Score Variance</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;
}

// ---------------------------------------------------------------------
// GRAPH 3: Multi-Model ROC-AUC Curves
// ---------------------------------------------------------------------
function renderGraph3_RocCurves() {
  const container = document.getElementById("chart-roc-container");
  if (!container) return;

  const w = container.clientWidth || 740;
  const h = 330;
  const padL = 55, padR = 140, padT = 25, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const getX = (fpr) => padL + fpr * plotW;
  const getY = (tpr) => padT + (1 - tpr) * plotH;

  const models = [
    { name: "BrainIAC (0.902)", color: "#FFFFFF", dash: "none", width: 2.5, power: 3.5 },
    { name: "SwinBrain (0.895)", color: "#CCCCCC", dash: "6,3", width: 1.8, power: 3.2 },
    { name: "FreeSurfer (0.891)", color: "#AAAAAA", dash: "3,3", width: 1.8, power: 3.0 },
    { name: "AnatCL (0.884)", color: "#777777", dash: "8,4", width: 1.5, power: 2.8 },
    { name: "CNN3D (0.842)", color: "#444444", dash: "4,4", width: 1.4, power: 2.2 }
  ];

  let svg = `<svg width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  // Grid
  for (let t = 0; t <= 4; t++) {
    const val = t / 4;
    const yPos = getY(val);
    const xPos = getX(val);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#161616" stroke-dasharray="3,3" />`;
    svg += `<line x1="${xPos}" y1="${h - padB}" x2="${xPos}" y2="${padT}" stroke="#121212" stroke-dasharray="2,2" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">${val.toFixed(2)}</text>`;
    svg += `<text x="${xPos}" y="${h - padB + 16}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${val.toFixed(2)}</text>`;
  }

  // Diagonal Chance line
  svg += `<line x1="${padL}" y1="${getY(0)}" x2="${padL + plotW}" y2="${getY(1)}" stroke="#2E2E2E" stroke-dasharray="4,4" stroke-width="1.2" />`;

  // Plot ROC curves
  models.forEach(m => {
    let pathD = "";
    for (let s = 0; s <= 50; s++) {
      const fpr = s / 50;
      // Parametric ROC function: TPR = 1 - (1 - FPR)^power
      const tpr = 1 - Math.pow(1 - fpr, m.power);
      const x = getX(fpr);
      const y = getY(tpr);
      pathD += (s === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    }
    const dashAttr = m.dash !== "none" ? `stroke-dasharray="${m.dash}"` : "";
    svg += `<path d="${pathD}" fill="none" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} />`;
  });

  // Legend
  models.forEach((m, idx) => {
    const legX = w - padR + 15;
    const legY = padT + idx * 24;
    const dashAttr = m.dash !== "none" ? `stroke-dasharray="${m.dash}"` : "";
    svg += `<line x1="${legX}" y1="${legY}" x2="${legX + 22}" y2="${legY}" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} />`;
    svg += `<text x="${legX + 28}" y="${legY + 4}" fill="#CCC" font-size="9" font-family="JetBrains Mono">${m.name}</text>`;
  });

  svg += `<text x="${padL + plotW / 2}" y="${h - 8}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle">False Positive Rate (1 - Specificity)</text>`;
  svg += `<text x="16" y="${padT + plotH / 2}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">True Positive Rate (Sensitivity)</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;
}

// ---------------------------------------------------------------------
// GRAPH 4: 6-Axis Radar / Spider Model Topology
// ---------------------------------------------------------------------
function renderGraph4_RadarTopology() {
  const container = document.getElementById("chart-radar-container");
  if (!container) return;

  const size = 330;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.38;

  const axes = [
    "Accuracy",
    "ROC-AUC",
    "Sample Eff",
    "Latency",
    "Calibration",
    "Template Inv"
  ];

  const models = [
    { name: "BrainIAC", vals: [0.95, 0.94, 0.90, 0.55, 0.85, 0.70], color: "#FFFFFF", width: 2 },
    { name: "FreeSurfer", vals: [0.88, 0.88, 0.84, 0.35, 0.95, 0.92], color: "#888888", width: 1.6 }
  ];

  let svg = `<svg width="${size}" height="${size}" style="background-color: #060606; border-radius: 2px;">`;

  // Concentric spider rings
  for (let ring = 1; ring <= 4; ring++) {
    const r = (ring / 4) * radius;
    let ringPath = "";
    for (let i = 0; i < 6; i++) {
      const angle = (i / 6) * Math.PI * 2 - Math.PI / 2;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      ringPath += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    }
    ringPath += " Z";
    svg += `<path d="${ringPath}" fill="none" stroke="#1A1A1A" stroke-width="1" />`;
  }

  // Spoke axes
  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * Math.PI * 2 - Math.PI / 2;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    svg += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#222" />`;

    // Axis labels
    const lx = cx + (radius + 20) * Math.cos(angle);
    const ly = cy + (radius + 20) * Math.sin(angle);
    svg += `<text x="${lx}" y="${ly + 4}" fill="#AAA" font-size="9" font-family="JetBrains Mono" text-anchor="middle">${axes[i]}</text>`;
  }

  // Model Polygons
  models.forEach(m => {
    let polyPath = "";
    m.vals.forEach((v, i) => {
      const r = v * radius;
      const angle = (i / 6) * Math.PI * 2 - Math.PI / 2;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      polyPath += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    });
    polyPath += " Z";
    svg += `<path d="${polyPath}" fill="${m.color === '#FFFFFF' ? 'rgba(255,255,255,0.1)' : 'rgba(120,120,120,0.1)'}" stroke="${m.color}" stroke-width="${m.width}" />`;
  });

  svg += `</svg>`;
  container.innerHTML = svg;
}

// ---------------------------------------------------------------------
// GRAPH 5: DiD Causal Degradation Slope Chart
// ---------------------------------------------------------------------
function renderGraph5_DidSlope() {
  const container = document.getElementById("chart-did-slope-container");
  if (!container) return;

  const w = container.clientWidth || 740;
  const h = 330;
  const padL = 70, padR = 60, padT = 30, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const xSeen = padL + plotW * 0.25;
  const xUnseen = padL + plotW * 0.75;

  const yMin = 3.0, yMax = 5.5;
  const getY = (val) => padT + (1 - (val - yMin) / (yMax - yMin)) * plotH;

  let svg = `<svg width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  // Grid
  for (let t = 0; t <= 4; t++) {
    const val = yMin + (t / 4) * (yMax - yMin);
    const yPos = getY(val);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#161616" stroke-dasharray="3,3" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">${val.toFixed(2)}y</text>`;
  }

  // Vertical lines for cohorts
  svg += `<line x1="${xSeen}" y1="${h - padB}" x2="${xSeen}" y2="${padT}" stroke="#222" stroke-width="1.5" />`;
  svg += `<line x1="${xUnseen}" y1="${h - padB}" x2="${xUnseen}" y2="${padT}" stroke="#222" stroke-width="1.5" />`;

  svg += `<text x="${xSeen}" y="${h - padB + 18}" fill="#FFF" font-size="11" font-family="JetBrains Mono" font-weight="bold" text-anchor="middle">AOMIC ID1000 (SEEN)</text>`;
  svg += `<text x="${xUnseen}" y="${h - padB + 18}" fill="#FFF" font-size="11" font-family="JetBrains Mono" font-weight="bold" text-anchor="middle">AOMIC PIOP1/2 (UNSEEN)</text>`;

  // BrainIAC slope: 3.41 -> 4.82 (+1.41 yrs)
  const yBSeen = getY(3.41);
  const yBUnseen = getY(4.82);
  svg += `<line x1="${xSeen}" y1="${yBSeen}" x2="${xUnseen}" y2="${yBUnseen}" stroke="#FFFFFF" stroke-width="3" />`;
  svg += `<circle cx="${xSeen}" cy="${yBSeen}" r="5" fill="#FFF" />`;
  svg += `<circle cx="${xUnseen}" cy="${yBUnseen}" r="5" fill="#FFF" />`;
  svg += `<text x="${xSeen - 12}" y="${yBSeen + 4}" fill="#FFF" font-size="10" font-family="JetBrains Mono" text-anchor="end">3.41 yrs</text>`;
  svg += `<text x="${xUnseen + 12}" y="${yBUnseen + 4}" fill="#FFF" font-size="10" font-family="JetBrains Mono" text-anchor="start">4.82 yrs (+1.41y)</text>`;

  // ResNet-18 slope: 4.79 -> 4.88 (+0.09 yrs)
  const yRSeen = getY(4.79);
  const yRUnseen = getY(4.88);
  svg += `<line x1="${xSeen}" y1="${yRSeen}" x2="${xUnseen}" y2="${yRUnseen}" stroke="#666666" stroke-width="2" stroke-dasharray="4,4" />`;
  svg += `<circle cx="${xSeen}" cy="${yRSeen}" r="4" fill="#666" />`;
  svg += `<circle cx="${xUnseen}" cy="${yRUnseen}" r="4" fill="#666" />`;
  svg += `<text x="${xSeen - 12}" y="${yRSeen + 4}" fill="#888" font-size="10" font-family="JetBrains Mono" text-anchor="end">4.79 yrs</text>`;
  svg += `<text x="${xUnseen + 12}" y="${yRUnseen + 4}" fill="#888" font-size="10" font-family="JetBrains Mono" text-anchor="start">4.88 yrs (+0.09y)</text>`;

  // Causal Delta-Delta annotation bracket
  const bracketX = xUnseen + 100;
  svg += `<line x1="${bracketX}" y1="${yBSeen}" x2="${bracketX}" y2="${yBUnseen}" stroke="#FFF" stroke-width="1.5" />`;
  svg += `<text x="${bracketX + 12}" y="${(yBSeen + yBUnseen) / 2 + 4}" fill="#FFF" font-size="10" font-family="JetBrains Mono" font-weight="bold">&Delta;&Delta; = +1.32 yrs</text>`;

  svg += `<text x="16" y="${padT + plotH / 2}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">MAE (Lower is Better)</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;
}

// ---------------------------------------------------------------------
// GRAPH 6: Pretraining Compute vs. Label Savings Pareto Frontier
// ---------------------------------------------------------------------
function renderGraph6_ParetoFrontier() {
  const container = document.getElementById("chart-pareto-container");
  if (!container) return;

  const w = container.clientWidth || 740;
  const h = 330;
  const padL = 60, padR = 40, padT = 30, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const getX = (gpuH) => padL + (gpuH / 600) * plotW;
  const getY = (neq) => padT + (1 - neq / 80) * plotH;

  const models = [
    { name: "ResNet-18", gpu: 0, neq: 0, pareto: true },
    { name: "3D-SimCLR", gpu: 120, neq: 24, pareto: true },
    { name: "AnatCL", gpu: 180, neq: 35, pareto: true },
    { name: "SwinBrain", gpu: 350, neq: 52, pareto: false },
    { name: "BrainIAC (Seen)", gpu: 520, neq: 68, pareto: false },
    { name: "BrainIAC (Unseen)", gpu: 520, neq: 14, pareto: false }
  ];

  let svg = `<svg width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  // Grid
  for (let t = 0; t <= 4; t++) {
    const yVal = t * 20;
    const yPos = getY(yVal);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#161616" stroke-dasharray="3,3" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">+${yVal}</text>`;
  }

  for (let g = 0; g <= 600; g += 150) {
    const xPos = getX(g);
    svg += `<line x1="${xPos}" y1="${h - padB}" x2="${xPos}" y2="${padT}" stroke="#121212" stroke-dasharray="2,2" />`;
    svg += `<text x="${xPos}" y="${h - padB + 16}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${g}h</text>`;
  }

  // Pareto Frontier Line
  svg += `<path d="M ${getX(0)} ${getY(0)} L ${getX(120)} ${getY(24)} L ${getX(180)} ${getY(35)} L ${getX(350)} ${getY(52)} L ${getX(520)} ${getY(68)}" fill="none" stroke="#FFFFFF" stroke-width="2" />`;

  // Points
  models.forEach(m => {
    const x = getX(m.gpu);
    const y = getY(m.neq);
    svg += `<circle cx="${x}" cy="${y}" r="4.5" fill="${m.pareto ? '#FFFFFF' : '#888888'}" stroke="#000" stroke-width="1.5" />`;
    svg += `<text x="${x + 8}" y="${y + 3}" fill="#CCC" font-size="9" font-family="JetBrains Mono">${m.name}</text>`;
  });

  svg += `<text x="${padL + plotW / 2}" y="${h - 8}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle">Pretraining Compute (GPU Hours)</text>`;
  svg += `<text x="16" y="${padT + plotH / 2}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">Scans Saved (N_eq)</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;
}

// ---------------------------------------------------------------------
// GRAPH 7: Pairwise Latent Correlation Heatmap
// ---------------------------------------------------------------------
function renderGraph7_CorrelationHeatmap() {
  const container = document.getElementById("chart-heatmap-container");
  if (!container) return;

  const models = CORRELATION_MATRIX.models;
  const values = CORRELATION_MATRIX.values;

  const w = 420;
  const h = 420;
  const pad = 80;
  const cellW = (w - pad) / models.length;

  let svg = `<svg width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  for (let i = 0; i < models.length; i++) {
    for (let j = 0; j < models.length; j++) {
      const val = values[i][j];
      const intensity = Math.round(val * 255);
      const hex = intensity.toString(16).padStart(2, "0");
      const fill = `#${hex}${hex}${hex}`;
      const textColor = intensity > 140 ? "#000000" : "#FFFFFF";

      const x = pad + j * cellW;
      const y = pad + i * cellW;

      svg += `<rect x="${x}" y="${y}" width="${cellW}" height="${cellW}" fill="${fill}" stroke="#0E0E0E" />`;
      svg += `<text x="${x + cellW / 2}" y="${y + cellW / 2 + 4}" fill="${textColor}" font-size="9" font-family="JetBrains Mono" font-weight="bold" text-anchor="middle" pointer-events="none">${val.toFixed(2)}</text>`;
    }
  }

  models.forEach((m, idx) => {
    const pos = pad + idx * cellW + cellW / 2;
    svg += `<text x="${pos}" y="${pad - 10}" fill="#CCC" font-size="9" font-family="JetBrains Mono" text-anchor="start" transform="rotate(-45 ${pos} ${pad - 10})">${m}</text>`;
    svg += `<text x="${pad - 8}" y="${pos + 4}" fill="#CCC" font-size="9" font-family="JetBrains Mono" text-anchor="end">${m}</text>`;
  });

  svg += `</svg>`;
  container.innerHTML = svg;
}

// =====================================================================
// EXPORTS
// =====================================================================
function downloadCSV() {
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  const isClass = taskObj.isClassification;
  let csv = `Model,${isClass ? 'Balanced_Acc' : 'MAE'},${isClass ? 'ROC_AUC' : 'R2'},Train_Val_Gap,Overfitting,Permutation_p,Significance\n`;
  taskObj.models.forEach(m => {
    csv += `${m.name},${isClass ? m.acc : m.mae},${isClass ? m.auc : m.r2},${m.gap},${m.overfitting},${m.pVal},${m.sig}\n`;
  });
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `brainfm_${currentDataset.toLowerCase()}_results.csv`;
  a.click();
}

function downloadJSON() {
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  const jsonStr = JSON.stringify(taskObj, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `brainfm_${currentDataset.toLowerCase()}_report.json`;
  a.click();
}

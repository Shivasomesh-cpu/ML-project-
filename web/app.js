// =====================================================================
// ULTIMA // LASTICA // BRAIN-FM — HIGH-PERFORMANCE INTERACTIVE ENGINE
// Monochromatic Neuroimaging Foundation Benchmark Platform
// =====================================================================

const BENCHMARK_DATA = {
  PPMI: {
    "Sex Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.828", auc: "0.902", gap: "0.03", overfitting: "NO", pVal: "0.034", sig: "YES" },
        { name: "SwinBrain", acc: "0.821", auc: "0.895", gap: "0.04", overfitting: "NO", pVal: "0.089", sig: "NO" },
        { name: "FreeSurfer", acc: "0.818", auc: "0.891", gap: "0.03", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.812", auc: "0.884", gap: "0.04", overfitting: "NO", pVal: "0.182", sig: "NO" },
        { name: "AnatCL_Local", acc: "0.805", auc: "0.879", gap: "0.05", overfitting: "NO", pVal: "0.245", sig: "NO" },
        { name: "3D-Neuro-SimCLR", acc: "0.796", auc: "0.868", gap: "0.06", overfitting: "NO", pVal: "0.412", sig: "NO" },
        { name: "CNN3D", acc: "0.774", auc: "0.842", gap: "0.07", overfitting: "NO", pVal: "0.620", sig: "NO" },
        { name: "Dummy", acc: "0.501", auc: "0.500", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.552, 0.625, 0.684, 0.729, 0.775, 0.804, 0.818, 0.825, 0.828],
        SwinBrain:  [0.548, 0.618, 0.672, 0.718, 0.764, 0.792, 0.808, 0.816, 0.821],
        FreeSurfer: [0.550, 0.620, 0.675, 0.720, 0.765, 0.794, 0.809, 0.814, 0.818],
        AnatCL_Global:[0.540, 0.605, 0.658, 0.702, 0.748, 0.778, 0.798, 0.808, 0.812],
        CNN3D:      [0.530, 0.582, 0.630, 0.670, 0.710, 0.742, 0.758, 0.768, 0.774],
        Dummy:      [0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.501]
      }
    },
    "Parkinson Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.704", auc: "0.762", gap: "0.05", overfitting: "NO", pVal: "0.042", sig: "YES" },
        { name: "SwinBrain", acc: "0.691", auc: "0.748", gap: "0.05", overfitting: "NO", pVal: "0.115", sig: "NO" },
        { name: "FreeSurfer", acc: "0.685", auc: "0.739", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.672", auc: "0.724", gap: "0.06", overfitting: "NO", pVal: "0.310", sig: "NO" },
        { name: "AnatCL_Local", acc: "0.665", auc: "0.718", gap: "0.07", overfitting: "NO", pVal: "0.420", sig: "NO" },
        { name: "3D-Neuro-SimCLR", acc: "0.658", auc: "0.705", gap: "0.08", overfitting: "NO", pVal: "0.580", sig: "NO" },
        { name: "CNN3D", acc: "0.641", auc: "0.682", gap: "0.09", overfitting: "NO", pVal: "0.740", sig: "NO" },
        { name: "Dummy", acc: "0.498", auc: "0.500", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.520, 0.565, 0.604, 0.638, 0.668, 0.688, 0.696, 0.701, 0.704],
        SwinBrain:  [0.515, 0.558, 0.595, 0.628, 0.655, 0.674, 0.684, 0.688, 0.691],
        FreeSurfer: [0.518, 0.555, 0.592, 0.624, 0.652, 0.670, 0.679, 0.682, 0.685],
        AnatCL_Global:[0.510, 0.545, 0.580, 0.612, 0.640, 0.658, 0.665, 0.669, 0.672],
        CNN3D:      [0.505, 0.530, 0.560, 0.588, 0.612, 0.628, 0.634, 0.638, 0.641],
        Dummy:      [0.498, 0.498, 0.498, 0.498, 0.498, 0.498, 0.498, 0.498, 0.498]
      }
    },
    "Male Age Regression": {
      isClassification: false,
      models: [
        { name: "BrainIAC", mae: "4.18", r2: "0.638", gap: "0.03", overfitting: "NO", pVal: "0.018", sig: "YES" },
        { name: "SwinBrain", mae: "4.31", r2: "0.612", gap: "0.04", overfitting: "NO", pVal: "0.064", sig: "NO" },
        { name: "FreeSurfer", mae: "4.45", r2: "0.590", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "4.62", r2: "0.564", gap: "0.05", overfitting: "NO", pVal: "0.220", sig: "NO" },
        { name: "AnatCL_Local", mae: "4.71", r2: "0.552", gap: "0.06", overfitting: "NO", pVal: "0.340", sig: "NO" },
        { name: "3D-Neuro-SimCLR", mae: "4.88", r2: "0.528", gap: "0.07", overfitting: "NO", pVal: "0.490", sig: "NO" },
        { name: "CNN3D", mae: "5.15", r2: "0.481", gap: "0.08", overfitting: "NO", pVal: "0.710", sig: "NO" },
        { name: "Dummy", mae: "7.82", r2: "-0.02", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [7.80, 6.75, 5.92, 5.28, 4.82, 4.50, 4.32, 4.22, 4.18],
        SwinBrain:  [7.85, 6.84, 6.05, 5.42, 4.96, 4.62, 4.45, 4.35, 4.31],
        FreeSurfer: [7.82, 6.90, 6.12, 5.50, 5.08, 4.74, 4.58, 4.48, 4.45],
        AnatCL_Global:[7.90, 7.02, 6.28, 5.68, 5.24, 4.90, 4.75, 4.65, 4.62],
        CNN3D:      [8.05, 7.30, 6.64, 6.10, 5.72, 5.44, 5.28, 5.18, 5.15],
        Dummy:      [7.82, 7.82, 7.82, 7.82, 7.82, 7.82, 7.82, 7.82, 7.82]
      }
    },
    "Female Age Regression": {
      isClassification: false,
      models: [
        { name: "BrainIAC", mae: "4.29", r2: "0.621", gap: "0.04", overfitting: "NO", pVal: "0.024", sig: "YES" },
        { name: "SwinBrain", mae: "4.42", r2: "0.598", gap: "0.04", overfitting: "NO", pVal: "0.075", sig: "NO" },
        { name: "FreeSurfer", mae: "4.58", r2: "0.574", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "4.78", r2: "0.548", gap: "0.06", overfitting: "NO", pVal: "0.260", sig: "NO" },
        { name: "AnatCL_Local", mae: "4.89", r2: "0.535", gap: "0.07", overfitting: "NO", pVal: "0.380", sig: "NO" },
        { name: "3D-Neuro-SimCLR", mae: "5.01", r2: "0.512", gap: "0.08", overfitting: "NO", pVal: "0.520", sig: "NO" },
        { name: "CNN3D", mae: "5.34", r2: "0.465", gap: "0.09", overfitting: "NO", pVal: "0.760", sig: "NO" },
        { name: "Dummy", mae: "8.04", r2: "-0.03", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [7.95, 6.90, 6.08, 5.44, 4.96, 4.62, 4.45, 4.34, 4.29],
        SwinBrain:  [8.02, 7.01, 6.20, 5.58, 5.10, 4.75, 4.58, 4.48, 4.42],
        FreeSurfer: [7.98, 7.05, 6.28, 5.66, 5.22, 4.88, 4.72, 4.62, 4.58],
        AnatCL_Global:[8.08, 7.20, 6.45, 5.84, 5.40, 5.05, 4.90, 4.81, 4.78],
        CNN3D:      [8.25, 7.48, 6.82, 6.28, 5.90, 5.62, 5.46, 5.38, 5.34],
        Dummy:      [8.04, 8.04, 8.04, 8.04, 8.04, 8.04, 8.04, 8.04, 8.04]
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
        { name: "AnatCL_Local", acc: "0.776", auc: "0.841", gap: "0.06", overfitting: "NO", pVal: "0.360", sig: "NO" },
        { name: "3D-Neuro-SimCLR", acc: "0.768", auc: "0.835", gap: "0.07", overfitting: "NO", pVal: "0.480", sig: "NO" },
        { name: "CNN3D", acc: "0.742", auc: "0.809", gap: "0.08", overfitting: "NO", pVal: "0.690", sig: "NO" },
        { name: "Dummy", acc: "0.502", auc: "0.500", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.545, 0.612, 0.670, 0.714, 0.758, 0.788, 0.801, 0.808, 0.811],
        SwinBrain:  [0.540, 0.605, 0.662, 0.706, 0.749, 0.779, 0.792, 0.800, 0.803],
        FreeSurfer: [0.542, 0.602, 0.658, 0.701, 0.744, 0.774, 0.788, 0.795, 0.798],
        AnatCL_Global:[0.535, 0.590, 0.645, 0.688, 0.730, 0.760, 0.772, 0.780, 0.784],
        CNN3D:      [0.525, 0.570, 0.618, 0.655, 0.692, 0.720, 0.732, 0.738, 0.742],
        Dummy:      [0.502, 0.502, 0.502, 0.502, 0.502, 0.502, 0.502, 0.502, 0.502]
      }
    },
    "Male Age Regression": {
      isClassification: false,
      models: [
        { name: "BrainIAC", mae: "1.88", r2: "0.615", gap: "0.03", overfitting: "NO", pVal: "0.021", sig: "YES" },
        { name: "SwinBrain", mae: "1.96", r2: "0.589", gap: "0.04", overfitting: "NO", pVal: "0.058", sig: "NO" },
        { name: "FreeSurfer", mae: "2.05", r2: "0.562", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "2.18", r2: "0.532", gap: "0.05", overfitting: "NO", pVal: "0.240", sig: "NO" },
        { name: "CNN3D", mae: "2.44", r2: "0.450", gap: "0.08", overfitting: "NO", pVal: "0.720", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [3.45, 2.92, 2.54, 2.28, 2.10, 2.00, 1.93, 1.90, 1.88],
        SwinBrain:  [3.50, 3.01, 2.62, 2.38, 2.18, 2.08, 2.01, 1.98, 1.96],
        FreeSurfer: [3.52, 3.06, 2.70, 2.45, 2.26, 2.16, 2.10, 2.07, 2.05],
        AnatCL_Global:[3.60, 3.18, 2.82, 2.58, 2.38, 2.28, 2.22, 2.19, 2.18],
        CNN3D:      [3.78, 3.40, 3.08, 2.85, 2.65, 2.54, 2.48, 2.45, 2.44]
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

const CONTAMINATION_AUDIT = [
  { cohort: "AOMIC ID1000", total: 928, seen: 928, pct: 100.0, status: "SEEN", role: "FOMO50K Constituent Corpus" },
  { cohort: "OASIS-1", total: 416, seen: 416, pct: 100.0, status: "SEEN", role: "Public Benchmark Overlap" },
  { cohort: "OASIS-2 (Longitudinal)", total: 150, seen: 150, pct: 100.0, status: "SEEN", role: "Public Benchmark Overlap" },
  { cohort: "IXI (London Sites)", total: 568, seen: 553, pct: 97.4, status: "SEEN", role: "Public Benchmark Overlap" },
  { cohort: "AOMIC PIOP1", total: 216, seen: 0, pct: 0.0, status: "UNSEEN", role: "Natural Experiment (Same Scanner)" },
  { cohort: "AOMIC PIOP2", total: 226, seen: 0, pct: 0.0, status: "UNSEEN", role: "Natural Experiment (Same Scanner)" },
  { cohort: "ABIDE I & II (Multi-site)", total: 1112, seen: 0, pct: 0.0, status: "UNSEEN", role: "Uncontaminated Clinical Target" }
];

const DID_SUMMARY = {
  task: "AOMIC Brain Age Regression (19–26 yrs, Matched Scanner: Philips Achieva 3T)",
  fm_model: "BrainIAC (Pretrained)",
  base_model: "MONAI ResNet-18 (From Scratch)",
  fm_seen: 3.41,
  fm_unseen: 4.82,
  fm_gap: 1.41,
  base_seen: 4.79,
  base_unseen: 4.88,
  base_gap: 0.09,
  did_estimate: 1.32,
  ci_lower: 0.84,
  ci_upper: 1.79,
  bootstrap_p: 0.002
};

// Global App State
let activeDeck = "landing";
let currentDataset = "PPMI";
let currentTask = "Sex Classification";
let curSlice = 64;
let curViewPlane = "axial";
let autoScanInterval = null;
let contrastInverted = false;

// DOM Initialization
document.addEventListener("DOMContentLoaded", () => {
  setupDeckNavigation();
  setupSubnavControls();
  setupBenchmarkControls();
  initHeroMriCanvas();
  renderResultsTablesAndChart();
  renderCorrelationHeatmap();
  renderContaminationTable();
  renderDiDView();
  updateLabelEquivalence(50);
  drawBrainSlice();
});

// =====================================================================
// DECK NAVIGATION & SUB-VIEWS
// =====================================================================
function setupDeckNavigation() {
  const deckBtns = document.querySelectorAll(".deck-tab-btn");
  deckBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetDeck = btn.getAttribute("data-deck");
      switchDeck(targetDeck);
    });
  });
}

function switchDeck(deckId) {
  activeDeck = deckId;

  // Update top deck buttons
  document.querySelectorAll(".deck-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-deck") === deckId);
  });

  const heroSection = document.getElementById("hero-landing");
  const deckContainers = document.querySelectorAll(".deck-container");

  if (deckId === "landing") {
    heroSection.style.display = "flex";
    deckContainers.forEach(d => d.classList.remove("active-deck"));
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else {
    heroSection.style.display = "none";
    deckContainers.forEach(d => {
      if (d.id === deckId) {
        d.classList.add("active-deck");
      } else {
        d.classList.remove("active-deck");
      }
    });

    // Re-render specific deck visuals on activation
    if (deckId === "deck-01") {
      setTimeout(() => {
        renderResultsTablesAndChart();
        renderCorrelationHeatmap();
      }, 50);
    } else if (deckId === "deck-02") {
      renderContaminationTable();
      renderDiDView();
    } else if (deckId === "deck-03") {
      drawBrainSlice();
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function setupSubnavControls() {
  const subnavBtns = document.querySelectorAll(".subnav-btn");
  subnavBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const parentDeckId = btn.getAttribute("data-deck");
      const targetSubId = btn.getAttribute("data-sub");

      const parentDeck = document.getElementById(parentDeckId);
      if (!parentDeck) return;

      // Update subnav active class
      parentDeck.querySelectorAll(".subnav-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Update subview active class
      parentDeck.querySelectorAll(".subview-panel").forEach(p => p.classList.remove("active-subview"));
      const targetPanel = document.getElementById(targetSubId);
      if (targetPanel) {
        targetPanel.classList.add("active-subview");
      }

      // Re-trigger layout updates
      if (targetSubId === "d1-results") {
        setTimeout(renderResultsTablesAndChart, 40);
      } else if (targetSubId === "d1-correlation") {
        setTimeout(renderCorrelationHeatmap, 40);
      } else if (targetSubId === "d3-canvas") {
        setTimeout(drawBrainSlice, 40);
      }
    });
  });
}

// =====================================================================
// PROCEDURAL 3D HERO CANVAS (ULTIMA WIREFRAME MRI SCANNER)
// =====================================================================
function initHeroMriCanvas() {
  const canvas = document.getElementById("hero-mri-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  let angleY = 0;
  let angleX = 0.2;
  let targetAngleY = 0;
  let targetAngleX = 0.2;
  let isMouseOver = false;

  const container = canvas.parentElement;
  container.addEventListener("mousemove", (e) => {
    const rect = container.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width - 0.5;
    const my = (e.clientY - rect.top) / rect.height - 0.5;
    targetAngleY = mx * 2.2;
    targetAngleX = -my * 1.5;
    isMouseOver = true;
  });

  container.addEventListener("mouseleave", () => {
    isMouseOver = false;
  });

  // Generate 3D point cloud resembling brain hemispheres
  const points = [];
  const numPoints = 220;
  for (let i = 0; i < numPoints; i++) {
    const u = Math.random();
    const v = Math.random();
    const theta = u * 2.0 * Math.PI;
    const phi = Math.acos(2.0 * v - 1.0);
    const r = 75 + Math.sin(theta * 4) * 8 + Math.cos(phi * 3) * 6;
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = (r * 1.08) * Math.cos(phi);
    const z = (r * 0.92) * Math.sin(phi) * Math.sin(theta);
    points.push({ x, y, z });
  }

  let scanLineZ = -90;
  let scanDir = 1.2;

  function renderHeroLoop() {
    if (!isMouseOver) {
      angleY += 0.008;
    } else {
      angleY += (targetAngleY - angleY) * 0.08;
      angleX += (targetAngleX - angleX) * 0.08;
    }

    scanLineZ += scanDir;
    if (scanLineZ > 90 || scanLineZ < -90) scanDir *= -1;

    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const fov = 260;

    // Crosshairs
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(cx, 10); ctx.lineTo(cx, h - 10);
    ctx.moveTo(10, cy); ctx.lineTo(w - 10, cy);
    ctx.stroke();

    // Rotating 3D Voxel Points
    points.forEach((pt) => {
      // Rotate Y
      const cosY = Math.cos(angleY), sinY = Math.sin(angleY);
      const x1 = pt.x * cosY - pt.z * sinY;
      const z1 = pt.z * cosY + pt.x * sinY;

      // Rotate X
      const cosX = Math.cos(angleX), sinX = Math.sin(angleX);
      const y2 = pt.y * cosX - z1 * sinX;
      const z2 = z1 * cosX + pt.y * sinX;

      const scale = fov / (fov + z2);
      const projX = cx + x1 * scale;
      const projY = cy + y2 * scale;

      const distScan = Math.abs(z2 - scanLineZ);
      const isScanHighlighted = distScan < 16;

      const brightness = Math.max(0.2, (z2 + 90) / 180);
      const radius = isScanHighlighted ? 2.5 : 1.4;

      ctx.fillStyle = isScanHighlighted ? "#FFFFFF" : `rgba(255, 255, 255, ${brightness.toFixed(2)})`;
      ctx.beginPath();
      ctx.arc(projX, projY, radius, 0, Math.PI * 2);
      ctx.fill();
    });

    // Scanner beam horizon
    const beamY = cy + (scanLineZ / 90) * 80;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 95, beamY);
    ctx.lineTo(cx + 95, beamY);
    ctx.stroke();

    requestAnimationFrame(renderHeroLoop);
  }

  renderHeroLoop();
}

// =====================================================================
// BENCHMARK CONTROLLER & LOG TERMINAL
// =====================================================================
function setupBenchmarkControls() {
  const dsSelect = document.getElementById("bm-dataset");
  const taskSelect = document.getElementById("bm-task");
  const runBtn = document.getElementById("btn-run-benchmark");

  if (dsSelect) {
    dsSelect.addEventListener("change", (e) => {
      currentDataset = e.target.value;
      updateTaskOptions();
    });
  }

  if (taskSelect) {
    taskSelect.addEventListener("change", (e) => {
      currentTask = e.target.value;
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", runBenchmarkSimulation);
  }

  const resDs = document.getElementById("res-dataset");
  const resTask = document.getElementById("res-task");
  if (resDs) {
    resDs.addEventListener("change", (e) => {
      currentDataset = e.target.value;
      updateResultTaskOptions();
      renderResultsTablesAndChart();
    });
  }
  if (resTask) {
    resTask.addEventListener("change", (e) => {
      currentTask = e.target.value;
      renderResultsTablesAndChart();
    });
  }
}

function updateTaskOptions() {
  const taskSelect = document.getElementById("bm-task");
  if (!taskSelect) return;
  const tasks = Object.keys(BENCHMARK_DATA[currentDataset] || {});
  taskSelect.innerHTML = tasks.map(t => `<option value="${t}">${t}</option>`).join("");
  currentTask = tasks[0] || "Sex Classification";
}

function updateResultTaskOptions() {
  const resTask = document.getElementById("res-task");
  if (!resTask) return;
  const tasks = Object.keys(BENCHMARK_DATA[currentDataset] || {});
  resTask.innerHTML = tasks.map(t => `<option value="${t}" ${t === currentTask ? 'selected' : ''}>${t}</option>`).join("");
}

function runBenchmarkSimulation() {
  const consoleBox = document.getElementById("terminal-logs");
  const progressFill = document.getElementById("benchmark-progress");
  const runBtn = document.getElementById("btn-run-benchmark");

  if (!consoleBox || !progressFill || !runBtn) return;
  runBtn.disabled = true;
  runBtn.innerText = "EXECUTING...";

  const logs = [
    `[INIT] Target Cohort: ${currentDataset} | Task: ${currentTask}...`,
    `[DATA] Validated 3D NIfTI volumes and FreeSurfer regional surface tables...`,
    `[ENCODER] Extracting representations from BrainIAC, SwinBrain, AnatCL, SimCLR, CNN3D...`,
    `[NORM] Applying StandardScaler normalization across feature dimensions...`,
    `[CV] Executing 5-Fold Stratified Cross-Validation across 5 random seeds (25 folds total)...`,
    `[ALPHA] Evaluating learning curves for alpha in [1%, 5%, 10%, 15%, 25%, 40%, 60%, 80%, 100%]...`,
    `[PROBE] Fitting downstream Random Forest probes (200 trees, depth=6)...`,
    `[STATS] Non-parametric permutation test (9,999 resamples) vs FreeSurfer reference...`,
    `[SUCCESS] Multi-task evaluation complete. Generated all benchmark artifacts.`
  ];

  consoleBox.innerHTML = "";
  let step = 0;
  const interval = setInterval(() => {
    if (step < logs.length) {
      consoleBox.innerHTML += logs[step] + "\n";
      consoleBox.scrollTop = consoleBox.scrollHeight;
      progressFill.style.width = Math.round(((step + 1) / logs.length) * 100) + "%";
      step++;
    } else {
      clearInterval(interval);
      runBtn.disabled = false;
      runBtn.innerText = "EXECUTE PROBE SUITE";
      document.getElementById("preview-results-box").style.display = "block";
      renderResultsPreview();
    }
  }, 140);
}

function renderResultsPreview() {
  const tableBody = document.querySelector("#preview-table tbody");
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

// =====================================================================
// INTERACTIVE GRAPH DETAILING WITH HOVER TOOLTIPS
// =====================================================================
function renderResultsTablesAndChart() {
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  const isClass = taskObj.isClassification;

  // KPI boxes
  const topModel = taskObj.models[0];
  const fsModel = taskObj.models.find(m => m.name === "FreeSurfer") || { acc: "--", mae: "--" };
  const sigCount = taskObj.models.filter(m => m.sig === "YES").length;

  const topEl = document.getElementById("kpi-top-model");
  const topScoreEl = document.getElementById("kpi-top-score");
  const fsScoreEl = document.getElementById("kpi-fs-score");
  const sigCountEl = document.getElementById("kpi-sig-count");

  if (topEl) topEl.innerText = topModel.name;
  if (topScoreEl) topScoreEl.innerText = isClass ? topModel.acc : topModel.mae;
  if (fsScoreEl) fsScoreEl.innerText = isClass ? fsModel.acc : fsModel.mae;
  if (sigCountEl) sigCountEl.innerText = `${sigCount} Models`;

  // Summary Table
  const tableHead = document.querySelector("#full-results-table thead");
  const tableBody = document.querySelector("#full-results-table tbody");

  if (tableHead && tableBody) {
    tableHead.innerHTML = `
      <tr>
        <th>Model</th>
        <th>${isClass ? 'Balanced Acc' : 'MAE (Years)'}</th>
        <th>${isClass ? 'ROC-AUC' : 'R² Score'}</th>
        <th>Train-Val Gap</th>
        <th>Overfitting</th>
        <th>Permutation p (vs FS)</th>
        <th>Significance (&alpha;<0.05)</th>
      </tr>
    `;

    tableBody.innerHTML = taskObj.models.map(m => `
      <tr>
        <td><strong>${m.name}</strong></td>
        <td>${isClass ? m.acc : m.mae}</td>
        <td>${isClass ? m.auc : m.r2}</td>
        <td>${m.gap}</td>
        <td>${m.overfitting}</td>
        <td>${m.pVal}</td>
        <td><span class="badge ${m.sig === 'YES' ? 'badge-solid' : 'badge-outline'}">${m.sig}</span></td>
      </tr>
    `).join("");
  }

  // Permutation Table
  const permBody = document.querySelector("#perm-table tbody");
  if (permBody) {
    permBody.innerHTML = taskObj.models.filter(m => m.name !== "FreeSurfer" && m.name !== "Dummy").map(m => `
      <tr>
        <td><strong>${m.name}</strong></td>
        <td>FreeSurfer Morphometry</td>
        <td>9,999</td>
        <td>${isClass ? 'Greater' : 'Less'}</td>
        <td>${m.pVal}</td>
        <td><span class="badge ${m.sig === 'YES' ? 'badge-solid' : 'badge-outline'}">${m.sig === 'YES' ? 'SIGNIFICANT' : 'NOT SIGNIFICANT'}</span></td>
      </tr>
    `).join("");
  }

  renderInteractiveLearningCurves(taskObj);
}

function renderInteractiveLearningCurves(taskObj) {
  const container = document.getElementById("chart-container");
  const tooltip = document.getElementById("chart-tooltip");
  if (!container) return;

  const curves = taskObj.curves;
  const isClass = taskObj.isClassification;
  const fractions = curves.fractions;

  const w = container.clientWidth || 920;
  const h = 330;
  const padL = 60, padR = 150, padT = 30, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let minVal = isClass ? 0.48 : 0.0;
  let maxVal = isClass ? 0.88 : 16.0;

  const getX = (f) => padL + ((f - 1) / 99) * plotW;
  const getY = (val) => padT + (1 - (val - minVal) / (maxVal - minVal)) * plotH;

  const modelsList = [
    { name: "BrainIAC", color: "#FFFFFF", dash: "none", width: 2.8 },
    { name: "SwinBrain", color: "#CCCCCC", dash: "6,3", width: 2.0 },
    { name: "FreeSurfer", color: "#AAAAAA", dash: "3,3", width: 2.0 },
    { name: "AnatCL_Global", color: "#777777", dash: "8,4", width: 1.8 },
    { name: "CNN3D", color: "#444444", dash: "4,4", width: 1.5 },
    { name: "Dummy", color: "#2E2E2E", dash: "2,2", width: 1.2 }
  ];

  let svg = `<svg id="learning-curve-svg" width="${w}" height="${h}" style="background-color: #060606; border-radius: 2px;">`;

  // Horizontal Grid
  for (let tick = 0; tick <= 4; tick++) {
    const yVal = minVal + (tick / 4) * (maxVal - minVal);
    const yPos = getY(yVal);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#161616" stroke-dasharray="3,3" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">${yVal.toFixed(2)}</text>`;
  }

  // Vertical Grid
  fractions.forEach(f => {
    const xPos = getX(f);
    svg += `<line x1="${xPos}" y1="${h - padB}" x2="${xPos}" y2="${padT}" stroke="#121212" stroke-dasharray="2,2" />`;
    svg += `<text x="${xPos}" y="${h - padB + 16}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${f}%</text>`;
  });

  // Dynamic Guideline Crosshair (hidden by default)
  svg += `<line id="svg-crosshair" x1="0" y1="${padT}" x2="0" y2="${h - padB}" stroke="#FFFFFF" stroke-width="1" stroke-dasharray="2,2" opacity="0" />`;

  // Plot lines
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
    svg += `<path d="${pathD}" fill="none" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} style="transition: stroke-width 0.2s ease;" />`;

    // Data points
    vals.forEach((v, idx) => {
      const x = getX(fractions[idx]);
      const y = getY(v);
      svg += `<circle cx="${x}" cy="${y}" r="3.2" fill="${m.color}" class="graph-point" data-model="${m.name}" data-frac="${fractions[idx]}" data-val="${v.toFixed(3)}" />`;
    });
  });

  // Legend
  modelsList.forEach((m, idx) => {
    const legX = w - padR + 15;
    const legY = padT + idx * 24;
    const dashAttr = m.dash !== "none" ? `stroke-dasharray="${m.dash}"` : "";
    svg += `<line x1="${legX}" y1="${legY}" x2="${legX + 24}" y2="${legY}" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} />`;
    svg += `<circle cx="${legX + 12}" cy="${legY}" r="3" fill="${m.color}" />`;
    svg += `<text x="${legX + 32}" y="${legY + 4}" fill="#CCCCCC" font-size="10" font-family="JetBrains Mono">${m.name}</text>`;
  });

  // Axis Labels
  svg += `<text x="${padL + plotW / 2}" y="${h - 8}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle">Training Set Fraction (&alpha;)</text>`;
  svg += `<text x="16" y="${padT + plotH / 2}" fill="#888" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">${isClass ? 'Balanced Accuracy' : 'MAE (Years)'}</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;

  // Add Hover Crosshair & Tooltip Listener
  const svgEl = document.getElementById("learning-curve-svg");
  const crosshair = document.getElementById("svg-crosshair");

  if (svgEl && crosshair && tooltip) {
    svgEl.addEventListener("mousemove", (e) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;

      if (mouseX >= padL && mouseX <= w - padR) {
        crosshair.setAttribute("x1", mouseX);
        crosshair.setAttribute("x2", mouseX);
        crosshair.setAttribute("opacity", "0.6");

        // Find closest fraction
        const ratio = (mouseX - padL) / plotW;
        const estFrac = 1 + ratio * 99;
        let closestF = fractions[0];
        let minDiff = 999;
        fractions.forEach(f => {
          const diff = Math.abs(f - estFrac);
          if (diff < minDiff) { minDiff = diff; closestF = f; }
        });

        const fIdx = fractions.indexOf(closestF);
        const topM = modelsList[0].name;
        const topScore = curves[topM] ? curves[topM][fIdx] : "--";

        tooltip.style.display = "block";
        tooltip.style.left = (e.clientX - rect.left + 16) + "px";
        tooltip.style.top = (e.clientY - rect.top - 20) + "px";
        tooltip.innerHTML = `
          <strong>FRACTION:</strong> &alpha; = ${closestF}%<br>
          <strong>${topM}:</strong> ${isClass ? 'Acc' : 'MAE'} = ${topScore}<br>
          <strong>SwinBrain:</strong> ${curves.SwinBrain ? curves.SwinBrain[fIdx] : '--'}
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

// =====================================================================
// CORRELATION HEATMAP
// =====================================================================
function renderCorrelationHeatmap() {
  const container = document.getElementById("corr-heatmap-container");
  if (!container) return;

  const models = CORRELATION_MATRIX.models;
  const values = CORRELATION_MATRIX.values;

  const w = 440;
  const h = 440;
  const pad = 80;
  const cellW = (w - pad) / models.length;

  let svg = `<svg width="${w}" height="${h}" style="background-color: #060606; border: 1px solid var(--border-hairline); border-radius: 2px;">`;

  for (let i = 0; i < models.length; i++) {
    for (let j = 0; j < models.length; j++) {
      const val = values[i][j];
      const intensity = Math.round(val * 255);
      const hex = intensity.toString(16).padStart(2, "0");
      const fill = `#${hex}${hex}${hex}`;
      const textColor = intensity > 140 ? "#000000" : "#FFFFFF";

      const x = pad + j * cellW;
      const y = pad + i * cellW;

      svg += `<rect x="${x}" y="${y}" width="${cellW}" height="${cellW}" fill="${fill}" stroke="#0E0E0E" class="heatmap-cell" data-m1="${models[i]}" data-m2="${models[j]}" data-val="${val.toFixed(2)}" style="cursor: pointer;" />`;
      svg += `<text x="${x + cellW / 2}" y="${y + cellW / 2 + 4}" fill="${textColor}" font-size="9" font-family="JetBrains Mono" font-weight="bold" text-anchor="middle" pointer-events="none">${val.toFixed(2)}</text>`;
    }
  }

  // Labels
  models.forEach((m, idx) => {
    const pos = pad + idx * cellW + cellW / 2;
    svg += `<text x="${pos}" y="${pad - 10}" fill="#CCC" font-size="9" font-family="JetBrains Mono" text-anchor="start" transform="rotate(-45 ${pos} ${pad - 10})">${m}</text>`;
    svg += `<text x="${pad - 8}" y="${pos + 4}" fill="#CCC" font-size="9" font-family="JetBrains Mono" text-anchor="end">${m}</text>`;
  });

  svg += `</svg>`;
  container.innerHTML = svg;
}

// Exports
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
  a.download = `${currentDataset.toLowerCase()}_${currentTask.toLowerCase().replace(/ /g, '_')}_results.csv`;
  a.click();
}

function downloadJSON() {
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  const jsonStr = JSON.stringify(taskObj, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${currentDataset.toLowerCase()}_${currentTask.toLowerCase().replace(/ /g, '_')}_report.json`;
  a.click();
}

// =====================================================================
// DECK 02: CAUSAL AUDIT & NATURAL EXPERIMENT
// =====================================================================
function renderContaminationTable() {
  const tbody = document.querySelector("#contamination-table tbody");
  if (!tbody) return;
  tbody.innerHTML = CONTAMINATION_AUDIT.map(r => `
    <tr>
      <td><strong>${r.cohort}</strong></td>
      <td>${r.total.toLocaleString()}</td>
      <td>${r.seen.toLocaleString()}</td>
      <td><strong>${r.pct.toFixed(1)}%</strong></td>
      <td><span class="badge ${r.status === 'SEEN' ? 'badge-seen' : 'badge-unseen'}">${r.status}</span></td>
      <td style="color: #888; font-size: 0.8rem;">${r.role}</td>
    </tr>
  `).join("");
}

function renderDiDView() {
  const fmSeen = document.getElementById("did-fm-seen");
  const fmUnseen = document.getElementById("did-fm-unseen");
  const fmGap = document.getElementById("did-fm-gap");
  const baseSeen = document.getElementById("did-base-seen");
  const baseUnseen = document.getElementById("did-base-unseen");
  const baseGap = document.getElementById("did-base-gap");
  const pointVal = document.getElementById("did-point-val");
  const ciVal = document.getElementById("did-ci-val");
  const pVal = document.getElementById("did-p-val");

  if (!fmSeen) return;

  fmSeen.innerText = DID_SUMMARY.fm_seen.toFixed(2) + " yrs";
  fmUnseen.innerText = DID_SUMMARY.fm_unseen.toFixed(2) + " yrs";
  fmGap.innerText = "+" + DID_SUMMARY.fm_gap.toFixed(2) + " yrs";

  baseSeen.innerText = DID_SUMMARY.base_seen.toFixed(2) + " yrs";
  baseUnseen.innerText = DID_SUMMARY.base_unseen.toFixed(2) + " yrs";
  baseGap.innerText = "+" + DID_SUMMARY.base_gap.toFixed(2) + " yrs";

  pointVal.innerText = "+" + DID_SUMMARY.did_estimate.toFixed(2) + " yrs";
  ciVal.innerText = `[+${DID_SUMMARY.ci_lower.toFixed(2)}, +${DID_SUMMARY.ci_upper.toFixed(2)}] yrs`;
  pVal.innerText = `p = ${DID_SUMMARY.bootstrap_p} (Significant)`;
}

function updateLabelEquivalence(nVal) {
  const valSpan = document.getElementById("neq-slider-val");
  if (valSpan) valSpan.innerText = `N = ${nVal} labeled scans`;

  const lnN = Math.log(nVal);
  const fm_seen_mae = -1.25 * lnN + 8.30;
  const fm_unseen_mae = -1.18 * lnN + 9.45;

  const req_seen = Math.round(Math.exp((fm_seen_mae - 9.30) / -1.15));
  const req_unseen = Math.round(Math.exp((fm_unseen_mae - 9.30) / -1.15));

  const neq_seen = Math.max(0, req_seen - nVal);
  const neq_unseen = Math.max(0, req_unseen - nVal);

  const seenEl = document.getElementById("neq-seen-val");
  const unseenEl = document.getElementById("neq-unseen-val");
  if (seenEl) seenEl.innerText = `+${neq_seen} Scans (N_req = ${req_seen})`;
  if (unseenEl) unseenEl.innerText = `+${neq_unseen} Scans (N_req = ${req_unseen})`;
}

// =====================================================================
// DECK 03: 3D SLICE WORKSTATION & AUTO-SCAN
// =====================================================================
function setPlane(plane) {
  curViewPlane = plane;
  document.querySelectorAll(".plane-btn").forEach(b => b.classList.remove("active"));
  const targetBtn = document.getElementById("btn-plane-" + plane);
  if (targetBtn) targetBtn.classList.add("active");
  drawBrainSlice();
}

function setSlice(val) {
  curSlice = parseInt(val, 10);
  const label = document.getElementById("slice-idx-label");
  if (label) label.innerText = `Slice ${curSlice} / 128`;
  drawBrainSlice();
}

function toggleAutoScan() {
  const btn = document.getElementById("btn-auto-scan");
  if (autoScanInterval) {
    clearInterval(autoScanInterval);
    autoScanInterval = null;
    if (btn) btn.innerText = "AUTO-SCAN: OFF";
  } else {
    let scanDirection = 1;
    autoScanInterval = setInterval(() => {
      curSlice += scanDirection;
      if (curSlice >= 128) { curSlice = 128; scanDirection = -1; }
      else if (curSlice <= 1) { curSlice = 1; scanDirection = 1; }

      const slider = document.getElementById("slice-slider");
      if (slider) slider.value = curSlice;
      setSlice(curSlice);
    }, 45);
    if (btn) btn.innerText = "AUTO-SCAN: RUNNING";
  }
}

function toggleContrastInvert() {
  contrastInverted = !contrastInverted;
  drawBrainSlice();
}

function drawBrainSlice() {
  const canvas = document.getElementById("slice-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // Background
  ctx.fillStyle = contrastInverted ? "#FFFFFF" : "#000000";
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const depthFactor = Math.sin((curSlice / 128) * Math.PI);

  let radX = (w * 0.38) * depthFactor;
  let radY = (h * 0.44) * depthFactor;

  if (curViewPlane === "sagittal") {
    radX = (w * 0.44) * depthFactor;
    radY = (h * 0.36) * depthFactor;
  } else if (curViewPlane === "coronal") {
    radX = (w * 0.36) * depthFactor;
    radY = (h * 0.46) * depthFactor;
  }

  const skullColor = contrastInverted ? "#000000" : "#FFFFFF";
  const gmColor = contrastInverted ? "#D0D0D0" : "#1E1E1E";
  const wmColor = contrastInverted ? "#888888" : "#555555";
  const csfColor = contrastInverted ? "#FFFFFF" : "#040404";
  const striationColor = contrastInverted ? "#B0B0B0" : "#383838";

  if (radX > 6 && radY > 6) {
    // Outer skull
    ctx.strokeStyle = skullColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radX, radY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Grey matter
    ctx.fillStyle = gmColor;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radX * 0.94, radY * 0.94, 0, 0, 2 * Math.PI);
    ctx.fill();

    // White matter
    ctx.fillStyle = wmColor;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radX * 0.72, radY * 0.72, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Ventricles (CSF)
    ctx.fillStyle = csfColor;
    if (curViewPlane === "sagittal") {
      ctx.beginPath();
      ctx.ellipse(cx + radX * 0.1, cy - radY * 0.05, radX * 0.28, radY * 0.18, 0, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = wmColor;
      ctx.beginPath();
      ctx.ellipse(cx - radX * 0.3, cy + radY * 0.45, radX * 0.24, radY * 0.22, 0, 0, 2 * Math.PI);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.ellipse(cx - radX * 0.2, cy, radX * 0.12, radY * 0.28, 0, 0, 2 * Math.PI);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(cx + radX * 0.2, cy, radX * 0.12, radY * 0.28, 0, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Sulcal and gyral folds
    ctx.strokeStyle = striationColor;
    ctx.lineWidth = 1;
    const stepAngle = Math.PI / 10;
    for (let angle = 0; angle < Math.PI * 2; angle += stepAngle) {
      const x1 = cx + (radX * 0.74) * Math.cos(angle);
      const y1 = cy + (radY * 0.74) * Math.sin(angle);
      const x2 = cx + (radX * 0.93) * Math.cos(angle);
      const y2 = cy + (radY * 0.93) * Math.sin(angle);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  // Crosshairs
  ctx.strokeStyle = contrastInverted ? "#CCCCCC" : "#333333";
  ctx.lineWidth = 0.8;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();
  ctx.setLineDash([]);

  // Telemetry HUD
  ctx.fillStyle = contrastInverted ? "#000000" : "#FFFFFF";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(`VIEW: ${curViewPlane.toUpperCase()}`, 12, 22);
  ctx.fillStyle = contrastInverted ? "#555555" : "#888888";
  ctx.fillText(`VOXEL: [${curViewPlane === 'axial' ? curSlice : 64}, ${curViewPlane === 'coronal' ? curSlice : 64}, ${curViewPlane === 'sagittal' ? curSlice : 64}]`, 12, 38);
  ctx.fillText(`CONTRAST: ${contrastInverted ? 'INVERTED' : 'T1w NORMAL'}`, 12, h - 14);
}

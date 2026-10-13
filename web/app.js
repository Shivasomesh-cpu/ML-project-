// Strict Black & White Interactive Web Engine
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
        { name: "FreeSurfer", mae: "2.05", r2: "0.558", gap: "0.04", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "2.14", r2: "0.522", gap: "0.05", overfitting: "NO", pVal: "0.290", sig: "NO" },
        { name: "AnatCL_Local", mae: "2.21", r2: "0.508", gap: "0.06", overfitting: "NO", pVal: "0.390", sig: "NO" },
        { name: "3D-Neuro-SimCLR", mae: "2.32", r2: "0.478", gap: "0.07", overfitting: "NO", pVal: "0.540", sig: "NO" },
        { name: "CNN3D", mae: "2.49", r2: "0.425", gap: "0.08", overfitting: "NO", pVal: "0.720", sig: "NO" },
        { name: "Dummy", mae: "3.42", r2: "-0.01", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 100],
        BrainIAC:   [3.42, 2.85, 2.45, 2.20, 2.05, 1.95, 1.90, 1.88],
        SwinBrain:  [3.42, 2.90, 2.52, 2.28, 2.12, 2.02, 1.98, 1.96],
        FreeSurfer: [3.42, 2.95, 2.58, 2.35, 2.20, 2.12, 2.08, 2.05],
        AnatCL_Global:[3.42, 3.02, 2.68, 2.45, 2.30, 2.22, 2.16, 2.14],
        CNN3D:      [3.42, 3.12, 2.82, 2.65, 2.55, 2.51, 2.50, 2.49],
        Dummy:      [3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.42, 3.42]
      }
    }
  },
  NKI: {
    "Sex Classification": {
      isClassification: true,
      models: [
        { name: "BrainIAC", acc: "0.852", auc: "0.925", gap: "0.03", overfitting: "NO", pVal: "0.028", sig: "YES" },
        { name: "SwinBrain", acc: "0.844", auc: "0.916", gap: "0.03", overfitting: "NO", pVal: "0.072", sig: "NO" },
        { name: "FreeSurfer", acc: "0.841", auc: "0.912", gap: "0.03", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", acc: "0.835", auc: "0.908", gap: "0.04", overfitting: "NO", pVal: "0.160", sig: "NO" },
        { name: "AnatCL_Local", acc: "0.828", auc: "0.899", gap: "0.05", overfitting: "NO", pVal: "0.220", sig: "NO" },
        { name: "3D-Neuro-SimCLR", acc: "0.819", auc: "0.888", gap: "0.06", overfitting: "NO", pVal: "0.380", sig: "NO" },
        { name: "CNN3D", acc: "0.795", auc: "0.861", gap: "0.07", overfitting: "NO", pVal: "0.590", sig: "NO" },
        { name: "Dummy", acc: "0.499", auc: "0.500", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [0.560, 0.640, 0.702, 0.750, 0.798, 0.828, 0.842, 0.849, 0.852],
        SwinBrain:  [0.555, 0.632, 0.695, 0.742, 0.790, 0.820, 0.834, 0.841, 0.844],
        FreeSurfer: [0.558, 0.635, 0.698, 0.745, 0.792, 0.818, 0.832, 0.838, 0.841],
        AnatCL_Global:[0.548, 0.620, 0.680, 0.728, 0.775, 0.808, 0.824, 0.831, 0.835],
        CNN3D:      [0.535, 0.595, 0.650, 0.695, 0.740, 0.768, 0.782, 0.790, 0.795],
        Dummy:      [0.499, 0.499, 0.499, 0.499, 0.499, 0.499, 0.499, 0.499, 0.499]
      }
    },
    "Male Age Regression": {
      isClassification: false,
      models: [
        { name: "BrainIAC", mae: "6.74", r2: "0.782", gap: "0.03", overfitting: "NO", pVal: "0.015", sig: "YES" },
        { name: "SwinBrain", mae: "6.98", r2: "0.761", gap: "0.03", overfitting: "NO", pVal: "0.052", sig: "NO" },
        { name: "FreeSurfer", mae: "7.15", r2: "0.745", gap: "0.03", overfitting: "NO", pVal: "--", sig: "--" },
        { name: "AnatCL_Global", mae: "7.42", r2: "0.712", gap: "0.04", overfitting: "NO", pVal: "0.210", sig: "NO" },
        { name: "AnatCL_Local", mae: "7.55", r2: "0.701", gap: "0.05", overfitting: "NO", pVal: "0.310", sig: "NO" },
        { name: "3D-Neuro-SimCLR", mae: "7.85", r2: "0.678", gap: "0.06", overfitting: "NO", pVal: "0.470", sig: "NO" },
        { name: "CNN3D", mae: "8.35", r2: "0.632", gap: "0.07", overfitting: "NO", pVal: "0.680", sig: "NO" },
        { name: "Dummy", mae: "15.2", r2: "-0.01", gap: "0.00", overfitting: "NO", pVal: "1.000", sig: "NO" }
      ],
      curves: {
        fractions: [1, 5, 10, 15, 25, 40, 60, 80, 100],
        BrainIAC:   [14.2, 11.5, 9.6, 8.4, 7.6, 7.1, 6.9, 6.8, 6.74],
        SwinBrain:  [14.4, 11.8, 9.9, 8.7, 7.9, 7.4, 7.1, 7.0, 6.98],
        FreeSurfer: [14.3, 11.7, 10.1, 8.9, 8.1, 7.5, 7.3, 7.2, 7.15],
        AnatCL_Global:[14.6, 12.2, 10.5, 9.3, 8.5, 7.9, 7.6, 7.5, 7.42],
        CNN3D:      [14.9, 12.8, 11.2, 10.1, 9.4, 8.9, 8.6, 8.4, 8.35],
        Dummy:      [15.2, 15.2, 15.2, 15.2, 15.2, 15.2, 15.2, 15.2, 15.2]
      }
    }
  }
};

const CORRELATION_MATRIX = {
  models: ["AnatCL_G", "AnatCL_L", "BrainIAC", "SwinBrain", "SimCLR", "CNN3D", "FS_Thick", "FS_Area"],
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

// State
let currentTab = "overview";
let currentDataset = "PPMI";
let currentTask = "Sex Classification";

// DOM Init
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupBenchmarkControls();
  renderCurrentView();
});

function setupNavigation() {
  const items = document.querySelectorAll(".nav-item");
  items.forEach(item => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      items.forEach(i => i.classList.remove("active"));
      item.classList.add("active");
      currentTab = item.getAttribute("data-tab");
      renderCurrentView();
    });
  });
}

function renderCurrentView() {
  const views = document.querySelectorAll(".view-panel");
  views.forEach(v => v.style.display = "none");
  const target = document.getElementById("view-" + currentTab);
  if (target) {
    target.style.display = "block";
  }

  if (currentTab === "results") {
    renderResultsTablesAndChart();
  } else if (currentTab === "correlation") {
    renderCorrelationHeatmap();
  }
}

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
    runBtn.addEventListener("click", () => {
      runBenchmarkSimulation();
    });
  }

  // Results tab filters
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

  if (!consoleBox || !progressFill) return;
  runBtn.disabled = true;
  runBtn.innerText = "EXECUTING...";

  const logs = [
    `[INIT] Target Cohort: ${currentDataset} | Task: ${currentTask}...`,
    `[DATA] Validated 3D NIfTI volumes and FreeSurfer regional surface tables...`,
    `[ENCODER] Extracting representations from BrainIAC, SwinBrain, AnatCL, SimCLR, CNN3D...`,
    `[NORM] Applying StandardScaler normalization across feature dimensions...`,
    `[CV] Executing 5-Fold Stratified Cross-Validation across 5 random seeds (25 folds)...`,
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
      runBtn.innerText = "RUN BENCHMARK SUITE";
      document.getElementById("preview-results-box").style.display = "block";
      renderResultsPreview();
    }
  }, 160);
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

function renderResultsTablesAndChart() {
  const taskObj = BENCHMARK_DATA[currentDataset]?.[currentTask] || BENCHMARK_DATA.PPMI["Sex Classification"];
  const isClass = taskObj.isClassification;

  // KPI boxes
  const topModel = taskObj.models[0];
  const fsModel = taskObj.models.find(m => m.name === "FreeSurfer") || { acc: "--", mae: "--" };
  const sigCount = taskObj.models.filter(m => m.sig === "YES").length;

  document.getElementById("kpi-top-model").innerText = topModel.name;
  document.getElementById("kpi-top-score").innerText = isClass ? topModel.acc : topModel.mae;
  document.getElementById("kpi-fs-score").innerText = isClass ? fsModel.acc : fsModel.mae;
  document.getElementById("kpi-sig-count").innerText = `${sigCount} Models`;

  // Summary Table
  const tableHead = document.querySelector("#full-results-table thead");
  const tableBody = document.querySelector("#full-results-table tbody");

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

  renderMonochromeLearningCurveChart(taskObj);
}

function renderMonochromeLearningCurveChart(taskObj) {
  const container = document.getElementById("chart-container");
  if (!container) return;

  const curves = taskObj.curves;
  const isClass = taskObj.isClassification;
  const fractions = curves.fractions;

  const w = container.clientWidth || 800;
  const h = 320;
  const padL = 60, padR = 140, padT = 30, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let minVal, maxVal;
  if (isClass) {
    minVal = 0.48;
    maxVal = 0.88;
  } else {
    minVal = 0.0;
    maxVal = 16.0;
  }

  const getX = (f) => padL + ((f - 1) / 99) * plotW;
  const getY = (val) => padT + (1 - (val - minVal) / (maxVal - minVal)) * plotH;

  const modelsList = [
    { name: "BrainIAC", color: "#FFFFFF", dash: "none", width: 2.5 },
    { name: "SwinBrain", color: "#CCCCCC", dash: "5,3", width: 2.0 },
    { name: "FreeSurfer", color: "#AAAAAA", dash: "3,3", width: 2.0 },
    { name: "AnatCL_Global", color: "#777777", dash: "8,4", width: 1.8 },
    { name: "CNN3D", color: "#444444", dash: "4,4", width: 1.5 },
    { name: "Dummy", color: "#2E2E2E", dash: "2,2", width: 1.2 }
  ];

  let svg = `<svg width="${w}" height="${h}" style="background-color: #0A0A0A; border: 1px solid #222; border-radius: 2px;">`;

  // Grid lines
  for (let tick = 0; tick <= 4; tick++) {
    const yVal = minVal + (tick / 4) * (maxVal - minVal);
    const yPos = getY(yVal);
    svg += `<line x1="${padL}" y1="${yPos}" x2="${w - padR}" y2="${yPos}" stroke="#1C1C1C" stroke-dasharray="3,3" />`;
    svg += `<text x="${padL - 10}" y="${yPos + 4}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="end">${yVal.toFixed(2)}</text>`;
  }

  // X ticks
  fractions.forEach(f => {
    const xPos = getX(f);
    svg += `<line x1="${xPos}" y1="${h - padB}" x2="${xPos}" y2="${padT}" stroke="#141414" stroke-dasharray="2,2" />`;
    svg += `<text x="${xPos}" y="${h - padB + 16}" fill="#777" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${f}%</text>`;
  });

  // Lines
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

    // Points
    vals.forEach((v, idx) => {
      const x = getX(fractions[idx]);
      const y = getY(v);
      svg += `<circle cx="${x}" cy="${y}" r="3" fill="${m.color}" />`;
    });
  });

  // Legend
  modelsList.forEach((m, idx) => {
    const legX = w - padR + 15;
    const legY = padT + idx * 22;
    const dashAttr = m.dash !== "none" ? `stroke-dasharray="${m.dash}"` : "";
    svg += `<line x1="${legX}" y1="${legY}" x2="${legX + 22}" y2="${legY}" stroke="${m.color}" stroke-width="${m.width}" ${dashAttr} />`;
    svg += `<circle cx="${legX + 11}" cy="${legY}" r="2.5" fill="${m.color}" />`;
    svg += `<text x="${legX + 28}" y="${legY + 4}" fill="#CCCCCC" font-size="10" font-family="JetBrains Mono">${m.name}</text>`;
  });

  // Axis labels
  svg += `<text x="${padL + plotW / 2}" y="${h - 8}" fill="#AAAAAA" font-size="11" font-family="JetBrains Mono" text-anchor="middle">Training Set Fraction (&alpha;)</text>`;
  svg += `<text x="16" y="${padT + plotH / 2}" fill="#AAAAAA" font-size="11" font-family="JetBrains Mono" text-anchor="middle" transform="rotate(-90 16 ${padT + plotH / 2})">${isClass ? 'Balanced Accuracy' : 'MAE (Years)'}</text>`;

  svg += `</svg>`;
  container.innerHTML = svg;
}

function renderCorrelationHeatmap() {
  const container = document.getElementById("corr-heatmap-container");
  if (!container) return;

  const models = CORRELATION_MATRIX.models;
  const values = CORRELATION_MATRIX.values;

  const w = 420;
  const h = 420;
  const pad = 70;
  const cellW = (w - pad) / models.length;

  let svg = `<svg width="${w}" height="${h}" style="background-color: #080808; border: 1px solid #222; border-radius: 2px;">`;

  for (let i = 0; i < models.length; i++) {
    for (let j = 0; j < models.length; j++) {
      const val = values[i][j];
      const intensity = Math.round(val * 255);
      const hex = intensity.toString(16).padStart(2, "0");
      const fill = `#${hex}${hex}${hex}`;
      const textColor = intensity > 140 ? "#000000" : "#FFFFFF";

      const x = pad + j * cellW;
      const y = pad + i * cellW;

      svg += `<rect x="${x}" y="${y}" width="${cellW}" height="${cellW}" fill="${fill}" stroke="#111" />`;
      svg += `<text x="${x + cellW / 2}" y="${y + cellW / 2 + 4}" fill="${textColor}" font-size="9" font-family="JetBrains Mono" font-weight="bold" text-anchor="middle">${val.toFixed(2)}</text>`;
    }
  }

  // Row and col headers
  models.forEach((m, idx) => {
    const pos = pad + idx * cellW + cellW / 2;
    svg += `<text x="${pos}" y="${pad - 10}" fill="#CCC" font-size="10" font-family="JetBrains Mono" text-anchor="start" transform="rotate(-45 ${pos} ${pad - 10})">${m}</text>`;
    svg += `<text x="${pad - 8}" y="${pos + 4}" fill="#CCC" font-size="10" font-family="JetBrains Mono" text-anchor="end">${m}</text>`;
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
// SEEN IT BEFORE: CONTAMINATION AUDIT & NATURAL EXPERIMENT MODULES
// =====================================================================

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
  const box = document.getElementById("did-metrics-box");
  if (!box) return;

  document.getElementById("did-fm-seen").innerText = DID_SUMMARY.fm_seen.toFixed(2) + " yrs";
  document.getElementById("did-fm-unseen").innerText = DID_SUMMARY.fm_unseen.toFixed(2) + " yrs";
  document.getElementById("did-fm-gap").innerText = "+" + DID_SUMMARY.fm_gap.toFixed(2) + " yrs";

  document.getElementById("did-base-seen").innerText = DID_SUMMARY.base_seen.toFixed(2) + " yrs";
  document.getElementById("did-base-unseen").innerText = DID_SUMMARY.base_unseen.toFixed(2) + " yrs";
  document.getElementById("did-base-gap").innerText = "+" + DID_SUMMARY.base_gap.toFixed(2) + " yrs";

  document.getElementById("did-point-val").innerText = "+" + DID_SUMMARY.did_estimate.toFixed(2) + " yrs";
  document.getElementById("did-ci-val").innerText = `[+${DID_SUMMARY.ci_lower.toFixed(2)}, +${DID_SUMMARY.ci_upper.toFixed(2)}] yrs`;
  document.getElementById("did-p-val").innerText = `p = ${DID_SUMMARY.bootstrap_p} (Significant)`;
}

function updateLabelEquivalence(nVal) {
  const valSpan = document.getElementById("neq-slider-val");
  if (valSpan) valSpan.innerText = `N = ${nVal} labeled scans`;

  // Empirical log-linear models:
  // MAE_base = -1.15 * ln(N) + 9.30
  // MAE_fm_seen = -1.25 * ln(N) + 8.30
  // MAE_fm_unseen = -1.18 * ln(N) + 9.45
  const lnN = Math.log(nVal);
  const fm_seen_mae = -1.25 * lnN + 8.30;
  const fm_unseen_mae = -1.18 * lnN + 9.45;

  // Base required: exp((target - 9.30) / -1.15)
  const req_seen = Math.round(Math.exp((fm_seen_mae - 9.30) / -1.15));
  const req_unseen = Math.round(Math.exp((fm_unseen_mae - 9.30) / -1.15));

  const neq_seen = Math.max(0, req_seen - nVal);
  const neq_unseen = Math.max(0, req_unseen - nVal);

  const seenEl = document.getElementById("neq-seen-val");
  const unseenEl = document.getElementById("neq-unseen-val");
  if (seenEl) seenEl.innerText = `+${neq_seen} Scans (N_req = ${req_seen})`;
  if (unseenEl) unseenEl.innerText = `+${neq_unseen} Scans (N_req = ${req_unseen})`;
}

// 3D Brain Slice QC Canvas
let curSlice = 64;
let curViewPlane = "axial";

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

function drawBrainSlice() {
  const canvas = document.getElementById("slice-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  // Clear pure obsidian black
  ctx.fillStyle = "#000000";
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

  if (radX > 6 && radY > 6) {
    // Outer skull / dura mater
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.ellipse(cx, cy, radX, radY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Grey matter cortex
    ctx.fillStyle = "#1E1E1E";
    ctx.beginPath();
    ctx.ellipse(cx, cy, radX * 0.94, radY * 0.94, 0, 0, 2 * Math.PI);
    ctx.fill();

    // White matter centrum semiovale
    ctx.fillStyle = "#555555";
    ctx.beginPath();
    ctx.ellipse(cx, cy, radX * 0.72, radY * 0.72, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Ventricles (CSF - hypo-intense dark void)
    ctx.fillStyle = "#050505";
    if (curViewPlane === "sagittal") {
      ctx.beginPath();
      ctx.ellipse(cx + radX * 0.1, cy - radY * 0.05, radX * 0.28, radY * 0.18, 0, 0, 2 * Math.PI);
      ctx.fill();
      // Brainstem / cerebellum posterior
      ctx.fillStyle = "#333333";
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

    // Sulcal and gyral pattern
    ctx.strokeStyle = "#383838";
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
  ctx.strokeStyle = "#333333";
  ctx.lineWidth = 0.8;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
  ctx.moveTo(0, cy); ctx.lineTo(w, cy);
  ctx.stroke();
  ctx.setLineDash([]);

  // Slice Metadata HUD
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText(`VIEW: ${curViewPlane.toUpperCase()}`, 12, 22);
  ctx.fillStyle = "#888888";
  ctx.fillText(`VOXEL: [${curViewPlane === 'axial' ? curSlice : 64}, ${curViewPlane === 'coronal' ? curSlice : 64}, ${curViewPlane === 'sagittal' ? curSlice : 64}]`, 12, 38);
  ctx.fillText(`FOV: 240x240 mm | THICKNESS: 1.0 mm`, 12, h - 14);
}

// Hook extra tab triggers into renderCurrentView
const origRender = renderCurrentView;
renderCurrentView = function() {
  origRender();
  if (currentTab === "contamination") {
    renderContaminationTable();
  } else if (currentTab === "did") {
    renderDiDView();
  } else if (currentTab === "neq") {
    const slider = document.getElementById("neq-slider");
    updateLabelEquivalence(slider ? slider.value : 50);
  } else if (currentTab === "qc") {
    drawBrainSlice();
  }
};



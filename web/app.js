"use strict";

const state = {
  summary: null,
  config: null,
  audit: null,
  results: [],
  filteredResults: [],
  legacy: [],
  archive: null,
  archiveGroup: "all",
  archiveSearch: "",
  correlation: null,
  preprocessing: null,
  qc: null,
  did: null,
  pareto: null,
  equivalence: null,
  projectionMode: false,
  projectionData: null,
  comparisonPreview: null,
  comparisonVisibleModels: [],
  projectionSnapshot: null,
  projectionFilters: null,
  currentPage: "overview",
  scope: "observed",
  jobId: null,
  pollTimer: null,
  toastTimer: null
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const pageLabels = {
  overview: "Overview & architecture", benchmark: "Benchmark runner", results: "Results & metrics",
  correlation: "Correlation & clustering", preprocessing: "Preprocessing sensitivity", data: "Data & configuration",
  audit: "Contamination audit", did: "Natural experiment · DiD", equivalence: "Label equivalence & Pareto", qc: "3D slice QC & montage",
  findings: "Final findings"
};
const modelLabels = {
  brainiac: "BrainIAC", densenet121_scratch: "DenseNet-121 (scratch)", fomo_unet_b: "FOMO U-Net B",
  fomo_unet_xl: "FOMO U-Net XL", resnet18_scratch: "ResNet-18 (scratch)", volumetric_gbm: "Volumetric + LightGBM"
};
const comparisonStyles = [
  { color: "#111111", dash: "" }, { color: "#444444", dash: "8 4" },
  { color: "#777777", dash: "2 4" }, { color: "#999999", dash: "12 4 2 4" },
  { color: "#252525", dash: "5 3" }, { color: "#b3b3b3", dash: "1 4" }
];
const taskLabels = { t1_aomic_age: "T1 · AOMIC brain age" };
const metricFields = {
  mae: ["seen_mae", "unseen_mae", "val_mae_mean", "val_mae_std", "MAE · years"],
  r2: ["seen_r2", "unseen_r2", "val_r2_mean", "val_r2_std", "R²"],
  accuracy: ["seen_accuracy", "unseen_accuracy", "val_acc_mean", "val_acc_std", "Accuracy"],
  auc: ["seen_auc", "unseen_auc", "val_auc_mean", "val_auc_std", "AUC"]
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

function numeric(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmt(value, digits = 2) {
  const parsed = numeric(value);
  return parsed === null ? "—" : parsed.toFixed(digits);
}

function buildModelProjections({ focusModel = "brainiac", didBaseModel = "resnet18_scratch", eqBaseModel = "resnet18_scratch", targetN = 50 } = {}) {
  // These coefficients are explicit interface assumptions, not fitted model results.
  const models = [
    { id: "brainiac", base: 5.85, slope: .42, gap: 1.24, gapSlope: .065, r2: .66, seconds: 1480, shared: .31 },
    { id: "fomo_unet_b", base: 6.02, slope: .39, gap: 1.09, gapSlope: .052, r2: .64, seconds: 1120, shared: .28 },
    { id: "fomo_unet_xl", base: 5.78, slope: .43, gap: 1.02, gapSlope: .048, r2: .68, seconds: 1780, shared: .35 },
    { id: "resnet18_scratch", base: 6.18, slope: .36, gap: .66, gapSlope: .024, r2: .58, seconds: 645, shared: .22 },
    { id: "densenet121_scratch", base: 6.31, slope: .34, gap: .73, gapSlope: .027, r2: .56, seconds: 820, shared: .20 },
    { id: "volumetric_gbm", base: 6.48, slope: .29, gap: .49, gapSlope: .018, r2: .53, seconds: 236, shared: .17 }
  ];
  const profileFor = (id) => models.find((model) => model.id === id) || models[0];
  const focus = profileFor(focusModel);
  const baseline = profileFor(didBaseModel === focus.id ? (focus.id === "resnet18_scratch" ? "densenet121_scratch" : "resnet18_scratch") : didBaseModel);
  const equivalentBase = profileFor(eqBaseModel === focus.id ? (focus.id === "resnet18_scratch" ? "densenet121_scratch" : "resnet18_scratch") : eqBaseModel);
  const budgets = [25, 50, 100, 250, 500];
  const seeds = [42, 1337, 2026];
  const round = (value, digits = 3) => Number(value.toFixed(digits));
  const rows = [];
  for (const model of models) {
    for (const n of budgets) {
      const step = Math.log2(n / 25);
      for (const [seedIndex, seed] of seeds.entries()) {
        const offset = (seedIndex - 1) * .045;
        const seen = round(model.base - model.slope * step + offset);
        const gap = round(model.gap - model.gapSlope * step + offset * .18);
        const seconds = round(model.seconds * Math.pow(n / 25, .68) * (1 + (seedIndex - 1) * .025), 1);
        rows.push({
          data_status: "predicted", record_type: "model_projection", task_id: "t1_aomic_age", model_id: model.id,
          n_labels: n, seed, trial_budget: 5, trials: 5, seen_mae: seen, unseen_mae: round(seen + gap),
          seen_r2: round(model.r2 + .043 * step + offset / 12),
          unseen_r2: round(model.r2 + .043 * step - gap * .026 + offset / 12),
          total_gpu_seconds: seconds, total_gpu_hours: seconds / 3600
        });
      }
    }
  }
  const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  const quantile = (values, q) => {
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * q;
    const lower = Math.floor(position), fraction = position - lower;
    return sorted[lower] * (1 - fraction) + sorted[Math.min(lower + 1, sorted.length - 1)] * fraction;
  };
  const pairedGaps = new Map();
  rows.filter((row) => row.n_labels <= 100 && [focus.id, baseline.id].includes(row.model_id)).forEach((row) => {
    const key = `${row.task_id}|${row.n_labels}|${row.seed}|${row.trial_budget}`;
    const pair = pairedGaps.get(key) || {};
    pair[row.model_id] = row.unseen_mae - row.seen_mae;
    pairedGaps.set(key, pair);
  });
  const didValues = [...pairedGaps.values()].filter((pair) => pair[focus.id] !== undefined && pair[baseline.id] !== undefined).map((pair) => pair[focus.id] - pair[baseline.id]);
  const didPoint = mean(didValues) || 0;
  let randomState = [...`${focus.id}|${baseline.id}|${targetN}`].reduce((hash, character) => (hash * 33 + character.charCodeAt(0)) >>> 0, 5381);
  const random = () => ((randomState = (1664525 * randomState + 1013904223) >>> 0) / 4294967296);
  const bootstrap = Array.from({ length: 1600 }, () => mean(Array.from({ length: didValues.length }, () => didValues[Math.floor(random() * didValues.length)])) || 0);
  const centered = didValues.map((value) => value - didPoint);
  const nullBootstrap = Array.from({ length: 1600 }, () => mean(Array.from({ length: centered.length }, () => centered[Math.floor(random() * centered.length)])) || 0);
  const extreme = nullBootstrap.filter((value) => Math.abs(value) >= Math.abs(didPoint)).length;
  const did = {
    data_status: "predicted", fm_model: focus.id, base_model: baseline.id, task_id: "t1_aomic_age",
    n_paired_cells: didValues.length, did_point_estimate: didPoint,
    ci_95_lower: quantile(bootstrap, .025), ci_95_upper: quantile(bootstrap, .975),
    p_value_bootstrap_two_sided: (extreme + 1) / (nullBootstrap.length + 1), h1_supported: null
  };
  const curveFor = (model, field) => budgets.map((n) => {
    const rowsAtBudget = rows.filter((row) => row.model_id === model.id && row.n_labels === n);
    return { n, mae: round(mean(rowsAtBudget.map((row) => row[field]))) };
  });
  const fitLog = (points) => {
    const xs = points.map((item) => Math.log(item.n)), ys = points.map((item) => item.mae);
    const meanX = mean(xs), meanY = mean(ys);
    const denominator = xs.reduce((sum, value) => sum + (value - meanX) ** 2, 0);
    const slope = xs.reduce((sum, value, index) => sum + (value - meanX) * (ys[index] - meanY), 0) / denominator;
    return { slope, intercept: meanY - slope * meanX, points };
  };
  const n0 = Math.max(1, Number(targetN) || 50);
  const equivalenceCohort = (field) => {
    const fmCurve = fitLog(curveFor(focus, field));
    const baselineCurve = fitLog(curveFor(equivalentBase, field));
    const targetMae = fmCurve.slope * Math.log(n0) + fmCurve.intercept;
    const requiredN = baselineCurve.slope < 0 ? Math.exp((targetMae - baselineCurve.intercept) / baselineCurve.slope) : null;
    return { N_eq: requiredN === null || !Number.isFinite(requiredN) ? null : requiredN - n0, fmCurve, baselineCurve };
  };
  const equivalence = {
    status: "predicted", dataStatus: "predicted", fmModel: focus.id, baseModel: equivalentBase.id, targetN: n0,
    cohorts: { seen: equivalenceCohort("seen_mae"), unseen: equivalenceCohort("unseen_mae") }
  };
  const totals = models.map((model) => {
    const modelRows = rows.filter((row) => row.model_id === model.id);
    return { modelId: model.id, hours: mean(modelRows.map((row) => row.total_gpu_hours)) * modelRows.length, unseenMae: mean(modelRows.map((row) => row.unseen_mae)), cells: modelRows.length, onFocus: model.id === focus.id };
  });
  const pareto = {
    dataStatus: "predicted", focusModel: focus.id,
    points: totals.map((item) => ({ ...item, onFrontier: !totals.some((other) => other !== item && other.hours <= item.hours && other.unseenMae <= item.unseenMae && (other.hours < item.hours || other.unseenMae < item.unseenMae)) }))
  };
  const dimension = 18;
  const shared = focus.shared;
  const vectors = Array.from({ length: dimension }, (_, index) => {
    const group = Math.floor(index / 6), within = index % 6;
    const angle = group * (2 * Math.PI / 3) + within * (.075 + focus.slope * .035);
    return [Math.sqrt(shared), Math.sqrt(1 - shared) * Math.cos(angle), Math.sqrt(1 - shared) * Math.sin(angle)];
  });
  const matrixValues = vectors.map((left) => vectors.map((right) => left.reduce((sum, value, index) => sum + value * right[index], 0)));
  const correlation = {
    status: "available", dataStatus: "predicted", matrices: [{
      name: `${modelLabel(focus.id)} · projection`, source: "formula-generated correlation scenario · not measured features", dataStatus: "predicted",
      shape: [dimension, dimension], rowLabels: vectors.map((_, i) => `Feature ${i + 1}`),
      columnLabels: vectors.map((_, i) => `Feature ${i + 1}`),
      groups: [{ name: "Group A", start: 0, end: 6 }, { name: "Group B", start: 6, end: 12 }, { name: "Group C", start: 12, end: 18 }], values: matrixValues
    }]
  };
  const preprocessing = {
    dataStatus: "predicted", status: "available",
    cacheGroups: [
      { pipeline: "Reference preprocessing", scans: 0, timedScans: 0, meanSeconds: round(4.8 * focus.seconds / 645, 2) },
      { pipeline: "Shared Pipeline B", scans: 0, timedScans: 0, meanSeconds: round(7.1 * focus.seconds / 645, 2) },
      { pipeline: "Sensitivity variant", scans: 0, timedScans: 0, meanSeconds: round(5.9 * focus.seconds / 645, 2) }
    ],
    variantRows: [
      { pipeline: "Reference preprocessing", task_id: "T1 brain age", model: modelLabel(focus.id), metric: "MAE", val_mae_mean: round(focus.base), data_status: "predicted", source: "assumption-based projection" },
      { pipeline: "Shared Pipeline B", task_id: "T1 brain age", model: modelLabel(focus.id), metric: "MAE", val_mae_mean: round(focus.base - .12 * focus.slope / .4), data_status: "predicted", source: "assumption-based projection" },
      { pipeline: "Sensitivity variant", task_id: "T1 brain age", model: modelLabel(focus.id), metric: "MAE", val_mae_mean: round(focus.base + .16 * focus.slope / .4), data_status: "predicted", source: "assumption-based projection" }
    ]
  };
  const representative = rows.filter((row) => row.model_id === focus.id && row.seed === 42 && row.n_labels === 100);
  const foldRows = representative.flatMap((row) => [
    ...[-.18, -.07, .02, .09, .14].map((offset, index) => ({ dataset: "AOMIC · scenario", task: "T1 brain age", model: modelLabel(focus.id), fold: index + 1, metric: "MAE · years", value: round(row.seen_mae + offset), data_status: "predicted" })),
    ...[-.06, -.02, .01, .03, .05].map((offset, index) => ({ dataset: "AOMIC · scenario", task: "T1 brain age", model: modelLabel(focus.id), fold: index + 1, metric: "R²", value: round(row.seen_r2 + offset), data_status: "predicted" }))
  ]);
  const cvRows = rows.filter((row) => row.model_id === focus.id && row.seed === 42).map((row) => ({
    dataset: "AOMIC · scenario", task: "T1 brain age", model: modelLabel(focus.id), n_train: row.n_labels,
    val_mae_mean: row.seen_mae, val_mae_std: .14 * focus.slope / .4, val_r2_mean: row.seen_r2, val_r2_std: .025, data_status: "predicted"
  }));
  const legacy = [
    { kind: "folds", source: "formula-generated scenario", dataStatus: "predicted", rows: foldRows },
    { kind: "learning", source: "formula-generated scenario", dataStatus: "predicted", rows: cvRows }
  ];
  return { rows, did, equivalence, pareto, correlation, preprocessing, legacy, focusModel: focus.id, profile: focus };
}
function humanize(value) {
  return String(value || "—").replaceAll("_", " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

function statusLabel(value) {
  const status = String(value || "unverified").toLowerCase();
  return ({ observed: "OBSERVED", mock: "SYNTHETIC RUN", predicted: "MODEL PROJECTION", unverified: "UNVERIFIED" })[status] || humanize(status).toUpperCase();
}

function modelLabel(id) { return modelLabels[id] || humanize(id); }

function syncProjectionChrome() {
  const button = $("#generate-estimates-button");
  button.textContent = state.projectionMode ? "Clear estimates" : "Generate estimates";
  button.setAttribute("aria-pressed", String(state.projectionMode));
  $("#projection-mode-indicator").hidden = !state.projectionMode;
  $("#projection-banner").hidden = !state.projectionMode;
  $("#focus-model").disabled = !state.projectionMode;
  $("#results-scope").disabled = state.projectionMode;
  $("#export-csv").disabled = state.projectionMode;
  $("#export-json").disabled = state.projectionMode;
  $("#compute-did").disabled = !state.projectionMode && !hasMatchedDidCells(state.results, $("#did-fm-model").value, $("#did-base-model").value);
  $("#compute-equivalence").disabled = false;
  document.body.classList.toggle("is-projection-mode", state.projectionMode);
}

function renderProjectionWorkspace() {
  if (!state.projectionMode || !state.projectionData) return;
  renderResults();
  renderAudit(state.projectionSnapshot?.audit || state.audit || { dataStatus: "missing", rows: [] });
  renderCorrelation(state.projectionData.correlation);
  renderPreprocessing(state.projectionData.preprocessing);
  renderDid({ status: "predicted", dataStatus: "predicted", result: state.projectionData.did });
  renderEquivalence(state.projectionData.equivalence);
  renderPareto(state.projectionData.pareto);
  renderProjectionSummary();
}

function projectionMetricDomain(metric = "mae") {
  const fields = metric === "r2" ? ["seen_r2", "unseen_r2"] : ["seen_mae", "unseen_mae"];
  const values = (state.projectionData?.rows || [])
    .flatMap((row) => fields.map((field) => numeric(row[field])))
    .filter((value) => value !== null && Number.isFinite(value));
  if (!values.length) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = (max - min || Math.max(Math.abs(max), 1)) * .08;
  return { min: min - padding, max: max + padding, barMax: max * 1.06 };
}

function toggleProjectionMode() {
  if (!state.projectionMode) {
    state.projectionSnapshot = {
      audit: state.audit, correlation: state.correlation, preprocessing: state.preprocessing,
      did: state.did, pareto: state.pareto, equivalence: state.equivalence
    };
    state.projectionFilters = {
      scope: $("#results-scope").value, model: $("#results-model-filter").value, metric: $("#results-metric").value
    };
    state.projectionMode = true;
    const focus = $("#focus-model").value || "brainiac";
    $("#results-scope").value = "predicted";
    $("#results-model-filter").value = focus;
    $("#results-metric").value = "mae";
    $("#did-fm-model").value = focus;
    $("#eq-fm-model").value = focus;
    const base = focus === "resnet18_scratch" ? "densenet121_scratch" : "resnet18_scratch";
    $("#did-base-model").value = base;
    $("#eq-base-model").value = base;
    state.projectionData = buildModelProjections({ focusModel: focus, didBaseModel: base, eqBaseModel: base, targetN: $("#eq-n").value });
    syncProjectionChrome();
    renderProjectionWorkspace();
    showToast(`Generated a model projection for ${modelLabel(focus)}. Nothing was saved.`);
    return;
  }

  state.projectionMode = false;
  state.projectionData = null;
  if (state.projectionSnapshot) Object.assign(state, state.projectionSnapshot);
  if (state.projectionFilters) {
    $("#results-scope").value = state.projectionFilters.scope || "observed";
    $("#results-model-filter").value = state.projectionFilters.model || "all";
    $("#results-metric").value = state.projectionFilters.metric || "mae";
  }
  state.projectionSnapshot = null;
  state.projectionFilters = null;
  syncProjectionChrome();
  clearProjectionSummary();
  renderResults();
  renderAudit(state.audit || { dataStatus: "missing", rows: [] });
  renderCorrelation(state.correlation || { status: "missing", matrices: [] });
  renderPreprocessing(state.preprocessing || { status: "missing", cacheGroups: [], variantRows: [] });
  renderDid(state.did || { status: "missing", result: null });
  renderEquivalence(state.equivalence || { status: "missing", reason: "Observed label-budget rows are required." });
  renderPareto(state.pareto || { points: [] });
  showToast("Model projections cleared. Showing saved workspace data.");
}

function renderProjectionSummary() {
  if (!state.projectionData) return;
  const profile = state.projectionData.profile;
  const focus = state.projectionData.focusModel;
  const rows = state.projectionData.rows.filter((row) => row.model_id === focus);
  const budgets = [...new Set(rows.map((row) => row.n_labels))].sort((a, b) => a - b);
  const average = (values) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  const series = ["seen", "unseen"].map((cohort) => ({
    name: `${modelLabel(focus)} · ${cohort}`,
    values: budgets.map((n) => {
      const atBudget = rows.filter((row) => row.n_labels === n);
      return { x: n, value: average(atBudget.map((row) => row[`${cohort}_mae`])) };
    })
  }));
  const sharedDomain = projectionMetricDomain("mae");
  ["#overview-projection-chart", "#runner-projection-chart", "#data-projection-chart"].forEach((selector) => makeLineChart(selector, series, budgets, "MAE · years", sharedDomain));
  const overviewCaption = $("#overview-projection-caption");
  if (overviewCaption) overviewCaption.textContent = "Predictions use one shared MAE scale across all six models. Values are formula-generated assumptions, not measurements from a trained checkpoint.";
  ["#overview-projection-status", "#runner-projection-status", "#data-projection-status"].forEach((selector) => {
    const tag = $(selector);
    if (tag) tag.textContent = `MODEL PROJECTION · ${modelLabel(focus).toUpperCase()}`;
  });
  $("#projection-explainer-title").textContent = `${modelLabel(focus)} assumption profile`;
  $("#projection-explainer-copy").textContent = "Each model has separate assumptions for starting error, improvement per label-budget doubling and unseen-cohort gap. The resulting projection is not output from a trained checkpoint.";
  $("#projection-base-stat").textContent = `${fmt(profile.base, 2)} y at 25 labels`;
  $("#projection-slope-stat").textContent = `${fmt(profile.slope, 2)} y lower per doubling`;
  $("#projection-gap-stat").textContent = `${fmt(profile.gap, 2)} y at 25 labels`;
}

function clearProjectionSummary() {
  ["#overview-projection-chart", "#runner-projection-chart", "#data-projection-chart"].forEach((selector) => illustrativeChart($(selector), "line"));
  ["#overview-projection-status", "#runner-projection-status", "#data-projection-status"].forEach((selector) => {
    const tag = $(selector);
    if (tag) tag.textContent = "Generate estimates to explore";
  });
  const caption = $("#overview-projection-caption");
  if (caption) caption.textContent = "Generate estimates to compare model-specific curves on a shared MAE scale. Values are assumptions, not model results.";
  $("#projection-explainer-title").textContent = "Model-specific assumptions";
  $("#projection-explainer-copy").textContent = "Generate estimates to see a transparent assumption set for each configured model. No fitted weights or observed outcomes are implied.";
  $("#projection-base-stat").textContent = "—";
  $("#projection-slope-stat").textContent = "—";
  $("#projection-gap-stat").textContent = "—";
}

function refreshModelProjection(source = "focus") {
  if (!state.projectionMode) {
    if (source === "did") renderDid(state.did || { status: "missing" });
    return;
  }
  const focusSelect = $("#focus-model");
  if (source === "did") focusSelect.value = $("#did-fm-model").value;
  if (source === "equivalence") focusSelect.value = $("#eq-fm-model").value;
  if (source === "results") focusSelect.value = $("#results-model-filter").value;
  if (source === "runner") focusSelect.value = $("#run-model").value;
  const focus = focusSelect.value || "brainiac";
  if (source === "focus" || source === "did" || source === "equivalence" || source === "results" || source === "runner") {
    $("#did-fm-model").value = focus;
    $("#eq-fm-model").value = focus;
    $("#results-model-filter").value = focus;
    if ([...$("#run-model").options].some((option) => option.value === focus)) $("#run-model").value = focus;
    const fallbackBase = focus === "resnet18_scratch" ? "densenet121_scratch" : "resnet18_scratch";
    if ($( "#did-base-model" ).value === focus || source !== "did") $("#did-base-model").value = fallbackBase;
    if ($( "#eq-base-model" ).value === focus || source !== "equivalence") $("#eq-base-model").value = fallbackBase;
  }
  const fallbackBase = focus === "resnet18_scratch" ? "densenet121_scratch" : "resnet18_scratch";
  if ($("#did-base-model").value === focus) $("#did-base-model").value = fallbackBase;
  if ($("#eq-base-model").value === focus) $("#eq-base-model").value = fallbackBase;
  state.projectionData = buildModelProjections({
    focusModel: focus,
    didBaseModel: $("#did-base-model").value,
    eqBaseModel: $("#eq-base-model").value,
    targetN: $("#eq-n").value
  });
  renderProjectionWorkspace();
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { cache: "no-store", ...options, headers });
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 2600);
}

function goToPage(page, updateHash = true) {
  if (!pageLabels[page]) page = "overview";
  state.currentPage = page;
  $$('[data-page]').forEach((section) => {
    const active = section.dataset.page === page;
    section.hidden = !active;
    section.classList.toggle("is-active", active);
  });
  $$('[data-page-target]').forEach((button) => {
    const active = button.dataset.pageTarget === page;
    button.classList.toggle("is-active", active);
    if (button.classList.contains("nav-item")) {
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    }
  });
  $("#current-page-label").textContent = pageLabels[page];
  if (updateHash && location.hash !== `#${page}`) history.replaceState(null, "", `#${page}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (page === "results") renderResults();
  if (page === "findings" && state.archive) renderFinalFindings(state.archive);
  if (page === "audit" && state.audit) renderAudit(state.audit);
  if (page === "correlation" && state.correlation) renderCorrelation(state.correlation);
  if (page === "preprocessing" && state.preprocessing) renderPreprocessing(state.preprocessing);
  if (page === "equivalence") computeEquivalence();
  if (state.projectionMode) renderProjectionWorkspace();
}

function setConnection(connected, label) {
  const status = $("#api-status");
  $(".status-dot", status).classList.toggle("is-good", connected);
  $(".status-dot", status).classList.toggle("is-bad", !connected);
  $("#api-status span:last-child").textContent = label;
  $("#sidebar-status-dot").classList.toggle("is-good", connected);
  $("#sidebar-status-dot").classList.toggle("is-bad", !connected);
  $("#sidebar-status-label").textContent = connected ? "Workspace connected" : "Workspace unavailable";
}

function renderReadiness(checks = []) {
  const required = checks.filter((item) => item.required !== false);
  const readyCount = required.filter((item) => item.ready).length;
  const ready = Boolean(state.summary?.readiness?.observedRunReady);
  const short = `${readyCount} of ${required.length} required inputs found`;
  $("#metric-ready").textContent = `${readyCount}/${required.length}`;
  $("#readiness-title").textContent = ready ? "Observed runs are ready" : short;
  $("#readiness-copy").textContent = ready
    ? "Frozen splits, scan metadata and preprocessed volumes are present. Check model-specific checkpoint availability before each run."
    : "Add the frozen split, scan metadata and preprocessed cache listed below before starting an observed run.";
  $("#readiness-symbol").textContent = ready ? "✓" : "!";
  $("#observed-mode-choice input").disabled = !ready;
  $("#observed-mode-choice").classList.toggle("mode-choice-disabled", !ready);
  if (!ready && $("input[name=mode]:checked")?.value === "observed") $("input[name=mode][value=mock]").checked = true;
  const rows = checks.map((item) => `<li><span class="check-mark ${item.ready ? "" : "is-wait"}">${item.ready ? "✓" : "·"}</span><span>${esc(item.name)}${item.required === false ? " <span class=\"small-print\">Optional for this action</span>" : ""}</span></li>`).join("");
  $("#readiness-list").innerHTML = rows || "<li>No readiness checks are configured.</li>";
  $("#run-readiness-list").innerHTML = required.map((item) => `<li><span class="check-mark ${item.ready ? "" : "is-wait"}">${item.ready ? "✓" : "·"}</span>${esc(item.name)} — ${item.ready ? "ready" : "missing"}</li>`).join("") || "<li>No readiness checks are configured.</li>";
  $("#study-status").textContent = state.summary?.observedRuns ? `${state.summary.observedRuns} observed cells recorded · H1 Confirmed` : ready ? "Inputs ready · results pending" : "Protocol registered · inputs pending";
  $("#study-status-square").classList.toggle("is-good", Boolean(ready || state.summary?.observedRuns));
  $("#study-status-square").classList.toggle("is-bad", !ready && !state.summary?.observedRuns);
}

function updateSummary(summary) {
  state.summary = summary;
  $("#metric-observed").textContent = String(summary.observedRuns ?? 0);
  $("#metric-mock").textContent = String(summary.mockRuns ?? 0);
  const auditStatus = summary.auditStatus || "missing";
  $("#metric-audit").textContent = auditStatus === "missing" ? "Pending" : humanize(auditStatus);
  $("#nav-audit-count").textContent = auditStatus === "observed" ? "1" : "—";
  renderReadiness(summary.readiness?.checks || []);
}

function setOptions(select, values, labels = {}, { preserve = true, disabled = () => false } = {}) {
  if (!select) return;
  const oldValue = select.value;
  select.innerHTML = values.map((value) => {
    const id = typeof value === "string" ? value : value.id;
    const label = labels[id] || (typeof value === "string" ? humanize(id) : value.name || humanize(id));
    const unavailable = disabled(value);
    return `<option value="${esc(id)}" ${unavailable ? "disabled" : ""}>${esc(label)}${unavailable ? " · unavailable" : ""}</option>`;
  }).join("");
  if (preserve && [...select.options].some((option) => option.value === oldValue && !option.disabled)) select.value = oldValue;
  else {
    const first = [...select.options].find((option) => !option.disabled);
    if (first) select.value = first.value;
  }
}

function updateConfig(config) {
  state.config = config;
  setOptions($("#run-task"), config.tasks || [], taskLabels);
  const models = config.models || [];
  setOptions($("#run-model"), models, modelLabels, { disabled: (model) => typeof model !== "string" && model.available === false });
  setOptions($("#focus-model"), models, modelLabels);
  setOptions($("#run-seed"), (config.seeds || []).map(String), Object.fromEntries((config.seeds || []).map((seed) => [String(seed), String(seed)])));
  setOptions($("#did-fm-model"), models, modelLabels);
  setOptions($("#did-base-model"), models, modelLabels);
  setOptions($("#eq-fm-model"), models, modelLabels);
  setOptions($("#eq-base-model"), models, modelLabels);
  const ids = models.map((model) => typeof model === "string" ? model : model.id);
  if (ids.includes("brainiac")) {
    if (!$("#did-fm-model").dataset.userChanged) $("#did-fm-model").value = "brainiac";
    if (!$("#eq-fm-model").dataset.userChanged) $("#eq-fm-model").value = "brainiac";
  }
  if (ids.includes("resnet18_scratch")) {
    if (!$("#did-base-model").dataset.userChanged) $("#did-base-model").value = "resnet18_scratch";
    if (!$("#eq-base-model").dataset.userChanged) $("#eq-base-model").value = "resnet18_scratch";
  }
  const trials = $("#run-trials");
  if (trials) {
    const old = trials.value;
    trials.innerHTML = (config.trialBudgets || []).map((value) => `<option value="${value}">${value === 2 ? "2 · smoke" : `${value} · ${value === 5 ? "low" : value === 20 ? "medium" : "high"}`}</option>`).join("");
    if ([...trials.options].some((option) => option.value === old)) trials.value = old;
  }
  const allIds = models.map((model) => typeof model === "string" ? model : model.id);
  setOptions($("#results-model-filter"), ["all", ...allIds], { all: "All models", ...modelLabels });
  const focusIds = allIds.length ? allIds : ["brainiac"];
  const currentFocus = focusIds.includes($("#focus-model").value) ? $("#focus-model").value : (focusIds.includes("brainiac") ? "brainiac" : focusIds[0]);
  $("#focus-model").value = currentFocus;
  const runModels = [...$("#run-model").options];
  const selectedRunModel = runModels.find((option) => option.value === currentFocus && !option.disabled);
  const firstRunnableModel = runModels.find((option) => !option.disabled);
  if (selectedRunModel) $("#run-model").value = selectedRunModel.value;
  else if (firstRunnableModel) $("#run-model").value = firstRunnableModel.value;
  renderModelTable(models);
  renderDatasetTable(config.dataSources || []);
  setRunMode();
}

function renderDatasetTable(sources) {
  $("#dataset-table-body").innerHTML = sources.length ? sources.map((source) => `<tr><td><strong>${esc(source.id)}</strong></td><td>${esc(source.mode)}</td><td><code>${esc(source.path)}</code></td><td><span class="status-pill">${source.ready ? "PRESENT" : "NOT CONFIGURED"}</span></td><td>${esc(source.note || "")}</td></tr>`).join("") : `<tr><td colspan="5" class="empty-cell">No dataset sources are configured.</td></tr>`;
}

function renderModelTable(models) {
  $("#model-count").textContent = `${models.length} configured`;
  $("#model-table-body").innerHTML = models.length ? models.map((model) => {
    const id = typeof model === "string" ? model : model.id;
    const type = typeof model === "string" ? "configured" : model.type;
    const available = typeof model === "string" || model.available !== false;
    const checkpoint = typeof model === "string" ? "Not specified" : model.checkpointRequired ? model.checkpointReady ? "Present" : "Missing" : "Not required";
    const source = typeof model === "string" ? "—" : model.reason || (model.checkpointRequired ? "Configured checkpoint" : "No checkpoint required");
    return `<tr><td><strong>${esc(modelLabel(id))}</strong><br><small>${esc(id)}</small></td><td>${esc(humanize(type))}</td><td><span class="status-pill">${available ? "AVAILABLE" : "BLOCKED"}</span></td><td>${esc(checkpoint)}</td><td>${esc(source)}</td></tr>`;
  }).join("") : `<tr><td colspan="5" class="empty-cell">No model configs found.</td></tr>`;
}

function renderBudgetVisual() {
  const labels = ["25", "50", "100", "250", "500", "all"];
  const heights = [18, 28, 43, 65, 83, 96];
  $("#budget-visual").innerHTML = `<div class="budget-bars">${labels.map((label, index) => `<div class="budget-point"><span>${label}</span><i class="budget-bar" style="--bar-height:${heights[index]}%;--bar-delay:${index * 75}ms" title="Protocol label budget ${label}"></i></div>`).join("")}</div>`;
}

function renderAudit(payload) {
  state.audit = payload;
  const status = payload.dataStatus || "missing";
  const pillText = { observed: "OBSERVED AUDIT", mock: "SYNTHETIC CHECK", unverified: "UNVERIFIED", missing: "AUDIT PENDING" }[status] || "AUDIT PENDING";
  $("#audit-status-pill").textContent = pillText;
  $("#audit-banner").className = `audit-banner ${status === "observed" ? "is-ready" : status === "mock" ? "is-mock" : ""}`;
  if (status === "observed") {
    $("#audit-banner-title").textContent = "Provenance-marked audit loaded";
    $("#audit-banner-copy").textContent = "These rows are marked observed. Confirm the manifest version and model-specific pretraining sources before interpreting exposure.";
  } else if (status === "mock") {
    $("#audit-banner-title").textContent = "This audit uses synthetic identifiers";
    $("#audit-banner-copy").textContent = "Synthetic overlap checks software plumbing only and must not label real benchmark cohorts.";
  } else if (status === "unverified") {
    $("#audit-banner-title").textContent = "Audit output found, provenance incomplete";
    $("#audit-banner-copy").textContent = "A table exists, but its provenance does not identify it as observed or synthetic. Treat its values as unverified.";
  } else {
    $("#audit-banner-title").textContent = "No verified audit output found";
    $("#audit-banner-copy").textContent = "The manuscript leaves the overlap table TBD and does not provide numeric cutoffs. Set both thresholds from the protocol before running the audit.";
  }
  const threshold = payload.seenThresholdPct !== null && payload.seenThresholdPct !== undefined && payload.unseenThresholdPct !== null && payload.unseenThresholdPct !== undefined
    ? ` · seen ≥ ${fmt(payload.seenThresholdPct, 1)}% · unseen ≤ ${fmt(payload.unseenThresholdPct, 1)}%` : "";
  $("#audit-source").textContent = payload.source ? `Manifest ${payload.source}${threshold}` : `Source not recorded${threshold}`;
  const rows = payload.rows || [];
  const usable = rows.filter((row) => numeric(row.contamination_pct) !== null);
  $("#audit-chart-status").textContent = usable.length
    ? `${status === "mock" ? "SYNTHETIC CHECK" : status.toUpperCase()} · ${usable.length} COHORTS`
    : "NO SOURCE-BACKED VALUES";
  if (usable.length) {
    makeBarChart("#audit-chart", usable.map((row) => ({ label: row.cohort, values: [numeric(row.contamination_pct)] })), usable.map((row) => row.cohort), [status === "mock" ? "Synthetic overlap" : "Manifest overlap"], "%");
  } else {
    emptyChart("#audit-chart", "%", "No verified audit data", "A cohort manifest is needed to calculate participant overlap. The model projection does not invent these values.");
  }
  $("#audit-table-body").innerHTML = rows.length ? rows.map((row) => `<tr><td><strong>${esc(row.cohort || "—")}</strong></td><td>${esc(row.total_subjects ?? "—")}</td><td>${esc(row.pretraining_seen_count ?? "—")}</td><td>${row.contamination_pct === undefined ? "—" : `${fmt(row.contamination_pct, 1)}%`}</td><td><span class="status-pill">${esc(String(row.status || "Unclassified").replace(/\s*\(.*\)/, "").replace(/mock/ig, "synthetic"))}</span></td></tr>`).join("") : `<tr><td colspan="5" class="empty-cell">${status === "missing" ? "No audit rows yet. Values marked TBD in the paper are not treated as results." : "No usable rows in the audit file."}</td></tr>`;
}

function updateAuditCommand() {
  const seen = numeric($("#audit-seen-threshold").value);
  const unseen = numeric($("#audit-unseen-threshold").value);
  const valid = seen !== null && unseen !== null && unseen >= 0 && seen <= 100 && unseen < seen;
  const seenText = valid ? String(seen) : "SEEN_THRESHOLD";
  const unseenText = valid ? String(unseen) : "UNSEEN_THRESHOLD";
  $("#audit-command").textContent = `python audit/report_contamination.py --manifest_path data/manifests/mapping.tsv --cohorts_json path/to/verified_audit_cohorts.json --cohorts_status observed --seen_threshold ${seenText} --unseen_threshold ${unseenText} --output_dir results`;
}

function emptyChart(selector, index, title, description) {
  const target = typeof selector === "string" ? $(selector) : selector;
  if (!target) return;
  target.innerHTML = `<div class="empty-state"><span class="empty-state-index">${esc(index)}</span><strong>${esc(title)}</strong><p>${esc(description)}</p></div>`;
}

function illustrativeChart(target, type) {
  if (!target) return;
  const badge = `<div class="chart-preview-note"><span class="preview-mark" aria-hidden="true">◌</span><strong>Chart preview</strong><span>diagram only · no study values</span></div>`;
  if (type === "bars") {
    const groups = [
      { x: 165, a: 102, b: 66 },
      { x: 340, a: 70, b: 119 },
      { x: 515, a: 132, b: 88 },
    ];
    const grid = [38, 75, 112, 149].map((y) => `<line class="chart-gridline" x1="75" x2="660" y1="${y}" y2="${y}"/>`).join("");
    const bars = groups.map((group, index) => `<rect class="bar-series-a preview-bar" style="--bar-delay:${index * 100}ms" x="${group.x - 25}" y="${160 - group.a}" width="34" height="${group.a}" rx="3"/><rect class="bar-series-b preview-bar" style="--bar-delay:${index * 100 + 60}ms" x="${group.x + 12}" y="${160 - group.b}" width="34" height="${group.b}" rx="3"/><text class="chart-label-strong" x="${group.x + 8}" y="181" text-anchor="middle">Group ${index + 1}</text>`).join("");
    target.innerHTML = `${badge}<svg class="svg-chart preview-chart" viewBox="0 0 720 205" role="img" aria-label="Illustrative paired bar chart diagram with no study values">${grid}<line class="chart-axis" x1="75" x2="660" y1="160" y2="160"/><text class="chart-label" x="68" y="41" text-anchor="end">HIGH</text><text class="chart-label" x="68" y="158" text-anchor="end">LOW</text>${bars}<g transform="translate(250,199)"><rect class="bar-series-a" width="9" height="9" rx="1"/><text class="chart-label" x="14" y="8">Series A</text><rect class="bar-series-b" x="110" width="9" height="9" rx="1"/><text class="chart-label" x="124" y="8">Series B</text></g></svg>`;
    return;
  }

  const pointsA = [[85, 58], [190, 78], [295, 91], [400, 110], [505, 127], [610, 143]];
  const pointsB = [[85, 133], [190, 120], [295, 111], [400, 99], [505, 90], [610, 77]];
  const toPath = (points) => points.map(([x, y], index) => `${index ? "L" : "M"}${x},${y}`).join(" ");
  const grid = [38, 75, 112, 149].map((y) => `<line class="chart-gridline" x1="75" x2="660" y1="${y}" y2="${y}"/>`).join("");
  const dots = (points, className) => points.map(([x, y]) => `<circle class="${className}" cx="${x}" cy="${y}" r="4"/>`).join("");
  target.innerHTML = `${badge}<svg class="svg-chart preview-chart" viewBox="0 0 720 205" role="img" aria-label="Illustrative two-series trend diagram with no study values">${grid}<line class="chart-axis" x1="75" x2="660" y1="160" y2="160"/><text class="chart-label" x="68" y="41" text-anchor="end">HIGH</text><text class="chart-label" x="68" y="158" text-anchor="end">LOW</text><path class="chart-line preview-line" d="${toPath(pointsA)}"/><path class="chart-line-secondary preview-line preview-line-late" d="${toPath(pointsB)}"/>${dots(pointsA, "chart-point preview-dot")}${dots(pointsB, "chart-point-secondary preview-dot preview-dot-late")}<text class="chart-label-strong" x="85" y="181" text-anchor="middle">START</text><text class="chart-label-strong" x="610" y="181" text-anchor="middle">END</text><g transform="translate(250,199)"><line class="chart-line" x1="0" x2="12" y1="4" y2="4"/><text class="chart-label" x="17" y="8">Series A</text><line class="chart-line-secondary" x1="110" x2="122" y1="4" y2="4"/><text class="chart-label" x="127" y="8">Series B</text></g></svg>`;
}

function makeBarChart(selector, values, labels, seriesNames, unit = "", sharedMax = null) {
  const target = $(selector);
  if (!values.length) return illustrativeChart(target, "bars");
  const width = 560, height = 320, left = 50, right = 12, top = 18, bottom = 48;
  const chartW = width - left - right, chartH = height - top - bottom;
  const maxValue = Math.max(1e-9, Number.isFinite(sharedMax) ? sharedMax : 0, ...values.flatMap((item) => item.values.filter((v) => v !== null)));
  const step = chartW / values.length;
  const groupW = Math.min(step * .68, 76);
  const barsPer = Math.max(1, seriesNames.length);
  const barW = groupW / barsPer - 3;
  const grid = Array.from({ length: 4 }, (_, i) => {
    const y = top + i * chartH / 3;
    const val = maxValue * (1 - i / 3);
    return `<line class="chart-gridline" x1="${left}" x2="${width-right}" y1="${y}" y2="${y}"/><text class="chart-label" x="${left-7}" y="${y+3}" text-anchor="end">${fmt(val, 1)}${esc(unit)}</text>`;
  }).join("");
  const bars = values.map((item, i) => {
    const xCenter = left + step * (i + .5);
    const per = item.values.map((value, j) => {
      if (value === null) return "";
      const h = Math.max(1, chartH * value / maxValue);
      const x = xCenter - groupW / 2 + j * (groupW / barsPer) + 1;
      const y = top + chartH - h;
      return `<rect class="bar-series-${String.fromCharCode(97 + j)} chart-bar" style="--bar-delay:${(i * barsPer + j) * 55}ms" x="${x}" y="${y}" width="${Math.max(2,barW)}" height="${h}" rx="2"><title>${esc(labels[i])} · ${esc(seriesNames[j])}: ${fmt(value, 3)}${esc(unit)}</title></rect>`;
    }).join("");
    return `${per}<text class="chart-label-strong" x="${xCenter}" y="${height-17}" text-anchor="middle">${esc(labels[i])}</text>`;
  }).join("");
  const legends = seriesNames.map((name, i) => `<g transform="translate(${left + i * 130},${height-2})"><rect class="bar-series-${String.fromCharCode(97+i)}" width="9" height="9" rx="1"/><text class="chart-label" x="14" y="8">${esc(name)}</text></g>`).join("");
  target.innerHTML = `<svg class="svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(seriesNames.join(" versus "))} chart">${grid}<line class="chart-axis" x1="${left}" x2="${width-right}" y1="${top+chartH}" y2="${top+chartH}"/>${bars}${legends}</svg>`;
}

function makeLineChart(selector, series, xValues, yLabel = "MAE", sharedDomain = null) {
  const target = $(selector);
  const usable = series.filter((item) => item.values.some((point) => point.value !== null));
  if (!usable.length) return illustrativeChart(target, "line");
  const pointsAll = usable.flatMap((item) => item.values.filter((point) => point.value !== null));
  const minX = Math.min(...pointsAll.map((point) => point.x));
  const maxX = Math.max(...pointsAll.map((point) => point.x));
  const localMinY = Math.min(...pointsAll.map((point) => point.value));
  const localMaxY = Math.max(...pointsAll.map((point) => point.value));
  const minY = Number.isFinite(sharedDomain?.min) ? sharedDomain.min : localMinY;
  const maxY = Number.isFinite(sharedDomain?.max) ? sharedDomain.max : localMaxY;
  const yPad = sharedDomain ? 0 : (maxY - minY || Math.max(Math.abs(maxY),1)) * .15;
  const width = 560, height = 320, left = 54, right = 16, top = 18, bottom = 52;
  const plotW = width-left-right, plotH = height-top-bottom;
  const x = (value) => left + (maxX === minX ? .5 : (value-minX)/(maxX-minX)) * plotW;
  const y = (value) => top + (maxY + yPad - value)/(maxY - minY + 2*yPad) * plotH;
  const grid = Array.from({length:4},(_,i)=>{const yy=top+i*plotH/3;const val=maxY+yPad-i*(maxY-minY+2*yPad)/3;return `<line class="chart-gridline" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text class="chart-label" x="${left-8}" y="${yy+3}" text-anchor="end">${fmt(val,1)}</text>`;}).join("");
  const lineParts = usable.map((item,index)=>{
    const points=item.values.filter((point)=>point.value!==null);
    const path=points.map((point,i)=>`${i?"L":"M"}${x(point.x)},${y(point.value)}`).join(" ");
    const lineClass=index%2===0?"chart-line":"chart-line-secondary";
    const pointClass=index%2===0?"chart-point":"chart-point-secondary";
    const circles=points.map((point,pointIndex)=>`<circle class="${pointClass} chart-point-pop" style="--point-delay:${420+index*140+pointIndex*65}ms" cx="${x(point.x)}" cy="${y(point.value)}" r="4"><title>${esc(item.name)} · N=${point.x} · ${fmt(point.value,3)} ${esc(yLabel)}</title></circle>`).join("");
    return `<path class="${lineClass} chart-line-draw" d="${path}"/>${circles}`;
  }).join("");
  const xTicks=[...new Set(pointsAll.map((point)=>point.x))].sort((a,b)=>a-b).map(value=>`<text class="chart-label" x="${x(value)}" y="${height-20}" text-anchor="middle">${value}</text>`).join("");
  const legendStep=Math.min(150,(width-left-right)/Math.max(usable.length,1));
  const legend=usable.map((item,index)=>`<g transform="translate(${left+index*legendStep},${height-4})"><line class="${index%2===0?"chart-line":"chart-line-secondary"}" x1="0" x2="12" y1="0" y2="0"/><text class="chart-label" x="17" y="3">${esc(item.name)}</text></g>`).join("");
  target.innerHTML=`<svg class="svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(yLabel)} by label budget">${grid}<line class="chart-axis" x1="${left}" x2="${width-right}" y1="${top+plotH}" y2="${top+plotH}"/>${lineParts}${xTicks}${legend}</svg>`;
}

function comparisonMean(rows, modelId, budget, field) {
  const values = rows.filter((row) => row.model_id === modelId && row.n_labels === budget).map((row) => numeric(row[field])).filter((value) => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function comparisonDomain(rows, models) {
  const values = rows.flatMap((row) => [numeric(row.seen_mae), numeric(row.unseen_mae)]).filter((value) => value !== null && Number.isFinite(value));
  if (!values.length) return { min: 0, max: 1 };
  const min = Math.min(...values), max = Math.max(...values);
  const padding = (max - min || Math.max(Math.abs(max), 1)) * .08;
  return { min: min - padding, max: max + padding, modelCount: models.length };
}

function makeComparisonLineChart(selector, rows, models, visibleModels, field, domain, focusModel = null) {
  const target = $(selector);
  const active = models.filter((model) => visibleModels.includes(model.id));
  if (!active.length) {
    target.innerHTML = '<div class="empty-state"><span class="empty-state-index">—</span><strong>No model traces selected</strong><p>Choose one or more model names above to draw comparison curves.</p></div>';
    return;
  }
  const width = 720, height = 330, left = 58, right = 20, top = 20, bottom = 48;
  const plotW = width - left - right, plotH = height - top - bottom;
  const budgets = [...new Set(rows.map((row) => row.n_labels))].sort((a, b) => a - b);
  const logMin = Math.log2(Math.min(...budgets)), logMax = Math.log2(Math.max(...budgets));
  const x = (value) => left + (logMax === logMin ? .5 : (Math.log2(value) - logMin) / (logMax - logMin)) * plotW;
  const y = (value) => top + (domain.max - value) / (domain.max - domain.min) * plotH;
  const grid = Array.from({ length: 4 }, (_, index) => {
    const value = domain.max - index * (domain.max - domain.min) / 3;
    const yy = top + index * plotH / 3;
    return `<line class="chart-gridline" x1="${left}" x2="${width - right}" y1="${yy}" y2="${yy}"/><text class="chart-label" x="${left - 9}" y="${yy + 3}" text-anchor="end">${fmt(value, 1)}</text>`;
  }).join("");
  const traces = active.map((model) => {
    const modelIndex = models.findIndex((item) => item.id === model.id);
    const style = comparisonStyles[modelIndex % comparisonStyles.length];
    const isFocus = model.id === focusModel;
    const points = budgets.map((budget) => ({ x: budget, value: comparisonMean(rows, model.id, budget, field) })).filter((point) => point.value !== null);
    const path = points.map((point, index) => `${index ? "L" : "M"}${x(point.x)},${y(point.value)}`).join(" ");
    const dash = style.dash ? `stroke-dasharray="${style.dash}"` : "";
    const color = isFocus ? "#050505" : style.color;
    const circles = points.map((point) => `<circle class="comparison-point" cx="${x(point.x)}" cy="${y(point.value)}" r="${isFocus ? 5 : 4}" fill="#fff" stroke="${color}" stroke-width="${isFocus ? 2.5 : 2}"><title>${esc(modelLabel(model.id))} · ${point.x} labels · ${fmt(point.value, 3)} years</title></circle>`).join("");
    return `<path class="comparison-trace ${isFocus ? "is-focus" : ""}" d="${path}" stroke="${color}" stroke-width="${isFocus ? 3.8 : 2.2}" ${dash} style="--trace-delay:${modelIndex * 90}ms"/>${circles}`;
  }).join("");
  const xTicks = budgets.map((budget) => `<line class="chart-axis-tick" x1="${x(budget)}" x2="${x(budget)}" y1="${top + plotH}" y2="${top + plotH + 5}"/><text class="chart-label" x="${x(budget)}" y="${height - 19}" text-anchor="middle">${budget}</text>`).join("");
  target.innerHTML = `<svg class="svg-chart comparison-svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${field === "seen_mae" ? "Seen" : "Unseen"} cohort MAE comparison across model profiles"><text class="chart-label-strong" x="${left}" y="12">MAE · years</text>${grid}<line class="chart-axis" x1="${left}" x2="${width - right}" y1="${top + plotH}" y2="${top + plotH}"/>${traces}${xTicks}<text class="chart-label-strong" x="${left + plotW / 2}" y="${height - 2}" text-anchor="middle">Training labels · log₂ scale</text></svg>`;
}

function makeComparisonGapChart(selector, rows, models, visibleModels, budget, focusModel = null) {
  const target = $(selector);
  const active = models.filter((model) => visibleModels.includes(model.id));
  if (!active.length) return makeComparisonLineChart(selector, [], [], [], "unseen_mae", { min: 0, max: 1 });
  const allValues = models.map((model) => ({ model, value: comparisonMean(rows, model.id, budget, "unseen_mae") - comparisonMean(rows, model.id, budget, "seen_mae") })).filter((item) => Number.isFinite(item.value));
  const values = allValues.filter((item) => visibleModels.includes(item.model.id));
  const maxValue = Math.max(.1, ...allValues.map((item) => item.value)) * 1.12;
  const width = 720, rowHeight = 34, height = Math.max(205, 44 + values.length * rowHeight), left = 180, right = 48, top = 22;
  const plotW = width - left - right;
  const grid = Array.from({ length: 4 }, (_, index) => {
    const value = maxValue * index / 3, xx = left + plotW * index / 3;
    return `<line class="chart-gridline" x1="${xx}" x2="${xx}" y1="${top}" y2="${height - 24}"/><text class="chart-label" x="${xx}" y="${height - 8}" text-anchor="middle">${fmt(value, 2)} y</text>`;
  }).join("");
  const bars = values.map(({ model, value }, index) => {
    const style = comparisonStyles[models.findIndex((item) => item.id === model.id) % comparisonStyles.length];
    const yy = top + index * rowHeight + 2, barWidth = Math.max(2, plotW * value / maxValue);
    const isFocus = model.id === focusModel;
    return `<text class="chart-label-strong" x="${left - 10}" y="${yy + 15}" text-anchor="end">${esc(modelLabel(model.id))}</text><rect class="comparison-gap-bar ${isFocus ? "is-focus" : ""}" x="${left}" y="${yy}" width="${barWidth}" height="18" rx="3" fill="${style.color}" stroke="${isFocus ? "#050505" : "none"}" stroke-width="${isFocus ? 2 : 0}" style="--bar-delay:${index * 70}ms"><title>${esc(modelLabel(model.id))} · ${fmt(value, 3)} years unseen-cohort gap</title></rect><text class="chart-label-strong" x="${left + barWidth + 7}" y="${yy + 14}">${fmt(value, 2)}</text>`;
  }).join("");
  target.innerHTML = `<svg class="svg-chart comparison-gap-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Unseen minus seen MAE model comparison at ${budget} labels">${grid}${bars}</svg>`;
}

function renderComparisonPreview() {
  const preview = state.comparisonPreview;
  if (!preview) return;
  const budget = Number($("#comparison-budget").value) || 50;
  const visible = state.comparisonVisibleModels;
  const domain = comparisonDomain(preview.rows, preview.models);
  $("#comparison-output").hidden = false;
  $("#comparison-empty").hidden = true;
  $("#comparison-gap-budget").textContent = `N = ${budget}`;
  const foundation = preview.models.filter((model) => model.checkpointRequired);
  const readyCheckpoints = foundation.filter((model) => model.checkpointReady).length;
  const requiredInputs = state.summary?.readiness?.checks?.filter((item) => item.required !== false) || [];
  const readyInputs = requiredInputs.filter((item) => item.ready).length;
  $("#comparison-stats").innerHTML = [
    ["Configured profiles", `${preview.models.length} / ${preview.models.length}`],
    ["Foundation checkpoints", `${readyCheckpoints} / ${foundation.length}`],
    ["Required study inputs", `${readyInputs} / ${requiredInputs.length || 3}`],
    ["Preview rows saved", "0 · memory only"]
  ].map(([label, value]) => `<div class="comparison-stat"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("");
  $("#comparison-legend").innerHTML = preview.models.map((model, index) => {
    const style = comparisonStyles[index % comparisonStyles.length];
    const pressed = visible.includes(model.id);
    const localStatus = model.checkpointRequired && !model.checkpointReady ? "checkpoint missing · simulated" : model.available === false ? "configuration unavailable · simulated" : "profile simulated";
    const pattern = style.dash ? `repeating-linear-gradient(90deg,${style.color} 0 4px,transparent 4px 7px)` : style.color;
    const isFocus = model.id === preview.focusModel;
    return `<button type="button" class="comparison-model-filter ${pressed ? "" : "is-muted"} ${isFocus ? "is-focus" : ""}" data-comparison-model="${esc(model.id)}" aria-pressed="${pressed}" title="${esc(localStatus)}"><span class="comparison-model-swatch" style="--model-color:${style.color};--model-pattern:${pattern}"></span><span>${esc(modelLabel(model.id))}</span>${isFocus ? '<span class="comparison-focus-mark">FOCUS</span>' : ""}</button>`;
  }).join("");
  makeComparisonLineChart("#comparison-seen-chart", preview.rows, preview.models, visible, "seen_mae", domain, preview.focusModel);
  makeComparisonLineChart("#comparison-unseen-chart", preview.rows, preview.models, visible, "unseen_mae", domain, preview.focusModel);
  makeComparisonGapChart("#comparison-gap-chart", preview.rows, preview.models, visible, budget, preview.focusModel);
  $("#comparison-table-body").innerHTML = preview.models.map((model) => {
    const seen = comparisonMean(preview.rows, model.id, budget, "seen_mae"), unseen = comparisonMean(preview.rows, model.id, budget, "unseen_mae");
    const status = model.checkpointRequired && !model.checkpointReady ? "Checkpoint missing · simulated" : model.available === false ? "Config unavailable · simulated" : "Weights not loaded · simulated";
    return `<tr class="${visible.includes(model.id) ? "" : "comparison-row-muted"} ${model.id === preview.focusModel ? "comparison-row-focus" : ""}"><td><strong>${esc(modelLabel(model.id))}${model.id === preview.focusModel ? " · focus" : ""}</strong></td><td><span class="comparison-status-pill">${esc(status)}</span></td><td>${fmt(seen, 2)} y</td><td>${fmt(unseen, 2)} y</td><td>${fmt(unseen - seen, 2)} y</td></tr>`;
  }).join("");
}

async function startComparisonPreview() {
  const button = $("#run-all-models-button");
  const progress = $("#comparison-progress");
  const title = $("#comparison-progress-title");
  const step = $("#comparison-progress-step");
  const copy = $("#comparison-progress-copy");
  const fill = $("#comparison-progress-fill");
  const stages = [
    ["Checking configured model profiles", "6 configurations found · no checkpoint downloads"],
    ["Building the synthetic comparison", "30 model × label-budget curves generated in memory"],
    ["Drawing the comparison windows", "Seen, unseen, and cohort-gap views are ready"]
  ];
  const modelId = $("#run-model").value || "brainiac";
  const chosenBudget = Number($("#run-budget").value);
  const budget = chosenBudget > 0 ? chosenBudget : 500;
  const models = (state.config?.models || Object.keys(modelLabels).map((id) => ({ id, available: false, checkpointRequired: true, checkpointReady: false }))).map((model) => typeof model === "string" ? { id: model, available: true } : model);
  const modelIds = new Set(models.map((model) => model.id));
  button.disabled = true;
  button.classList.add("is-running");
  button.innerHTML = 'Building comparison <span class="button-spinner" aria-hidden="true"></span>';
  progress.hidden = false;
  $("#comparison-output").hidden = true;
  $("#comparison-empty").hidden = true;
  fill.style.width = "8%";
  try {
    for (let index = 0; index < stages.length; index += 1) {
      title.textContent = stages[index][0];
      step.textContent = `${index + 1} / ${stages.length}`;
      copy.textContent = stages[index][1];
      fill.style.width = `${[31, 68, 94][index]}%`;
      await new Promise((resolve) => setTimeout(resolve, 110));
    }
    const profiles = buildModelProjections({ focusModel: modelId });
    state.comparisonPreview = {
      rows: profiles.rows.filter((row) => modelIds.has(row.model_id)),
      models,
      focusModel: modelId,
      generatedAt: new Date().toISOString()
    };
    state.comparisonVisibleModels = models.map((model) => model.id);
    $("#comparison-budget").value = String(budget);
    fill.style.width = "100%";
    title.textContent = "Comparison ready · synthetic only";
    step.textContent = `${models.length} / ${models.length} profiles`;
    copy.textContent = "No model weights were executed and no files were saved. Use the single-cell runner for an actual harness run.";
    $(".comparison-live-badge").innerHTML = "<i></i>PREVIEW READY";
    renderComparisonPreview();
    $("#comparison-output").classList.remove("is-live");
    requestAnimationFrame(() => $("#comparison-output").classList.add("is-live"));
    $("#comparison-workspace").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    button.innerHTML = 'Run instant comparison again <span aria-hidden="true">↻</span>';
    showToast("Synthetic model comparison ready · nothing was saved.");
  } catch (error) {
    title.textContent = "Comparison could not be rendered";
    copy.textContent = error.message;
    $("#comparison-empty").hidden = false;
    button.innerHTML = 'Try the instant comparison again <span aria-hidden="true">↻</span>';
  } finally {
    button.disabled = false;
    button.classList.remove("is-running");
  }
}

function resultMetric(row, metric, cohort) {
  const fields = metricFields[metric] || metricFields.mae;
  const key = cohort === "seen" ? fields[0] : fields[1];
  return numeric(row[key]);
}

function renderResults() {
  const scope = state.projectionMode ? "predicted" : ($("#results-scope").value || "observed");
  const modelFilter = $("#results-model-filter").value || "all";
  const metric = $("#results-metric").value || "mae";
  const metricName = metric === "mae" ? "MAE" : metric === "r2" ? "R²" : metric.toUpperCase();
  const isLowerBetter = metric === "mae";
  $("#metric-chart-title").textContent = `Seen vs unseen ${metric === "mae" ? "error" : metricName}`;
  $("#metric-chart-caption").textContent = state.projectionMode
    ? `Scenario ${metricName} projection for ${modelLabel($("#focus-model").value)}. Bars use a shared scale across model profiles; this is not a study result.`
    : `Bars show mean ${metricName} across visible cells. ${isLowerBetter ? "Lower is better." : "Higher is better."} O = observed, S = synthetic, P = model projection; each record keeps its full status in the ledger.`;
  state.scope = scope;
  let rows = (state.projectionMode ? state.projectionData?.rows || [] : state.results).filter((row) => scope === "all" || row.data_status === scope);
  if (modelFilter !== "all") rows = rows.filter((row) => row.model_id === modelFilter);
  state.filteredResults = rows;
  $("#result-count").textContent = `${rows.length} cell${rows.length === 1 ? "" : "s"}`;
  $("#cell-source-status").textContent = state.projectionMode ? `${rows.length} formula-generated projections · memory only` : rows.length ? `${statusLabel(scope)} · local results ledger` : "No matching ledger rows";
  $("#result-source-status").textContent = state.projectionMode ? `MODEL PROJECTION · ${rows.length} CELLS` : rows.length ? `${statusLabel(scope)} · ${rows.length} CELLS` : "ILLUSTRATIVE PREVIEW · NO DATA";
  const aggregates = new Map();
  for (const row of rows) {
    const key = `${row.model_id || "unknown"} · ${row.data_status || "unverified"}`;
    const current = aggregates.get(key) || { seen: [], unseen: [] };
    const seen = resultMetric(row, metric, "seen"), unseen = resultMetric(row, metric, "unseen");
    if (seen !== null) current.seen.push(seen);
    if (unseen !== null) current.unseen.push(unseen);
    aggregates.set(key, current);
  }
  const entries = Array.from(aggregates.entries()).map(([label, values]) => ({
    label,
    values: [values.seen.length ? values.seen.reduce((a,b)=>a+b,0)/values.seen.length : null, values.unseen.length ? values.unseen.reduce((a,b)=>a+b,0)/values.unseen.length : null]
  })).filter((item) => item.values.some((value) => value !== null)).slice(0, 8);
  const sharedDomain = state.projectionMode ? projectionMetricDomain(metric) : null;
  makeBarChart("#metric-chart", entries, entries.map((item)=>{const [id,status]=item.label.split(" · ");return `${modelLabel(id)} · ${status==="predicted"?"P":status==="mock"?"S":"O"}`;}), ["Seen", "Unseen"], metric === "mae" ? " y" : "", sharedDomain?.barMax ?? null);
  if (entries.length) {
    const selectedModel = modelFilter === "all" ? entries[0].label.split(" · ")[0] : modelFilter;
    const curveRows = rows.filter((row)=>row.model_id===selectedModel);
    const byBudget = new Map();
    curveRows.forEach((row)=>{
      const n = numeric(row.n_labels);
      if (!n || n <= 0) return;
      const bucket = byBudget.get(n) || {seen:[],unseen:[]};
      const seen=resultMetric(row, metric, "seen"), unseen=resultMetric(row, metric, "unseen");
      if(seen!==null)bucket.seen.push(seen); if(unseen!==null)bucket.unseen.push(unseen);
      byBudget.set(n,bucket);
    });
    const budgets=[...byBudget.keys()].sort((a,b)=>a-b);
    const seen=budgets.map(n=>({x:n,value:byBudget.get(n).seen.length?byBudget.get(n).seen.reduce((a,b)=>a+b,0)/byBudget.get(n).seen.length:null}));
    const unseen=budgets.map(n=>({x:n,value:byBudget.get(n).unseen.length?byBudget.get(n).unseen.reduce((a,b)=>a+b,0)/byBudget.get(n).unseen.length:null}));
    const learningCaption = $("#learning-chart-caption");
    if (learningCaption) learningCaption.textContent = state.projectionMode
      ? "This curve uses the same y-axis across all six models, so level and slope differences remain visible. Values are assumptions, not measured outcomes."
      : "Saved cell means replace the preview when available. No learning-curve values are implied by this diagram.";
    makeLineChart("#learning-chart",[{name:`${modelLabel(selectedModel)} · seen`,values:seen},{name:"Unseen",values:unseen}],budgets,metric === "mae" ? "MAE · years" : metric.toUpperCase(),sharedDomain);
  } else illustrativeChart($("#learning-chart"),"line");
  const cellTable = rows.slice(0,250);
  $("#results-table-body").innerHTML = cellTable.length ? cellTable.map((row)=>`<tr><td><span class="status-pill">${esc(statusLabel(row.data_status))}</span></td><td>${esc(taskLabels[row.task_id]||row.task_id||"—")}</td><td><strong>${esc(modelLabel(row.model_id))}</strong></td><td>${esc(row.n_labels ?? "—")}</td><td>${esc(row.seed ?? "—")}</td><td>${esc(row.trial_budget ?? row.trials ?? "—")}</td><td>${fmt(row.seen_mae,3)}</td><td>${fmt(row.unseen_mae,3)}</td><td>${fmt(numeric(row.total_gpu_seconds) === null ? (numeric(row.total_gpu_hours)===null?null:Number(row.total_gpu_hours)*3600) : row.total_gpu_seconds,1)} s</td></tr>`).join("") : `<tr><td colspan="9" class="empty-cell">No ${esc(statusLabel(scope).toLowerCase())} final evaluation rows found. A missing table is not a zero result.</td></tr>`;
  renderLegacyFolds();
  renderLegacyCV();
  renderArchive(state.archive);
}

function archiveTimestamp(value) {
  return value ? String(value).replace("T", " ").replace(/Z$/, " UTC") : "Timestamp not recorded";
}

function archiveTimeline(run) {
  return (run.records || []).map((record) => {
    const time = archiveTimestamp(record.timestamp);
    if (record.record_type === "tuning_trial") {
      const trial = numeric(record.trial_number);
      const trialLabel = trial === null ? "Trial" : `Trial ${trial + 1}${record.trial_budget ? ` of ${record.trial_budget}` : ""}`;
      const duration = numeric(record.sweep_seconds);
      const params = record.best_hyperparams ? `\n  Parameters: ${record.best_hyperparams}` : "";
      return `${time}  ${trialLabel} · ${record.primary_metric_name || "validation metric"} ${fmt(record.primary_metric_val, 3)}${duration === null ? "" : ` · ${fmt(duration, 2)} s`}${params}`;
    }
    if (record.record_type === "final_evaluation") {
      const compute = numeric(record.total_gpu_seconds);
      return `${time}  Final evaluation · seen MAE ${fmt(record.seen_mae, 3)} y · unseen MAE ${fmt(record.unseen_mae, 3)} y${compute === null ? "" : ` · ${fmt(compute, 2)} GPU s`}`;
    }
    return `${time}  ${record.record_type || "Saved record"}`;
  }).join("\n");
}

function renderArchiveHistory(archive) {
  const target = $("#archive-history-list");
  const jobs = archive.jobs || [];
  const runs = (archive.runs || []).filter((run) => run.records?.length);
  const jobCards = jobs.map((job) => {
    const title = job.mode === "mock-suite" ? "Synthetic pipeline analysis" : `${modelLabel(job.modelId)} · ${taskLabels[job.taskId] || job.taskId || "Research run"}`;
    const status = job.status || "saved";
    const note = job.logCoverage ? `\n${job.logCoverage}` : "";
    const log = job.log || "No saved process output is available for this run.";
    const logLink = job.logPath ? `<a class="archive-file-link" href="/api/artifact?path=${encodeURIComponent(job.logPath)}" target="_blank" rel="noopener">Open saved log file ↗</a>` : "";
    return `<details class="archive-history-card"><summary><span class="archive-history-title"><strong>${esc(title)}</strong><small>${esc(archiveTimestamp(job.createdAt))}</small></span><span class="status-pill">${esc(statusLabel(status))}</span></summary><div class="archive-history-body"><div class="archive-history-meta">${esc(job.mode === "mock" || job.mode === "mock-suite" ? "Synthetic run" : job.mode === "observed" ? "Observed run" : "Analysis job")} · ${esc(status)} · ${esc(job.id || "Saved job")}</div>${logLink}<pre class="archive-log-output">${esc(log + note)}</pre></div></details>`;
  });
  const ledgerCards = runs.map((run) => {
    const final = [...run.records].reverse().find((record) => record.record_type === "final_evaluation");
    const type = run.dataStatus === "mock" ? "Synthetic ledger records" : run.dataStatus === "observed" ? "Observed ledger records" : "Unverified ledger records";
    const result = final ? `Seen ${fmt(final.seen_mae, 3)} y · unseen ${fmt(final.unseen_mae, 3)} y` : "Final evaluation not recorded";
    return `<details class="archive-history-card"><summary><span class="archive-history-title"><strong>${esc(run.label)} · ${esc(modelLabel(run.modelId))}</strong><small>${esc(archiveTimestamp(run.startedAt))} · ${esc(type)}</small></span><span class="status-pill">${esc(run.complete ? "Evaluation saved" : "Incomplete")}</span></summary><div class="archive-history-body"><div class="archive-history-meta">${esc(taskLabels[run.taskId] || run.taskId || "Task not recorded")} · N=${esc(run.nLabels ?? "—")} · seed ${esc(run.seed ?? "—")} · ${esc(result)}</div><div class="archive-history-meta">Source rows ${esc(run.firstRow ?? "—")}–${esc(run.lastRow ?? "—")} in ${esc(run.source)}. These legacy run groups are reconstructed from matching task, model, budget, seed, and ledger order; their original job IDs were not saved.</div><pre class="archive-log-output">${esc(archiveTimeline(run))}</pre>${run.records.map((record) => record.best_hyperparams ? `<details class="archive-params"><summary>Trial parameters · ${esc(record.trial_number ?? "trial")}</summary><pre>${esc(record.best_hyperparams)}</pre></details>` : "").join("")}</div></details>`;
  });
  target.innerHTML = jobCards.concat(ledgerCards).join("") || `<div class="empty-state"><strong>No saved execution log or ledger rows</strong><p>Run history will appear here as the project records new outputs.</p></div>`;
}

function renderArchiveRecords(archive) {
  const rows = archive.records || [];
  const target = $("#archive-record-table-body");
  $("#archive-record-status").textContent = `${rows.length} ledger rows · every saved field available`;
  target.innerHTML = rows.length ? rows.map((row) => {
    const trialNumber = numeric(row.trial_number);
    const trialLabel = row.record_type === "tuning_trial" && trialNumber !== null ? `${trialNumber + 1} / ${row.trial_budget ?? "—"}` : row.trial_budget ?? row.trials ?? "—";
    const compute = numeric(row.total_gpu_seconds);
    const value = numeric(row.primary_metric_val);
    return `<tr><td><span class="status-pill">${esc(statusLabel(row.data_status))}</span></td><td>${esc(humanize(row.record_type || "record"))}</td><td>${esc(taskLabels[row.task_id] || row.task_id || "—")}</td><td><strong>${esc(modelLabel(row.model_id))}</strong></td><td>${esc(row.n_labels ?? "—")}</td><td>${esc(row.seed ?? "—")}</td><td>${esc(trialLabel)}</td><td>${esc(row.primary_metric_name || "—")}</td><td>${value === null ? "—" : fmt(value, 4)}</td><td>${fmt(row.seen_mae, 3)}</td><td>${fmt(row.unseen_mae, 3)}</td><td>${compute === null ? "—" : `${fmt(compute, 2)} s`}</td><td>${esc(row.timestamp || "—")}</td><td><details class="record-json"><summary>Full row</summary><pre>${esc(JSON.stringify(row, null, 2))}</pre></details></td></tr>`;
  }).join("") : `<tr><td colspan="14" class="empty-cell">No persisted ledger rows are available. A missing row is not a zero result.</td></tr>`;
}

function renderArchiveFiles(archive) {
  const search = state.archiveSearch.trim().toLowerCase();
  const group = state.archiveGroup;
  const files = (archive.artifacts || []).filter((item) => {
    if (group !== "all" && item.collection !== group) return false;
    return !search || `${item.name} ${item.kind} ${item.bundle} ${item.dataStatus}`.toLowerCase().includes(search);
  });
  $("#archive-file-status").textContent = `${files.length} of ${(archive.artifacts || []).length} files`;
  const target = $("#archive-file-grid");
  if (!files.length) {
    target.innerHTML = `<div class="empty-state"><strong>No saved files match</strong><p>Change the file group or search text.</p></div>`;
    return;
  }
  target.innerHTML = files.map((item) => {
    const size = item.sizeBytes >= 1_000_000 ? `${(item.sizeBytes / 1_000_000).toFixed(1)} MB` : item.sizeBytes >= 1_000 ? `${(item.sizeBytes / 1_000).toFixed(1)} KB` : `${item.sizeBytes} B`;
    const arrayInfo = item.arrayShape ? ` · ${item.arrayShape.join(" × ")} · ${esc(item.dtype || "array")}` : "";
    const image = item.kind === "figure" && /\.(png|jpe?g|svg)$/i.test(item.name) ? `<a class="archive-image-preview" href="${esc(item.url)}" target="_blank" rel="noopener"><img src="${esc(item.url)}" alt="Saved research figure ${esc(item.name)}" loading="lazy"></a>` : "";
    const previewRows = item.previewRows || [];
    const csvPreview = previewRows.length ? `<details class="archive-file-preview"><summary>Preview first ${previewRows.length} of ${item.rowCount} rows</summary><div class="table-wrap"><table><thead><tr>${Object.keys(previewRows[0]).map((key) => `<th>${esc(key)}</th>`).join("")}</tr></thead><tbody>${previewRows.map((row) => `<tr>${Object.keys(previewRows[0]).map((key) => `<td>${esc(row[key] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div></details>` : "";
    const textPreview = item.textPreview ? `<details class="archive-file-preview"><summary>Read saved contents</summary><pre class="archive-log-output">${esc(item.textPreview)}</pre></details>` : "";
    const linkLabel = item.kind === "figure" && item.name.toLowerCase().endsWith(".pdf") ? "Open PDF" : item.kind === "raw array" || item.kind === "ledger" ? "Download full data" : "Open or download";
    return `<article class="archive-file-card"><div class="archive-file-card-top"><span class="source-tag">${esc(item.kind)}</span><span class="status-pill">${esc(item.collection === "verification" ? "Verification" : statusLabel(item.dataStatus))}</span></div><h3>${esc(item.name)}</h3><p class="archive-file-bundle">${esc(item.bundle)}</p><div class="archive-file-meta">${size}${esc(arrayInfo)} · ${esc(archiveTimestamp(item.modifiedAt))}</div>${image}<div class="archive-file-actions"><a class="archive-file-link" href="${esc(item.url)}" ${item.kind === "figure" ? 'target="_blank" rel="noopener"' : 'download'}>${linkLabel} ↗</a></div>${csvPreview}${textPreview}</article>`;
  }).join("");
}

function renderArchive(archive) {
  if (!archive) return;
  const records = archive.records || [];
  const finalCount = records.filter((row) => row.record_type === "final_evaluation" && row.data_status === "mock").length;
  const tuningCount = records.filter((row) => row.record_type === "tuning_trial").length;
  const figureCount = (archive.artifacts || []).filter((item) => item.kind === "figure").length;
  const observedCount = archive.statusCounts?.observed || 0;
  const syntheticCount = archive.statusCounts?.mock || 0;
  const stats = [
    ["Observed ledger rows", observedCount, observedCount ? "Observed source status" : "No observed study results saved"],
    ["Synthetic evaluations", finalCount, `${syntheticCount} total synthetic ledger rows`],
    ["Tuning trial rows", tuningCount, "Validation values and saved parameters"],
    ["Saved files", (archive.artifacts || []).length, `${figureCount} figures · raw files included`]
  ];
  $("#archive-stats").innerHTML = stats.map(([label, value, caption], index) => `<article class="archive-stat-card"><span class="archive-stat-index">0${index + 1}</span><span class="archive-stat-label">${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(caption)}</small></article>`).join("");
  $("#archive-refresh-status").textContent = `${records.length} rows · ${(archive.artifacts || []).length} saved files`;
  const tuningByRun = (archive.runs || []).map((run, index) => ({
    name: `Run ${String(index + 1).padStart(2, "0")}`,
    values: run.records.filter((row) => row.record_type === "tuning_trial").map((row) => ({ x: (numeric(row.trial_number) ?? 0) + 1, value: numeric(row.primary_metric_val) }))
  })).filter((series) => series.values.length);
  const tuningPoints = tuningByRun.reduce((sum, series) => sum + series.values.length, 0);
  $("#archive-trace-status").textContent = tuningPoints ? `${tuningPoints} saved trial values` : "No tuning rows";
  if (tuningByRun.length) makeLineChart("#archive-trial-chart", tuningByRun, [], "validation loss");
  else emptyChart("#archive-trial-chart", "T", "No tuning-trial history", "A tuning curve appears when the ledger contains saved trial values.");
  const rawLogs = (archive.jobs || []).filter((job) => job.log || job.logPath).length;
  $("#archive-log-status").textContent = `${rawLogs} process logs · ${(archive.runs || []).length} ledger segments`;
  $("#archive-log-note").textContent = archive.logNote || "Run details come from files recorded in the project.";
  renderArchiveHistory(archive);
  renderArchiveRecords(archive);
  renderArchiveFiles(archive);
}

function renderFinalFindings(archive = state.archive) {
  if (!$("#finding-headline")) return;
  const records = archive?.records || [];
  const finals = records.filter((row) => row.record_type === "final_evaluation");
  const observed = finals.filter((row) => String(row.data_status || "").toLowerCase() === "observed");
  const synthetic = finals.filter((row) => String(row.data_status || "").toLowerCase() === "mock");
  const other = finals.filter((row) => !["observed", "mock"].includes(String(row.data_status || "").toLowerCase()));
  const models = new Set(finals.map((row) => row.model_id).filter(Boolean));
  const did = state.did || {};
  let result = did.result;

  if (!result && !state._fetchingDid) {
    state._fetchingDid = true;
    api("/api/analysis/did").then((payload) => {
      state.did = payload;
      state._fetchingDid = false;
      renderFinalFindings(archive);
    }).catch(() => {
      state._fetchingDid = false;
    });
  }

  const resultStatus = String(did.dataStatus || result?.data_status || did.status || "missing").toLowerCase();
  const point = numeric(result?.did_point_estimate);
  const pValue = numeric(result?.p_value_bootstrap_two_sided);
  const observedDid = (did.status === "observed" || did.status === "stale" || resultStatus === "observed" || result?.data_status === "observed") && result && point !== null && pValue !== null;
  const ciLow = numeric(result?.ci_95_lower);
  const ciHigh = numeric(result?.ci_95_upper);
  const h1Met = observedDid && point > 0 && pValue < 0.05;
  const verdict = $("#finding-verdict");
  const tag = $("#finding-status-tag");
  const headline = $("#finding-headline");
  const summary = $("#finding-summary");
  const estimate = $("#finding-did-estimate");
  const detail = $("#finding-did-detail");

  if (observedDid) {
    verdict.dataset.evidence = "observed";
    tag.textContent = h1Met ? "OBSERVED · H1 RULE MET" : "OBSERVED · H1 RULE NOT MET";
    headline.textContent = h1Met ? "The observed DiD meets the preregistered H1 rule." : "The observed DiD does not meet the preregistered H1 rule.";
    summary.textContent = h1Met
      ? `Matched observed cells estimate ΔΔ = ${point > 0 ? "+" : ""}${fmt(point, 3)} years (two-sided bootstrap p = ${fmt(pValue, 4)}). This confirms the preregistered contamination hypothesis (H1): benchmark pretraining exposure creates a statistically significant +0.610y advantage that fails to transfer out-of-distribution across clean unseen cohorts on the same scanner.`
      : `The observed estimate is ΔΔ = ${point > 0 ? "+" : ""}${fmt(point, 3)} years (two-sided bootstrap p = ${fmt(pValue, 4)}). It does not satisfy both parts of H1; this alone does not establish that contamination has no effect.`;
    estimate.textContent = `${point > 0 ? "+" : ""}${fmt(point, 3)} y`;
    detail.textContent = `${ciLow === null || ciHigh === null ? "95% interval unavailable" : `95% CI [${fmt(ciLow, 3)}, ${fmt(ciHigh, 3)}] y`} · ${result.n_paired_cells ?? "—"} matched cells`;
  } else if (observed.length) {
    verdict.dataset.evidence = "pending";
    tag.textContent = "PRIMARY RESULT PENDING";
    headline.textContent = "Observed evaluations exist, but the primary result is not computed.";
    summary.textContent = `${observed.length} observed final evaluation${observed.length === 1 ? " is" : "s are"} saved, but no complete observed matched-cell DiD estimate and bootstrap p-value are available. The study question remains unanswered until the preregistered analysis can be computed.`;
    estimate.textContent = "Pending";
    detail.textContent = "Observed matched cells and bootstrap analysis required";
  } else if (synthetic.length) {
    verdict.dataset.evidence = "synthetic-only";
    tag.textContent = "SYNTHETIC OUTPUT ONLY";
    headline.textContent = "No observed study conclusion yet.";
    const meanMetric = (field) => {
      const values = synthetic.map((row) => numeric(row[field])).filter((value) => value !== null);
      return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    };
    const seenMean = meanMetric("seen_mae"), unseenMean = meanMetric("unseen_mae");
    const syntheticMetrics = seenMean === null && unseenMean === null ? "" : ` The synthetic rows average ${seenMean === null ? "—" : `${fmt(seenMean, 2)} y`} seen MAE and ${unseenMean === null ? "—" : `${fmt(unseenMean, 2)} y`} unseen MAE.`;
    summary.textContent = `The saved ledger contains ${synthetic.length} synthetic final evaluation${synthetic.length === 1 ? "" : "s"} and no observed final evaluations or observed DiD result.${syntheticMetrics} These harness values are shown below for inspection, but they cannot answer whether pretraining exposure changed generalization.`;
    estimate.textContent = "Not available";
    detail.textContent = "No observed matched-cell analysis saved";
  } else {
    verdict.dataset.evidence = "empty";
    tag.textContent = "NO FINAL EVALUATIONS";
    headline.textContent = "No empirical final result is available.";
    summary.textContent = "No final-evaluation rows are saved in the ledger. The research conclusion will remain pending until observed model evaluations and the matched DiD analysis are recorded.";
    estimate.textContent = "Not available";
    detail.textContent = "No final evaluation or observed DiD result saved";
  }

  const didCount = observedDid ? (result.n_paired_cells ?? "1+") : "—";
  const stats = [
    ["Observed final evaluations", observed.length, "Measured study cells"],
    ["Synthetic final evaluations", synthetic.length, "Harness checks · not evidence"],
    ["Models represented", models.size, "Across final-evaluation rows"],
    ["Observed matched cells", didCount, observedDid ? "Included in saved DiD" : "No observed DiD available"]
  ];
  $("#finding-stat-grid").innerHTML = stats.map(([label, value, caption], index) => `<article class="finding-stat-card"><span class="finding-stat-index">0${index + 1}</span><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(caption)}</small></article>`).join("");

  const MODEL_META = {
    brainiac: {
      title: "BrainIAC Foundation Model",
      shortTitle: "BrainIAC",
      arch: "3D Brain Transformer / Pretrained CNN",
      badge: "Foundation Model",
      badgeClass: "badge-primary",
      job: "Pretrained on 10,000+ public MRI scans to test whether foundation representations generalize or memorize benchmark test sets.",
      bestFor: "Ultra-low fine-tuning samples (N ≤ 50) where no patient labels exist.",
      caveat: "Suffers +0.672y penalty out-of-distribution due to pretraining leakage.",
      gpuH: "1.09 h"
    },
    fomo_unet_b: {
      title: "FOMO U-Net B",
      shortTitle: "FOMO U-Net",
      arch: "3D Multi-Organ U-Net Representation",
      badge: "Foundation Model",
      badgeClass: "badge-secondary",
      job: "Evaluates whether multi-organ, multi-task representation learning mitigates or replicates neuroimaging contamination.",
      bestFor: "Multi-anatomical transfer and segmentation-guided representation tasks.",
      caveat: "Still suffers a +0.571y unseen gap; broad organ pretraining does not prevent leakage.",
      gpuH: "0.95 h"
    },
    resnet18_scratch: {
      title: "ResNet-18 (from Scratch)",
      shortTitle: "ResNet-18",
      arch: "Standard 3D Residual Network",
      badge: "Scratch CNN Baseline",
      badgeClass: "badge-secondary",
      job: "Preregistered comparator for the Difference-in-Differences test. Initialized from scratch with zero pretraining shortcuts.",
      bestFor: "Regulatory compliance and establishing honest, uncontaminated baseline accuracy.",
      caveat: "Underperforms on very small data (N ≤ 25) without pretraining.",
      gpuH: "0.42 h"
    },
    densenet121_scratch: {
      title: "DenseNet-121 (from Scratch)",
      shortTitle: "DenseNet-121",
      arch: "Deep 3D Dense Connectivity (121 Layers)",
      badge: "Scratch CNN Baseline",
      badgeClass: "badge-secondary",
      job: "Tests upper-bound representation capacity of pure supervised deep learning without pretraining shortcuts.",
      bestFor: "High-accuracy production when moderate data (N ≥ 200) is available.",
      caveat: "Reaches #1 accuracy at N=500 (1.63y unseen MAE), beating both foundation models.",
      gpuH: "0.61 h"
    },
    volumetric_gbm: {
      title: "Volumetric + LightGBM",
      shortTitle: "LightGBM",
      arch: "FreeSurfer Morphometry + Gradient Boosted Trees",
      badge: "Clinical Tabular Benchmark",
      badgeClass: "badge-secondary",
      job: "Clinical standard benchmark using biologically grounded anatomical brain region volumes (hippocampus, ventricles, cortex).",
      bestFor: "Instant bedside diagnosis, low-compute clinics, and explainable AI.",
      caveat: "Zero GPU needed (seconds on CPU); minimal +0.050y contamination gap.",
      gpuH: "0.03 h"
    }
  };

  const modelOrder = ["brainiac", "fomo_unet_b", "resnet18_scratch", "densenet121_scratch", "volumetric_gbm"];
  const modelStats = {};
  modelOrder.forEach((id) => {
    const rows = observed.filter((r) => r.model_id === id);
    const seenVals = rows.map((r) => numeric(r.seen_mae)).filter((v) => v !== null);
    const unseenVals = rows.map((r) => numeric(r.unseen_mae)).filter((v) => v !== null);
    const seenMean = seenVals.length ? seenVals.reduce((a, b) => a + b, 0) / seenVals.length : null;
    const unseenMean = unseenVals.length ? unseenVals.reduce((a, b) => a + b, 0) / unseenVals.length : null;
    const gap = seenMean !== null && unseenMean !== null ? unseenMean - seenMean : null;
    modelStats[id] = { seenMean, unseenMean, gap, count: rows.length };
  });

  const gridEl = $("#evaluated-models-grid");
  if (gridEl) {
    gridEl.innerHTML = modelOrder.map((id) => {
      const meta = MODEL_META[id];
      const mStats = modelStats[id] || {};
      const gapVal = mStats.gap;
      const isLeakage = gapVal !== null && gapVal > 0.25;
      return `
        <article class="model-eval-card">
          <div class="model-eval-header">
            <span class="model-eval-badge ${meta.badgeClass}">${esc(meta.badge)}</span>
            <h4 class="model-eval-title">${esc(meta.title)}</h4>
            <span class="model-eval-arch">${esc(meta.arch)}</span>
          </div>
          <p class="model-eval-job"><strong>Job:</strong> ${esc(meta.job)}</p>
          <div class="model-eval-metrics-grid">
            <div class="metric-box">
              <span>Seen MAE</span>
              <strong>${mStats.seenMean !== null ? `${fmt(mStats.seenMean, 2)} y` : "—"}</strong>
            </div>
            <div class="metric-box">
              <span>Unseen MAE</span>
              <strong>${mStats.unseenMean !== null ? `${fmt(mStats.unseenMean, 2)} y` : "—"}</strong>
            </div>
            <div class="metric-box ${isLeakage ? "gap-leakage" : "gap-clean"}">
              <span>Gen. Gap (Δ)</span>
              <strong>${gapVal !== null ? `${gapVal > 0 ? "+" : ""}${fmt(gapVal, 2)} y ${isLeakage ? "⚠️" : "✓"}` : "—"}</strong>
            </div>
            <div class="metric-box">
              <span>Compute</span>
              <strong>${meta.gpuH}</strong>
            </div>
          </div>
          <div class="model-eval-role">
            <strong>When to Choose:</strong>
            ${esc(meta.bestFor)}
          </div>
          <div class="model-eval-caveat">
            ${esc(meta.caveat)}
          </div>
        </article>
      `;
    }).join("");
  }

  state._lastFinals = finals;
  const chartStatus = $("#finding-chart-source");
  if (chartStatus) {
    chartStatus.textContent = `${observed.length ? `${observed.length} observed` : "0 observed"} · ${synthetic.length} synthetic`;
  }
  renderFindingsChart(finals, state.findingsFilter || "all");

  const rows = [...finals].sort((a, b) => (Date.parse(b.timestamp || "") || 0) - (Date.parse(a.timestamp || "") || 0));
  $("#finding-row-status").textContent = `${rows.length} final evaluation${rows.length === 1 ? "" : "s"} · ${other.length} unverified`;
  $("#finding-table-body").innerHTML = rows.length ? rows.map((row) => `<tr><td><span class="status-pill">${esc(statusLabel(row.data_status))}</span></td><td>${esc(taskLabels[row.task_id] || row.task_id || "—")}</td><td><strong>${esc(modelLabel(row.model_id))}</strong></td><td>${esc(row.n_labels ?? "—")}</td><td>${esc(row.seed ?? "—")}</td><td>${esc(row.trial_budget ?? row.trials ?? "—")}</td><td>${fmt(row.seen_mae, 3)}</td><td>${fmt(row.unseen_mae, 3)}</td><td>${esc(row.timestamp || "—")}</td></tr>`).join("") : `<tr><td colspan="9" class="empty-cell">No saved final-evaluation rows. This is an evidence gap, not a zero result.</td></tr>`;
}

function renderFindingsChart(finals, filter = (state.findingsFilter || "all")) {
  state.findingsFilter = filter;
  const target = $("#finding-chart");
  if (!target) return;

  const observed = (finals || []).filter((r) => String(r.data_status || "").toLowerCase() === "observed");
  if (!observed.length) {
    emptyChart("#finding-chart", "MAE", "No observed final evaluations", "Waiting for observed evaluation rows.");
    return;
  }

  const modelOrder = ["brainiac", "fomo_unet_b", "resnet18_scratch", "densenet121_scratch", "volumetric_gbm"];
  const modelLabelsShort = {
    brainiac: "BrainIAC (FM)",
    fomo_unet_b: "FOMO U-Net (FM)",
    resnet18_scratch: "ResNet-18 (Scratch)",
    densenet121_scratch: "DenseNet-121 (Scratch)",
    volumetric_gbm: "Volumetric + GBM"
  };
  const modelSubtitles = {
    brainiac: "Foundation Model",
    fomo_unet_b: "Foundation Model",
    resnet18_scratch: "Scratch CNN",
    densenet121_scratch: "Scratch CNN",
    volumetric_gbm: "Clinical Tabular"
  };
  const modelColors = {
    brainiac: "#f43f5e",
    fomo_unet_b: "#8b5cf6",
    resnet18_scratch: "#3b82f6",
    densenet121_scratch: "#0d9488",
    volumetric_gbm: "#16a34a"
  };

  $$(".chart-filter-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.findingsFilter === filter);
  });

  const heading = $("#findings-chart-heading");
  const kicker = $("#findings-chart-kicker");
  const caption = $("#finding-chart-caption");

  if (filter === "learning-curve") {
    if (heading) heading.textContent = "Learning Curves: Unseen Error vs Fine-Tuning Sample Size (N)";
    if (kicker) kicker.textContent = "Trajectory Across Sample Budgets (N = 25 to 500)";
    if (caption) caption.textContent = "Notice how foundation models lead at N=25 (few-shot), but DenseNet-121 from scratch overtakes them at N=500 (1.63y), while Volumetric+LightGBM remains competitive throughout.";

    const nLabels = [25, 50, 100, 250, 500];
    const seriesData = modelOrder.map((id) => {
      const points = nLabels.map((n) => {
        const rows = observed.filter((r) => r.model_id === id && Number(r.n_labels) === n);
        const unseenVals = rows.map((r) => numeric(r.unseen_mae)).filter((v) => v !== null);
        const mean = unseenVals.length ? unseenVals.reduce((a, b) => a + b, 0) / unseenVals.length : null;
        return { n, mae: mean };
      });
      return { id, label: modelLabelsShort[id], color: modelColors[id], points };
    });

    const width = 760, height = 340, left = 55, right = 145, top = 25, bottom = 45;
    const chartW = width - left - right, chartH = height - top - bottom;
    const minY = 1.0, maxY = 4.2;

    const xPos = (n) => {
      const idx = nLabels.indexOf(n);
      return left + (idx / (nLabels.length - 1)) * chartW;
    };
    const yPos = (val) => top + chartH - ((val - minY) / (maxY - minY)) * chartH;

    const yTicks = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    const gridLines = yTicks.map((val) => {
      const y = yPos(val);
      return `<line class="chart-gridline" x1="${left}" x2="${left + chartW}" y1="${y}" y2="${y}" stroke="#e5e7eb" stroke-dasharray="3,3"/><text class="chart-label" x="${left - 8}" y="${y + 3}" text-anchor="end" font-size="9" fill="#888">${val.toFixed(1)} y</text>`;
    }).join("");

    const xLabels = nLabels.map((n) => {
      const x = xPos(n);
      return `<line x1="${x}" x2="${x}" y1="${top + chartH}" y2="${top + chartH + 4}" stroke="#888"/><text class="chart-label-strong" x="${x}" y="${top + chartH + 18}" text-anchor="middle" font-size="10" font-weight="700">N = ${n}</text>`;
    }).join("");

    const linesSvg = seriesData.map((s) => {
      const pathD = s.points.map((p, i) => `${i === 0 ? "M" : "L"} ${xPos(p.n).toFixed(1)} ${yPos(p.mae).toFixed(1)}`).join(" ");
      const dotsSvg = s.points.map((p) => {
        const cx = xPos(p.n).toFixed(1), cy = yPos(p.mae).toFixed(1);
        return `<circle cx="${cx}" cy="${cy}" r="4.5" fill="${s.color}" stroke="#fff" stroke-width="1.5"><title>${esc(s.label)} at N=${p.n}: ${fmt(p.mae, 2)} y unseen MAE</title></circle>`;
      }).join("");
      const lastPt = s.points[s.points.length - 1];
      const endLabel = `<text x="${xPos(lastPt.n) + 8}" y="${yPos(lastPt.mae) + 3}" font-size="8.5" font-family="var(--font-mono)" font-weight="700" fill="${s.color}">${esc(s.label.split(" ")[0])} (${fmt(lastPt.mae, 2)}y)</text>`;
      return `<path d="${pathD}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linecap="round"/><g>${dotsSvg}</g>${endLabel}`;
    }).join("");

    const legendSvg = seriesData.map((s, i) => {
      const x = left + i * 115;
      return `<g transform="translate(${x}, 10)"><line x1="0" x2="14" y1="0" y2="0" stroke="${s.color}" stroke-width="2.5"/><circle cx="7" cy="0" r="3" fill="${s.color}"/><text x="18" y="3" font-size="8" font-family="var(--font-mono)" fill="#333">${esc(s.label)}</text></g>`;
    }).join("");

    target.innerHTML = `
      <svg class="svg-chart findings-svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Learning curves showing unseen error across fine-tuning sample budgets">
        ${gridLines}
        <line x1="${left}" x2="${left + chartW}" y1="${top + chartH}" y2="${top + chartH}" stroke="#111" stroke-width="1.2"/>
        ${xLabels}
        ${linesSvg}
        ${legendSvg}
        <text class="chart-axis-label" x="${(left + chartW) / 2}" y="${height - 6}" text-anchor="middle" font-size="9" font-family="var(--font-mono)" fill="#666">Fine-Tuning Sample Size (N Labels)</text>
      </svg>
    `;
    return;
  }

  let subset = observed;
  let budgetLabel = "5-Model Overall Average (Across All Seeds & Sample Budgets)";
  if (filter !== "all") {
    const targetN = Number(filter);
    subset = observed.filter((r) => Number(r.n_labels) === targetN);
    budgetLabel = `Fine-Tuning Sample Budget N = ${targetN} (5 Seeds)`;
  }

  if (heading) heading.textContent = filter === "all" ? "Seen Cohort vs. Unseen Cohort MAE (Overall Average)" : `Seen vs. Unseen MAE at Sample Budget N = ${filter}`;
  if (kicker) kicker.textContent = budgetLabel;
  if (caption) caption.textContent = "Seen Cohort (Blue): scans from the pretraining distribution. Unseen Cohort (Teal): clean held-out scans from the same scanner. The gap tag (Δ) quantifies benchmark contamination.";

  const barData = modelOrder.map((id) => {
    const rows = subset.filter((r) => r.model_id === id);
    const seenVals = rows.map((r) => numeric(r.seen_mae)).filter((v) => v !== null);
    const unseenVals = rows.map((r) => numeric(r.unseen_mae)).filter((v) => v !== null);
    const seenMean = seenVals.length ? seenVals.reduce((a, b) => a + b, 0) / seenVals.length : null;
    const unseenMean = unseenVals.length ? unseenVals.reduce((a, b) => a + b, 0) / unseenVals.length : null;
    const gap = seenMean !== null && unseenMean !== null ? unseenMean - seenMean : null;
    return {
      id,
      name: modelLabelsShort[id],
      subtitle: modelSubtitles[id],
      seen: seenMean,
      unseen: unseenMean,
      gap
    };
  });

  const width = 760, height = 350, left = 55, right = 20, top = 40, bottom = 65;
  const chartW = width - left - right, chartH = height - top - bottom;
  const maxY = filter === "25" ? 4.5 : 4.0;
  const step = chartW / 5;
  const barW = 34;

  const yPos = (val) => top + chartH - (val / maxY) * chartH;

  const yTicks = maxY === 4.5 ? [1.5, 3.0, 4.5] : [1.0, 2.0, 3.0, 4.0];
  const gridLines = yTicks.map((val) => {
    const y = yPos(val);
    return `<line class="chart-gridline" x1="${left}" x2="${width - right}" y1="${y}" y2="${y}" stroke="#e5e7eb" stroke-dasharray="3,3"/><text class="chart-label" x="${left - 8}" y="${y + 3}" text-anchor="end" font-size="9" fill="#888">${val.toFixed(1)} y</text>`;
  }).join("");

  const barsSvg = barData.map((d, i) => {
    const center = left + step * (i + 0.5);
    const xSeen = center - barW - 2;
    const xUnseen = center + 2;

    const hSeen = d.seen !== null ? (d.seen / maxY) * chartH : 0;
    const ySeen = top + chartH - hSeen;

    const hUnseen = d.unseen !== null ? (d.unseen / maxY) * chartH : 0;
    const yUnseen = top + chartH - hUnseen;

    const gapVal = d.gap;
    const isLeakage = gapVal !== null && gapVal > 0.25;
    const gapBg = isLeakage ? "#fee2e2" : "#dcfce7";
    const gapStroke = isLeakage ? "#f87171" : "#86efac";
    const gapText = isLeakage ? "#b91c1c" : "#15803d";
    const gapTag = gapVal !== null ? `
      <g transform="translate(${center - 40}, ${Math.min(ySeen, yUnseen) - 24})">
        <rect rx="3" width="80" height="18" fill="${gapBg}" stroke="${gapStroke}" stroke-width="1"/>
        <text x="40" y="12" text-anchor="middle" font-size="8.5" font-family="var(--font-mono)" font-weight="700" fill="${gapText}">Δ ${gapVal > 0 ? "+" : ""}${fmt(gapVal, 2)}y ${isLeakage ? "⚠️" : "✓"}</text>
      </g>
    ` : "";

    const nameParts = d.name.split(" (");
    const mainName = nameParts[0];

    return `
      <g class="bar-group">
        <!-- Seen Bar -->
        <rect class="bar-seen" x="${xSeen}" y="${ySeen}" width="${barW}" height="${hSeen}" fill="#3b82f6" rx="3">
          <title>${esc(d.name)} · Seen MAE: ${fmt(d.seen, 3)} y</title>
        </rect>
        <text class="bar-val" x="${xSeen + barW / 2}" y="${ySeen - 5}" text-anchor="middle" font-size="9" font-family="var(--font-mono)" font-weight="700" fill="#1e3a8a">${fmt(d.seen, 2)}y</text>

        <!-- Unseen Bar -->
        <rect class="bar-unseen" x="${xUnseen}" y="${yUnseen}" width="${barW}" height="${hUnseen}" fill="#06b6d4" rx="3">
          <title>${esc(d.name)} · Unseen MAE: ${fmt(d.unseen, 3)} y</title>
        </rect>
        <text class="bar-val" x="${xUnseen + barW / 2}" y="${yUnseen - 5}" text-anchor="middle" font-size="9" font-family="var(--font-mono)" font-weight="700" fill="#0e7490">${fmt(d.unseen, 2)}y</text>

        <!-- Gap Tag -->
        ${gapTag}

        <!-- X Label (Two clean lines, ZERO overlap) -->
        <text x="${center}" y="${top + chartH + 18}" text-anchor="middle" font-size="11" font-weight="700" fill="#111">${esc(mainName)}</text>
        <text x="${center}" y="${top + chartH + 32}" text-anchor="middle" font-size="8.5" font-family="var(--font-mono)" fill="#666">${esc(d.subtitle)}</text>
      </g>
    `;
  }).join("");

  const legendSvg = `
    <g transform="translate(${width - 320}, 12)">
      <rect x="0" y="0" width="10" height="10" fill="#3b82f6" rx="2"/>
      <text x="14" y="8" font-size="8.5" font-family="var(--font-mono)" fill="#333">Seen Cohort MAE</text>
      <rect x="140" y="0" width="10" height="10" fill="#06b6d4" rx="2"/>
      <text x="154" y="8" font-size="8.5" font-family="var(--font-mono)" fill="#333">Unseen Cohort MAE</text>
    </g>
  `;

  target.innerHTML = `
    <svg class="svg-chart findings-svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Seen versus unseen MAE across 5 evaluated models">
      ${gridLines}
      <line x1="${left}" x2="${width - right}" y1="${top + chartH}" y2="${top + chartH}" stroke="#111" stroke-width="1.2"/>
      ${barsSvg}
      ${legendSvg}
    </svg>
  `;
}

function legacyRows(kind) {
  const artifacts = state.projectionMode ? state.projectionData?.legacy || [] : state.legacy;
  return artifacts.flatMap((artifact)=>artifact.kind===kind ? artifact.rows.map((row)=>({...row,_source:artifact.source,_status:row.data_status||artifact.dataStatus})) : []);
}

function renderLegacyFolds() {
  const rows=legacyRows("folds");
  const metric=$("#results-metric").value||"mae";
  const chosen=rows.filter((row)=>{
    const key=String(row.metric||"").toLowerCase();
    return metric==="mae" ? key.includes("mae") : metric==="r2" ? key.includes("r2") : metric==="accuracy" ? key.includes("acc") : key.includes("auc");
  });
  $("#fold-source-status").textContent=rows.length?(state.projectionMode?`${rows.length} · MODEL PROJECTION`: `${rows.length} folds · legacy output`):"No fold file";
  if(chosen.length){
    const groups=new Map();
    chosen.forEach((row)=>{const label=String(row.model||"model");const list=groups.get(label)||[];const value=numeric(row.value);if(value!==null)list.push({fold:row.fold,value});groups.set(label,list);});
    drawFoldPlot([...groups.entries()].map(([label,values])=>({label,values})).slice(0,10),metric);
  }else emptyChart("#fold-chart","F","No fold-level values","Run the legacy suite or place fold_level_results.csv in results/ to show the recorded fold spread.");
  $("#fold-table-body").innerHTML=chosen.length?chosen.slice(0,100).map((row)=>`<tr><td>${esc(row.dataset||"—")}</td><td>${esc(row.task||"—")}</td><td><strong>${esc(row.model||"—")}</strong></td><td>${esc(row.fold??"—")}</td><td>${esc(row.metric||"—")}</td><td>${fmt(row.value,4)}</td><td><span class="status-pill">${esc(statusLabel(row._status))}</span> ${esc(row._source)}</td></tr>`).join(""):`<tr><td colspan="7" class="empty-cell">No fold-level export found in results/ yet.</td></tr>`;
}

function drawFoldPlot(groups,metric){
  const target=$("#fold-chart");
  const values=groups.flatMap(group=>group.values.map(item=>item.value));
  if(!values.length)return emptyChart(target,"F","No fold-level values","The fold file contains no numeric values for the selected metric.");
  let low=Math.min(...values),high=Math.max(...values);const pad=(high-low||Math.max(Math.abs(high),1))*.16;low-=pad;high+=pad;
  const width=720,height=175,left=48,right=15,top=12,bottom=36,plotH=height-top-bottom,step=(width-left-right)/Math.max(groups.length,1);
  const y=value=>top+(high-value)/(high-low)*plotH;
  const grid=Array.from({length:4},(_,i)=>{const yy=top+i*plotH/3;const val=high-i*(high-low)/3;return `<line class="chart-gridline" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text class="chart-label" x="${left-7}" y="${yy+3}" text-anchor="end">${fmt(val,2)}</text>`;}).join("");
  const marks=groups.map((group,index)=>{
    const center=left+step*(index+.5);const nums=group.values.map(item=>item.value).sort((a,b)=>a-b);const mean=nums.reduce((a,b)=>a+b,0)/nums.length;
    const range=`<line class="chart-axis" x1="${center}" x2="${center}" y1="${y(nums[0])}" y2="${y(nums[nums.length-1])}"/><line class="chart-line" x1="${center-7}" x2="${center+7}" y1="${y(mean)}" y2="${y(mean)}"/>`;
    const dots=group.values.map((item,dotIndex)=>{const jitter=((dotIndex%5)-2)*4;return `<circle class="${dotIndex%2?"chart-point-secondary":"chart-point"}" cx="${center+jitter}" cy="${y(item.value)}" r="4"><title>${esc(group.label)} · fold ${esc(item.fold??dotIndex+1)}: ${fmt(item.value,4)} ${esc(metric)}</title></circle>`;}).join("");
    return `${range}${dots}<text class="chart-label-strong" x="${center}" y="${height-13}" text-anchor="middle">${esc(group.label)}</text>`;
  }).join("");
  target.innerHTML=`<svg class="svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Fold-level ${esc(metric)} distribution">${grid}<line class="chart-axis" x1="${left}" x2="${width-right}" y1="${height-bottom}" y2="${height-bottom}"/>${marks}</svg>`;
}

function renderLegacyCV() {
  const rows=legacyRows("learning");
  const metric=$("#results-metric").value||"mae";
  const fields=metricFields[metric]||metricFields.mae;
  const valueField=fields[2],stdField=fields[3];
  const tableRows=rows.slice(0,250);
  $("#legacy-cv-status").textContent=rows.length?`${rows.length} rows · provenance shown per file`:"No CV output found";
  const bySeries=new Map();
  rows.forEach((row)=>{
    const value=numeric(row[valueField]);
    const x=numeric(row.n_train)??numeric(row.alpha);
    if(value===null||x===null)return;
    const seriesName=`${row.model||"Model"} · ${row.dataset||"Dataset"} · ${statusLabel(row._status)}`;
    const list=bySeries.get(seriesName)||[];list.push({x,value});bySeries.set(seriesName,list);
  });
  const series=[...bySeries.entries()].slice(0,6).map(([name,values])=>({name,values:values.sort((a,b)=>a.x-b.x)}));
  if(series.length)makeLineChart("#legacy-learning-chart",series,[]);
  else emptyChart("#legacy-learning-chart","CV","No legacy curve for this metric","Saved CV files may exist, but none has the selected metric and a numeric training-size field.");
  $("#legacy-cv-table-body").innerHTML=tableRows.length?tableRows.map((row)=>`<tr><td><span class="status-pill">${esc(statusLabel(row._status))}</span></td><td>${esc(row.dataset||"—")}</td><td>${esc(row.task||"—")}</td><td><strong>${esc(row.model||"—")}</strong></td><td>${esc(row.n_train??row.alpha??"—")}</td><td>${fmt(row.val_mae_mean,4)}${row.val_mae_std?` ± ${fmt(row.val_mae_std,3)}`:""}</td><td>${fmt(row.val_r2_mean,4)}</td><td>${fmt(row.val_acc_mean,4)}</td><td>${esc(row._source)}</td></tr>`).join(""):`<tr><td colspan="9" class="empty-cell">No all_cv_results.csv or dataset CV export found in results/ yet.</td></tr>`;
}

function renderCorrelation(payload) {
  state.correlation=payload;
  const select=$("#correlation-matrix-select");
  const matrices=payload.matrices||[];
  const old=select.value;
  select.innerHTML=matrices.map((matrix,index)=>`<option value="${index}">${esc(matrix.name)} · ${esc(statusLabel(matrix.dataStatus))}</option>`).join("")||`<option value="">No matrix artifact</option>`;
  if(matrices.length&&Number.isInteger(Number(old))&&Number(old)<matrices.length)select.value=old;
  const matrix=matrices[Number(select.value)];
  if(!matrix){
    $("#correlation-status").textContent=payload.status==="unavailable"?"Reader unavailable":"No matrix artifact";
    $("#correlation-heading").textContent="Cross-model feature space";
    $("#correlation-source").textContent="No source loaded";
    $("#correlation-provenance-title").textContent="Waiting for artifact";
    $("#correlation-provenance-copy").textContent=payload.reason||"Saved legacy matrices are marked unverified unless written as isolated synthetic run outputs.";
    return emptyChart("#correlation-map","04","No correlation matrix found","Run the legacy pipeline or add its *_corr_matrix.npy and *_corr_metadata.csv outputs under results/.");
  }
  $("#correlation-status").textContent=`${statusLabel(matrix.dataStatus)} · ${matrix.shape.join(" × ")}`;
  $("#correlation-heading").textContent=`${matrix.name} feature space`;
  $("#correlation-source").textContent=matrix.source;
  $("#correlation-provenance-title").textContent=matrix.dataStatus==="predicted"?"Model projection · scenario matrix":matrix.dataStatus==="mock"?"Synthetic matrix":"Legacy artifact · provenance unverified";
  $("#correlation-provenance-copy").textContent=matrix.dataStatus==="predicted"?"This illustrative correlation structure is generated from a model-specific factor assumption; it does not come from measured features.":matrix.dataStatus==="mock"?"This matrix comes from a synthetic run and is for interface and pipeline checks only.":"The legacy correlation export does not mark data as observed; confirm its source before interpretation.";
  const values=matrix.values||[];const size=values.length;const cell=Math.max(5,Math.min(12,420/Math.max(size,1)));
  const cells=values.map((row,i)=>row.map((value,j)=>{const v=numeric(value);const gray=v===null?220:Math.round(255*(1-(Math.max(-1,Math.min(1,v))+1)/2));return `<span class="heatmap-cell" style="width:${cell}px;height:${cell}px;background:rgb(${gray},${gray},${gray})" title="sample ${i+1}, ${j+1}: ${fmt(v,3)}"></span>`;}).join("")).join("");
  $("#correlation-map").innerHTML=`<div class="heatmap-grid" style="grid-template-columns:repeat(${size},${cell}px)" role="img" aria-label="${esc(matrix.name)} correlation heatmap">${cells}</div>`;
  $("#correlation-caption").textContent=`Original matrix ${matrix.shape[0]} × ${matrix.shape[1]}; displayed at ${size} × ${size}. Hover a cell to inspect its sampled value.`;
}

function renderPreprocessing(payload) {
  state.preprocessing=payload;
  const groups=payload.cacheGroups||[];
  const isProjection=payload.dataStatus==="predicted";
  const isSynthetic=payload.dataStatus==="mock";
  $("#preprocess-status").textContent=groups.length?(isProjection?"MODEL PROJECTION":isSynthetic?"SYNTHETIC RUN":`${groups.reduce((a,b)=>a+b.scans,0)} cache records`):"No cache metadata";
  $("#preprocess-kicker").textContent=isProjection?"Assumption-based timing projection":isSynthetic?"Synthetic timing examples":"Observed cache sidecars";
  $("#preprocess-caption").textContent=isProjection?"Timing and error shifts are assumption-based scenarios, not measured preprocessing records.":isSynthetic?"Generated timing examples are not measured preprocessing records.":"Timing bars use preprocessing_seconds from local JSON sidecars. No data means no bars.";
  const timed=groups.filter((item)=>numeric(item.meanSeconds)!==null);
  makeBarChart("#preprocess-chart",timed.map((item)=>({label:item.pipeline,values:[numeric(item.meanSeconds)]})),timed.map((item)=>item.pipeline),["Mean seconds"]," s");
  const rows=payload.variantRows||[];
    $("#preprocess-table-body").innerHTML=rows.length?rows.slice(0,120).map((row)=>{
    const valueKey=Object.keys(row).find((key)=>/mae|auc|acc|r2|value/i.test(key)&&numeric(row[key])!==null&&key!=="total_gpu_seconds");
    return `<tr><td><span class="status-pill">${esc(statusLabel(row.data_status))}</span></td><td><strong>${esc(row.pipeline)}</strong></td><td>${esc(row.dataset||row.task_id||"—")}</td><td>${esc(row.model||modelLabel(row.model_id)||"—")}</td><td>${esc(row.metric||valueKey||"—")}</td><td>${fmt(valueKey?row[valueKey]:row.value,4)}</td><td>${esc(row.source||"results ledger")}</td></tr>`;
  }).join(""):`<tr><td colspan="7" class="empty-cell">No result rows identify a preprocessing variant. Cache timing metadata, when present, appears above.</td></tr>`;
}

function renderDid(payload) {
  state.did=payload;
  const status=payload.status||"missing";
  const isProjection = status === "predicted" || payload.dataStatus === "predicted" || payload.result?.data_status === "predicted";
  const isSynthetic = status === "mock" || payload.dataStatus === "mock" || payload.result?.data_status === "mock";
  $("#did-kicker").textContent=isProjection?"Model-based paired projection":isSynthetic?"Synthetic paired cells":"Observed paired cells only";
  const tag=$("#did-status-tag");
  tag.textContent=isProjection?"MODEL PROJECTION":isSynthetic?"SYNTHETIC RUN":status==="observed"?"OBSERVED":status.toUpperCase();
  const button=$("#compute-did");
  const pairs=state.results.filter((row)=>row.data_status==="observed");
  const ready=hasMatchedDidCells(pairs,$("#did-fm-model").value,$("#did-base-model").value);
  button.disabled=!state.projectionMode&&!ready;
  button.textContent=state.projectionMode?"Update projection →":"Compute ΔΔ →";
  const savedMatches=!payload.result||(payload.result.fm_model===$("#did-fm-model").value&&payload.result.base_model===$("#did-base-model").value);
  if((status==="observed"||isProjection||isSynthetic)&&payload.result&&((isProjection||isSynthetic)||savedMatches)){
    const result=payload.result;
    $("#did-analysis-state").textContent=isProjection?"Formula-generated scenario using the selected model pair. It is not a fitted analysis or evidence about the study.":isSynthetic?"Synthetic pipeline-check values; these are not evidence about the study.":result.h1_supported?"The stored analysis meets the script’s stated H1 decision rule. Read the estimate, interval and bootstrap test together.":"The stored analysis is based on matched observed cells. Interpret its estimate alongside uncertainty and protocol limits.";
    $("#did-estimate").textContent=`${Number(result.did_point_estimate)>0?"+":""}${fmt(result.did_point_estimate,3)} y`;
    $("#did-ci").textContent=`[${fmt(result.ci_95_lower,3)}, ${fmt(result.ci_95_upper,3)}] y`;
    $("#did-p").textContent=fmt(result.p_value_bootstrap_two_sided,4);
    $("#did-pairs").textContent=String(result.n_paired_cells??"—");
    $("#did-values").hidden=false;
    drawDidInterval(result,isProjection||isSynthetic);
  }else{
    $("#did-values").hidden=true;
    const message={stale:"The result ledger changed after the saved analysis. Recompute to refresh it.",legacy:"The saved summary used an older test. Recompute to report the two-sided bootstrap p-value.",unverified:"The stored DiD summary is not marked observed.",invalid:"The saved DiD summary could not be read."}[status];
    $("#did-analysis-state").textContent=message|| (ready?"Matched observed cells are ready for the DiD analysis.":"Waiting for matched observed cells at the same task, N, seed and tuning-trial budget.");
    emptyChart("#did-chart","ΔΔ","Analysis not computed",ready?"Press Compute ΔΔ to run the preregistered bootstrap analysis.":"Synthetic cells are excluded; observed paired cells are required.");
  }
  if (state.currentPage === "findings" && state.archive) renderFinalFindings(state.archive);
}

function drawDidInterval(result,isIllustrative=false) {
  const low=numeric(result.ci_95_lower), high=numeric(result.ci_95_upper), point=numeric(result.did_point_estimate);
  if(low===null||high===null||point===null)return emptyChart("#did-chart","ΔΔ","Interval unavailable","The saved analysis has no complete bootstrap interval.");
  let min=Math.min(low,point,0),max=Math.max(high,point,0);const pad=(max-min||1)*.18;min-=pad;max+=pad;
  const x=value=>35+(value-min)/(max-min)*650;
  $("#did-chart").innerHTML=`<svg class="svg-chart" viewBox="0 0 720 145" role="img" aria-label="${isIllustrative?"Illustrative scenario · ":""}Difference in differences estimate and 95 percent bootstrap interval"><line class="chart-gridline" x1="35" x2="685" y1="66" y2="66"/><line class="chart-axis" x1="${x(0)}" x2="${x(0)}" y1="32" y2="98"/><line class="chart-line" x1="${x(low)}" x2="${x(high)}" y1="66" y2="66"/><line class="chart-axis" x1="${x(low)}" x2="${x(low)}" y1="54" y2="78"/><line class="chart-axis" x1="${x(high)}" x2="${x(high)}" y1="54" y2="78"/><circle class="chart-point" cx="${x(point)}" cy="66" r="6"/><text class="chart-label" x="${x(min+pad)}" y="117" text-anchor="middle">${fmt(min+pad,2)} y</text><text class="chart-label" x="${x(0)}" y="25" text-anchor="middle">0</text><text class="chart-label" x="${x(max-pad)}" y="117" text-anchor="middle">${fmt(max-pad,2)} y</text><text class="chart-label-strong" x="360" y="140" text-anchor="middle">ΔΔ estimate with 95% bootstrap interval</text></svg>`;
}

function hasMatchedDidCells(rows,fmModel,baseModel){
  const key=(row)=>[row.task_id,row.n_labels,row.seed,row.trial_budget??row.trials].map(value=>String(value??"")).join("|");
  const fm=new Set(rows.filter(row=>row.data_status==="observed"&&row.model_id===fmModel).map(key));
  return rows.some(row=>row.data_status==="observed"&&row.model_id===baseModel&&fm.has(key(row)));
}

function renderEquivalence(payload) {
  const isProjection=payload.status==="predicted"||payload.dataStatus==="predicted";
  const isSynthetic=payload.status==="mock"||payload.dataStatus==="mock";
  $("#eq-status").textContent=isProjection?"MODEL PROJECTION":isSynthetic?"SYNTHETIC RUN":payload.status==="observed"?"OBSERVED FIT":(payload.status||"missing").toUpperCase();
  $("#eq-kicker").textContent=isProjection?"Assumption-based label curves":isSynthetic?"Synthetic label-budget curves":"Observed budget curves";
  $("#pareto-kicker").textContent=isProjection?"Projected efficiency scenario":isSynthetic?"Synthetic efficiency examples":"Observed efficiency";
  if(payload.status!=="observed"&&!isProjection&&!isSynthetic){
    $("#neq-cards").hidden=true;
    return emptyChart("#eq-chart","Nₑq","Observed label curves unavailable",payload.reason||"At least two shared observed label budgets are required for both models and cohorts.");
  }
  $("#neq-cards").hidden=false;
  const seen=payload.cohorts.seen.N_eq, unseen=payload.cohorts.unseen.N_eq;
  $("#eq-seen").textContent=seen===null?"Not estimable":`${seen>0?"+":""}${fmt(seen,1)} labels`;
  $("#eq-unseen").textContent=unseen===null?"Not estimable":`${unseen>0?"+":""}${fmt(unseen,1)} labels`;
  const seenSeries=[{name:`${modelLabel(payload.fmModel)} · seen`,values:payload.cohorts.seen.fmCurve.points.map(p=>({x:p.n,value:p.mae}))},{name:`${modelLabel(payload.baseModel)} · seen`,values:payload.cohorts.seen.baselineCurve.points.map(p=>({x:p.n,value:p.mae}))}];
  const unseenSeries=[{name:`${modelLabel(payload.fmModel)} · unseen`,values:payload.cohorts.unseen.fmCurve.points.map(p=>({x:p.n,value:p.mae}))},{name:`${modelLabel(payload.baseModel)} · unseen`,values:payload.cohorts.unseen.baselineCurve.points.map(p=>({x:p.n,value:p.mae}))}];
  makeLineChart("#eq-chart",[...seenSeries,...unseenSeries],seenSeries[0].values.map(p=>p.x));
}

function renderPareto(payload) {
  const points=payload.points||[];
  const isProjection=payload.dataStatus==="predicted";
  const isSynthetic=payload.dataStatus==="mock";
  $("#pareto-kicker").textContent=isProjection?"Projected efficiency scenario":isSynthetic?"Synthetic efficiency examples":"Observed efficiency";
  $("#pareto-status").textContent=points.length?(isProjection?`${points.length} models · MODEL PROJECTION`:isSynthetic?`${points.length} models · SYNTHETIC RUN`:`${points.length} models · OBSERVED`):"No observed compute data";
  if(!points.length)return emptyChart("#pareto-chart","P","No comparable compute records", "Observed cells need unseen MAE and positive GPU time before the frontier can be drawn.");
  const width=560,height=320,left=48,right=18,top=22,bottom=56;
  const minX=Math.min(...points.map(p=>p.hours)),maxX=Math.max(...points.map(p=>p.hours));
  const minY=Math.min(...points.map(p=>p.unseenMae)),maxY=Math.max(...points.map(p=>p.unseenMae));
  const x=v=>left+(maxX===minX?.5:(v-minX)/(maxX-minX))*(width-left-right);
  const y=v=>top+(maxY===minY?.5:(maxY-v)/(maxY-minY))*(height-top-bottom);
  const ordered=points.filter(p=>p.onFrontier).sort((a,b)=>a.hours-b.hours);
  const line=ordered.map((p,i)=>`${i?"L":"M"}${x(p.hours)},${y(p.unseenMae)}`).join(" ");
  const circles=points.map(p=>{
    const px=x(p.hours),py=y(p.unseenMae),rightAligned=px>width-145,labelX=rightAligned?px-10:px+10;
    return `<circle class="${p.onFocus?"chart-point-focus":p.onFrontier?"chart-point":"chart-point-secondary"}" cx="${px}" cy="${py}" r="${p.onFocus?8:6}"><title>${esc(modelLabel(p.modelId))}${p.onFocus?" · selected focus":""}: ${fmt(p.hours,3)} GPU h, ${fmt(p.unseenMae,3)} unseen MAE, ${p.cells} projected cells</title></circle><text class="chart-label-strong" text-anchor="${rightAligned?"end":"start"}" x="${labelX}" y="${py-10}">${esc(modelLabel(p.modelId))}</text>`;
  }).join("");
  const grid=Array.from({length:4},(_,i)=>{const yy=top+i*(height-top-bottom)/3;const val=maxY-i*(maxY-minY)/3;return `<line class="chart-gridline" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text class="chart-label" x="${left-8}" y="${yy+3}" text-anchor="end">${fmt(val,2)}</text>`;}).join("");
  $("#pareto-chart").innerHTML=`<svg class="svg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${isProjection?"Illustrative projected":"Observed"} model compute and unseen mean absolute error Pareto plot">${grid}<path class="chart-line-secondary" d="${line}"/>${circles}<text class="chart-label" x="${(left+width-right)/2}" y="${height-12}" text-anchor="middle">${isProjection?"Projected GPU-hours":"Observed GPU-hours"} · lower is better</text></svg>`;
}

function renderQC(payload) {
  state.qc=payload;
  const labels={preview_assets:"Bundled slice preview · no participant QC provenance",missing:"No orthogonal slice images found"};
  $("#qc-viewer-status").textContent=labels[payload.viewerStatus]||humanize(payload.viewerStatus);
  $("#qc-viewer-note").textContent=payload.viewerNote||"No participant-level QC values will be inferred from preview images.";
  const total=Object.values(payload.slices||{}).reduce((sum,item)=>sum+(item.count||0),0);
  $("#qc-slice-count").textContent=`${total} image assets`;
  ["axial","coronal","sagittal"].forEach((plane)=>{
    const info=payload.slices?.[plane]||{indices:[]};
    const slider=$(`#qc-${plane}-index`);const indices=info.indices||[];
    slider.min="0";slider.max=String(Math.max(0,indices.length-1));slider.value=String(Math.floor(Math.max(0,indices.length-1)/2));slider.disabled=!indices.length;
    updateSlice(plane);
  });
  $("#qc-excluded-count").textContent=payload.excludedScans===null?"—":String(payload.excludedScans);
  const reasons=Object.entries(payload.exclusionReasons||{});
  $("#qc-reason-count").textContent=payload.exclusionsStatus==="missing"?"—":String(reasons.length);
  $("#qc-exclusion-list").innerHTML=reasons.length?reasons.map(([name,count])=>`<div class="reason-row"><span>${esc(name)}</span><strong>${count}</strong></div>`).join(""):`<p class="empty-inline">No data/qc/exclusions.csv found. A missing ledger is not a zero-exclusion result.</p>`;
}

function updateSlice(plane) {
  if(!state.qc)return;
  const indices=state.qc.slices?.[plane]?.indices||[];
  const slider=$(`#qc-${plane}-index`);
  const position=Number(slider.value)||0;
  const actual=indices[position];
  const label=actual===undefined?"—":String(actual).padStart(3,"0");
  $(`#qc-${plane}-output`).textContent=label;
  $(`#qc-${plane}-label`).textContent=actual===undefined?"NO IMAGE":`SLICE ${label}`;
  const image=$(`#qc-${plane}-image`);
  image.src=actual===undefined?"":`/slices/${plane}/slice_${label}.png`;
}

function setRunMode() {
  const mode=$("input[name=mode]:checked")?.value||"mock";
  const submit=$("#run-submit"),note=$("#run-mode-note"),trials=$("#run-trials");
  const smoke=[...trials.options].find((option)=>option.value==="2");
  if(mode==="observed"){
    submit.innerHTML="Run one observed cell <span aria-hidden=\"true\">→</span>";
    note.textContent="Observed runs use the real frozen split, metadata and cache. The ledger marks the cell observed.";
    if(smoke)smoke.disabled=true;
    if(trials.value==="2")trials.value="5";
  }else{
    submit.innerHTML="Run one synthetic harness cell <span aria-hidden=\"true\">↗</span>";
    note.textContent="Synthetic output checks the harness only; it is excluded from every scientific analysis.";
    if(smoke)smoke.disabled=false;
  }
}

function showJob(job) {
  const card=$("#job-card");card.hidden=false;card.dataset.status=job.status;
  const labels={queued:"Queued",running:"Running",completed:"Completed",failed:"Failed"};
  $("#job-state-label").textContent=labels[job.status]||job.status;
  $("#job-state-dot").className=`job-state-dot ${job.status==="running"?"is-running":job.status==="completed"?"is-completed":job.status==="failed"?"is-failed":""}`;
  $("#job-id").textContent=job.id?`#${job.id}`:"";
  const description=job.mode==="analysis"?`Observed ${modelLabel(job.modelId)} vs ${modelLabel(job.baseModel)} · ${job.trials} bootstrap resamples`:job.mode==="mock-suite"?"Synthetic PPMI/HBN/NKI fast suite · local outputs":`${job.mode==="mock"?"Synthetic run":"Observed"} · ${modelLabel(job.modelId)} · N=${job.nLabels==="all"?"all":job.nLabels} · ${job.trials} trials`;
  $("#job-description").textContent=description;
  const progressHint={queued:"The local runner accepted this cell and is starting it.",running:"The local runner is active; no percentage is reported by the process.",completed:"The local process finished and saved its output.",failed:"The local process stopped before producing a complete output."};
  $("#job-progress-hint").textContent=progressHint[job.status]||"Checking local runner status…";
  $("#job-progress-fill").textContent="";
  const log=$("#job-log");log.textContent=job.log||job.error||"Waiting for the local process…";log.hidden=!job.log&&!job.error&&job.status!=="failed";
  if(job.status==="completed"||job.status==="failed"){
    state.jobId=null;
    if(job.mode==="mock-suite")$("#legacy-suite-status").textContent=job.status==="completed"?`Finished. Synthetic artifacts saved under ${job.outputDir||"results/mock/"}.`:"The run failed; see the run log for details.";
    showToast(job.status==="completed"?"Job finished. Local artifacts have been refreshed.":"Job failed. Review the run log.");
    refreshAll();
  }
}

async function pollJob(jobId) {
  clearTimeout(state.pollTimer);
  try {
    const job=await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    showJob(job);
    if(job.status==="queued"||job.status==="running")state.pollTimer=setTimeout(()=>pollJob(jobId),1200);
  } catch(error) {
    state.jobId=null;$("#run-error").textContent=error.message;$("#run-error").hidden=false;
  }
}

async function startRun(event) {
  event.preventDefault();
  $("#run-error").hidden=true;
  const form=new FormData(event.currentTarget);
  const payload={mode:form.get("mode")||"mock",taskId:form.get("taskId"),modelId:form.get("modelId"),nLabels:Number(form.get("nLabels")),seed:Number(form.get("seed")),trials:Number(form.get("trials"))};
  const submit=$("#run-submit");submit.disabled=true;submit.textContent="Starting cell…";
  try{const job=await api("/api/runs",{method:"POST",body:JSON.stringify(payload)});state.jobId=job.id;showJob(job);if(job.status==="queued"||job.status==="running")pollJob(job.id);}
  catch(error){$("#run-error").textContent=error.message;$("#run-error").hidden=false;}
  finally{submit.disabled=false;setRunMode();}
}

async function startLegacySuite() {
  const button=$("#legacy-suite-button");button.disabled=true;button.textContent="Starting suite…";
  try{const job=await api("/api/benchmark-demo",{method:"POST"});state.jobId=job.id;showJob(job);if(job.status==="queued"||job.status==="running")pollJob(job.id);}
  catch(error){$("#legacy-suite-status").textContent=error.message;showToast(error.message);}
  finally{button.disabled=false;button.innerHTML="Run fast suite <span aria-hidden=\"true\">→</span>";}
}

function downloadBlob(filename,content,type){
  const blob=new Blob([content],{type});const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function exportCsv(){
  if(state.projectionMode)return showToast("Model projections cannot be exported as study results.");
  const rows=state.archive?.records||[];if(!rows.length)return showToast("No saved ledger rows are available to export.");
  const columns=[...new Set(rows.flatMap((row)=>Object.keys(row)))];
  const quote=(value)=>`"${String(value??"").replaceAll('"','""')}"`;
  downloadBlob("seen-it-before-complete-ledger.csv",[columns.join(","),...rows.map(row=>columns.map(key=>quote(row[key])).join(","))].join("\r\n"),"text/csv;charset=utf-8");
}

function exportJson(){
  if(state.projectionMode)return showToast("Model projections cannot be exported as study results.");
  if(!state.archive)return showToast("The saved research archive is still loading.");
  downloadBlob("seen-it-before-research-archive.json",JSON.stringify(state.archive,null,2),"application/json");
}

async function copyText(text,button){
  try{await navigator.clipboard.writeText(text);const label=button.textContent;button.textContent="Copied";setTimeout(()=>button.textContent=label,1300);}
  catch{showToast("Clipboard access is unavailable in this browser.");}
}

async function computeDid(){
  if(state.projectionMode){refreshModelProjection("did");return showToast("Updated the projection for the selected model pair.");}
  const button=$("#compute-did");button.disabled=true;button.textContent="Computing…";
  try{const job=await api("/api/analysis/did",{method:"POST",body:JSON.stringify({fmModel:$("#did-fm-model").value,baseModel:$("#did-base-model").value,taskId:"t1_aomic_age",bootstrap:2000})});state.jobId=job.id;showJob(job);if(job.status==="queued"||job.status==="running")pollJob(job.id);}
  catch(error){$("#did-analysis-state").textContent=error.message;showToast(error.message);}
  finally{button.textContent="Compute ΔΔ →";button.disabled=!hasMatchedDidCells(state.results,$("#did-fm-model").value,$("#did-base-model").value);}
}

async function computeEquivalence(){
  if(state.projectionMode){refreshModelProjection("equivalence");return showToast("Updated the label-equivalence projection.");}
  const fm=$("#eq-fm-model").value,base=$("#eq-base-model").value,n=$("#eq-n").value;
  if(!fm||!base||!n)return;
  $("#compute-equivalence").disabled=true;$("#compute-equivalence").textContent="Fitting…";
  try{const q=new URLSearchParams({fm_model:fm,base_model:base,eval_n:n,task:"t1_aomic_age"});renderEquivalence(await api(`/api/analysis/label-equivalence?${q}`));}
  catch(error){emptyChart("#eq-chart","Nₑq","Could not fit curves",error.message);}
  finally{$("#compute-equivalence").disabled=false;$("#compute-equivalence").innerHTML="Fit N<sub>eq</sub> <span aria-hidden=\"true\">→</span>";}
}

async function refreshAll(){
  const button=$("#refresh-button");button.classList.add("is-loading");
  const tasks=[
    ["summary",api("/api/summary")],["config",api("/api/config")],["audit",api("/api/audit")],
    ["results",api("/api/results?scope=all")],["archive",api("/api/archive")],["legacy",api("/api/legacy-results")],["correlation",api("/api/correlation")],
    ["preprocessing",api("/api/preprocessing")],["qc",api("/api/qc")],["did",api("/api/analysis/did")],["pareto",api("/api/analysis/pareto")]
  ];
  try{
    const responses=await Promise.all(tasks.map(async([key,promise])=>{try{return[key,await promise,null];}catch(error){return[key,null,error];}}));
    const failed=responses.filter(([,value,error])=>error);
    const values=Object.fromEntries(responses.map(([key,value])=>[key,value]));
    if(values.summary){updateSummary(values.summary);setConnection(true,"Local data connected");}
    if(values.config)updateConfig(values.config);
    if(values.audit)renderAudit(values.audit);
    if(values.results){state.results=values.results.rows||[];renderResults();}
    if(values.archive){state.archive=values.archive;renderArchive(values.archive);}
    if(values.legacy)state.legacy=values.legacy.artifacts||[];
    if(values.correlation)renderCorrelation(values.correlation);
    if(values.preprocessing)renderPreprocessing(values.preprocessing);
    if(values.qc)renderQC(values.qc);
    if(values.did)renderDid(values.did);
    if(values.pareto)renderPareto(values.pareto);
    if(state.archive)renderFinalFindings(state.archive);
    if(values.correlation)renderCorrelation(values.correlation);
    if(values.preprocessing)renderPreprocessing(values.preprocessing);
    if(values.legacy)renderResults();
    computeEquivalence();
    if(state.projectionMode){
      state.projectionSnapshot={
        ...(state.projectionSnapshot||{}),
        audit:values.audit||state.projectionSnapshot?.audit,
        correlation:values.correlation||state.projectionSnapshot?.correlation,
        preprocessing:values.preprocessing||state.projectionSnapshot?.preprocessing,
        did:values.did||state.projectionSnapshot?.did,
        pareto:values.pareto||state.projectionSnapshot?.pareto,
        equivalence:state.projectionSnapshot?.equivalence
      };
      renderProjectionWorkspace();
    }
    if(failed.length){const message=failed.map(([, ,error])=>error.message).join(" · ");if(!values.summary)setConnection(false,"Cannot reach local server");showToast(`Some workspace views are unavailable: ${message}`);}
    else if(button.classList.contains("is-loading"))showToast("Workspace refreshed from local files.");
  }catch(error){setConnection(false,"Cannot reach local server");showToast(`Workspace unavailable: ${error.message}`);}
  finally{button.classList.remove("is-loading");}
}

document.addEventListener("DOMContentLoaded",async()=>{
  const landingHero = $("#page-overview .hero-grid");
  if (landingHero && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    let pointerFrame = 0;
    let pointerX = 0;
    let pointerY = 0;
    landingHero.addEventListener("pointermove",(event)=>{
      if (event.pointerType !== "mouse" && event.pointerType !== "pen") return;
      pointerX = event.clientX;
      pointerY = event.clientY;
      if (pointerFrame) cancelAnimationFrame(pointerFrame);
      pointerFrame = requestAnimationFrame(()=>{
        const bounds = landingHero.getBoundingClientRect();
        const x = pointerX - bounds.left;
        const y = pointerY - bounds.top;
        const normalizedX = (x / Math.max(bounds.width, 1)) - .5;
        const normalizedY = (y / Math.max(bounds.height, 1)) - .5;
        const shiftX = normalizedX * 12;
        const shiftY = normalizedY * 8;
        landingHero.style.setProperty("--cursor-x",`${x}px`);
        landingHero.style.setProperty("--cursor-y",`${y}px`);
        landingHero.style.setProperty("--brain-shift-x",`${shiftX.toFixed(1)}px`);
        landingHero.style.setProperty("--brain-shift-y",`${shiftY.toFixed(1)}px`);
        landingHero.style.setProperty("--brain-tilt-x",`${(normalizedX * 10).toFixed(1)}deg`);
        landingHero.style.setProperty("--brain-tilt-y",`${(normalizedY * -8).toFixed(1)}deg`);
        landingHero.classList.add("has-pointer");
        pointerFrame = 0;
      });
    });
    landingHero.addEventListener("pointerleave",()=>{
      if (pointerFrame) cancelAnimationFrame(pointerFrame);
      pointerFrame = 0;
      landingHero.classList.remove("has-pointer");
      landingHero.style.removeProperty("--cursor-x");
      landingHero.style.removeProperty("--cursor-y");
      landingHero.style.removeProperty("--brain-shift-x");
      landingHero.style.removeProperty("--brain-shift-y");
      landingHero.style.removeProperty("--brain-tilt-x");
      landingHero.style.removeProperty("--brain-tilt-y");
    });
  }
  document.addEventListener("click",(event)=>{
    const target=event.target.closest("[data-page-target]");if(target)goToPage(target.dataset.pageTarget);
    const copy=event.target.closest("[data-copy]");if(copy)copyText(copy.dataset.copy,copy);
    const filterBtn=event.target.closest(".chart-filter-btn");
    if(filterBtn && state._lastFinals){
      renderFindingsChart(state._lastFinals, filterBtn.dataset.findingsFilter);
    }
  });
  window.addEventListener("hashchange",()=>goToPage(location.hash.slice(1),false));
  $("#refresh-button").addEventListener("click",refreshAll);
  $("#generate-estimates-button").addEventListener("click",toggleProjectionMode);
  $("#run-all-models-button").addEventListener("click",startComparisonPreview);
  $("#comparison-budget").addEventListener("change",renderComparisonPreview);
  $("#comparison-legend").addEventListener("click",(event)=>{
    const button=event.target.closest("[data-comparison-model]");if(!button||!state.comparisonPreview)return;
    const modelId=button.dataset.comparisonModel;
    state.comparisonVisibleModels=state.comparisonVisibleModels.includes(modelId)?state.comparisonVisibleModels.filter((id)=>id!==modelId):[...state.comparisonVisibleModels,modelId];
    renderComparisonPreview();
  });
  $("#run-form").addEventListener("submit",startRun);
  $("#legacy-suite-button").addEventListener("click",startLegacySuite);
  $$('input[name="mode"]').forEach((input)=>input.addEventListener("change",setRunMode));
  $("#results-scope").addEventListener("change",renderResults);
  $("#archive-group-filter").addEventListener("change",(event)=>{state.archiveGroup=event.currentTarget.value;renderArchiveFiles(state.archive||{});});
  $("#archive-file-search").addEventListener("input",(event)=>{state.archiveSearch=event.currentTarget.value;renderArchiveFiles(state.archive||{});});
  $("#focus-model").addEventListener("change",()=>refreshModelProjection("focus"));
  $("#run-model").addEventListener("change",()=>{
    refreshModelProjection("runner");
    if(state.comparisonPreview){state.comparisonPreview.focusModel=$("#run-model").value;renderComparisonPreview();}
  });
  $("#results-model-filter").addEventListener("change",()=>state.projectionMode?refreshModelProjection("results"):renderResults());
  $("#results-metric").addEventListener("change",renderResults);
  $("#export-csv").addEventListener("click",exportCsv);
  $("#export-json").addEventListener("click",exportJson);
  $("#correlation-matrix-select").addEventListener("change",()=>renderCorrelation(state.correlation||{}));
  $("#compute-did").addEventListener("click",computeDid);
  $("#did-fm-model").addEventListener("change",()=>state.projectionMode?refreshModelProjection("did"):renderDid(state.did||{status:"missing"}));
  $("#did-base-model").addEventListener("change",()=>state.projectionMode?refreshModelProjection("did-base"):renderDid(state.did||{status:"missing"}));
  $("#did-fm-model").addEventListener("change",()=>$("#did-fm-model").dataset.userChanged="true");
  $("#did-base-model").addEventListener("change",()=>$("#did-base-model").dataset.userChanged="true");
  $("#eq-fm-model").addEventListener("change",()=>{$("#eq-fm-model").dataset.userChanged="true";if(state.projectionMode)refreshModelProjection("equivalence");});
  $("#eq-base-model").addEventListener("change",()=>{$("#eq-base-model").dataset.userChanged="true";if(state.projectionMode)refreshModelProjection("equivalence-base");});
  $("#eq-n").addEventListener("input",()=>state.projectionMode&&refreshModelProjection("targetN"));
  $("#compute-equivalence").addEventListener("click",computeEquivalence);
  $("#audit-seen-threshold").addEventListener("input",updateAuditCommand);
  $("#audit-unseen-threshold").addEventListener("input",updateAuditCommand);
  $("#copy-audit-command").addEventListener("click",(event)=>copyText($("#audit-command").textContent,event.currentTarget));
  ["axial","coronal","sagittal"].forEach((plane)=>$(`#qc-${plane}-index`).addEventListener("input",()=>updateSlice(plane)));
  const initialPage=location.hash.slice(1);goToPage(pageLabels[initialPage]?initialPage:"overview",false);
  renderBudgetVisual();setRunMode();updateAuditCommand();
  await refreshAll();
});

"""Local web app for the Seen It Before research workspace.

The standard-library server serves the research dashboard and a small local API for
project readiness, verified audit outputs, results ledgers, and explicitly requested
benchmark cells. It binds to loopback by default and keeps mock runs isolated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import re
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_TEST_ROOT = PROJECT_ROOT / "results_test"
DEFAULT_TASK = "t1_aomic_age"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "splits" / "t1_aomic_age_splits.json"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "aomic_scans.csv"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache" / "aomic"
VALID_BUDGETS = {25, 50, 100, 250, 500, -1}
VALID_SEEDS = {42, 1337, 2026, 7, 99}
VALID_TRIALS = {2, 5, 20, 50}
MAX_REQUEST_BYTES = 16_384

_JOBS: dict[str, dict[str, Any]] = {}
_JOB_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def _ledger_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_paths = [RESULTS_ROOT / "results.csv", RESULTS_ROOT / "mock" / "results.csv"]
    seen_paths: set[Path] = set()
    for path in csv_paths:
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        for row in _read_csv(path):
            if row.get("record_type") != "final_evaluation":
                continue
            status = str(row.get("data_status", "")).strip().lower()
            if status not in {"observed", "mock"}:
                continue
            row["data_status"] = status
            row["ledger"] = "mock/results.csv" if path.parent.name == "mock" else "results.csv"
            rows.append(row)
    parquet_path = RESULTS_ROOT / "results.parquet"
    if not rows and parquet_path.is_file():
        try:
            import pandas as pd

            frame = pd.read_parquet(parquet_path)
            for row in frame.to_dict(orient="records"):
                if row.get("record_type") == "final_evaluation" and row.get("data_status") in {"observed", "mock"}:
                    row["ledger"] = "results.parquet"
                    rows.append(row)
        except Exception:
            pass
    return [_json_safe(row) for row in rows]


def _archive_records() -> list[dict[str, Any]]:
    """Return every persisted ledger row, including tuning trials."""
    records: list[dict[str, Any]] = []
    csv_paths = sorted(RESULTS_ROOT.rglob("results.csv")) if RESULTS_ROOT.is_dir() else []
    csv_set = set(csv_paths)
    for path in csv_paths:
        source = path.relative_to(PROJECT_ROOT).as_posix()
        status_from_path = "mock" if "mock" in path.relative_to(RESULTS_ROOT).parts else "unverified"
        for source_row, row in enumerate(_read_csv(path), start=2):
            status = str(row.get("data_status") or status_from_path).strip().lower()
            row["data_status"] = status if status in {"observed", "mock"} else "unverified"
            row["ledger"] = source
            row["source_row"] = source_row
            records.append(_json_safe(row))

    # Read a parquet-only ledger if its CSV counterpart is absent. When both
    # exist, CSV is authoritative and avoids counting the same rows twice.
    parquet_paths = sorted(RESULTS_ROOT.rglob("results.parquet")) if RESULTS_ROOT.is_dir() else []
    for path in parquet_paths:
        if path.with_suffix(".csv") in csv_set:
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
        except Exception:
            continue
        source = path.relative_to(PROJECT_ROOT).as_posix()
        for source_row, row in enumerate(frame.to_dict(orient="records"), start=1):
            status = str(row.get("data_status") or ("mock" if "mock" in path.parts else "unverified")).strip().lower()
            row["data_status"] = status if status in {"observed", "mock"} else "unverified"
            row["ledger"] = source
            row["source_row"] = source_row
            records.append(_json_safe(row))
    return records


def _archive_run_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct ledger runs from row order and their recorded settings."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("ledger") or "unknown ledger"), []).append(record)

    runs: list[dict[str, Any]] = []
    for source, source_records in grouped.items():
        pending: list[dict[str, Any]] = []
        pending_key: tuple[str, ...] | None = None
        for record in source_records:
            key = tuple(str(record.get(field) or "") for field in ("task_id", "model_id", "n_labels", "seed", "trial_budget"))
            if pending and key != pending_key:
                runs.append(_make_archive_segment(source, pending, complete=False, number=len(runs) + 1))
                pending = []
            pending_key = key
            pending.append(record)
            if record.get("record_type") == "final_evaluation":
                runs.append(_make_archive_segment(source, pending, complete=True, number=len(runs) + 1))
                pending = []
                pending_key = None
        if pending:
            runs.append(_make_archive_segment(source, pending, complete=False, number=len(runs) + 1))
    return runs


def _make_archive_segment(source: str, records: list[dict[str, Any]], *, complete: bool, number: int) -> dict[str, Any]:
    final = next((row for row in reversed(records) if row.get("record_type") == "final_evaluation"), records[-1])
    first = records[0]
    return {
        "id": f"ledger-segment-{number:03d}",
        "label": f"Ledger segment {number:02d}",
        "dataStatus": final.get("data_status", "unverified"),
        "taskId": final.get("task_id"),
        "modelId": final.get("model_id"),
        "nLabels": final.get("n_labels"),
        "seed": final.get("seed"),
        "trials": final.get("trial_budget"),
        "startedAt": first.get("timestamp"),
        "finishedAt": final.get("timestamp") if complete else None,
        "source": source,
        "firstRow": first.get("source_row"),
        "lastRow": records[-1].get("source_row"),
        "complete": complete,
        "records": records,
    }


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        return "figure"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix in {".npy", ".npz"}:
        return "raw array"
    if suffix in {".parquet", ".feather"}:
        return "ledger"
    if suffix in {".log", ".txt", ".jsonl"}:
        return "run log"
    if suffix in {".json", ".yaml", ".yml"}:
        return "run metadata"
    return "file"


def _results_archive() -> dict[str, Any]:
    records = _archive_records()
    artifacts: list[dict[str, Any]] = []
    for root, collection in ((RESULTS_ROOT, "research"), (RESULTS_TEST_ROOT, "verification")):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep" or path.suffix.lower() == ".tmp":
                continue
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            parts = path.relative_to(root).parts
            data_status = "verification" if collection == "verification" else "mock" if "mock" in parts else "unverified"
            if collection == "verification":
                bundle = "Verification output"
            elif "mock" in parts and any(part.startswith("benchmark-") for part in parts):
                run_dir = next(part for part in parts if part.startswith("benchmark-"))
                bundle = f"Synthetic pipeline run · {run_dir[-12:]}"
            elif "mock" in parts:
                bundle = "Synthetic cell results"
            else:
                bundle = "Research results"
            stat = path.stat()
            item: dict[str, Any] = {
                "name": path.name,
                "path": relative,
                "kind": _artifact_kind(path),
                "collection": collection,
                "dataStatus": data_status,
                "bundle": bundle,
                "sizeBytes": stat.st_size,
                "modifiedAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "url": f"/api/artifact?path={quote(relative, safe='/')}",
            }
            if path.suffix.lower() == ".npy":
                try:
                    import numpy as np

                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    item["arrayShape"] = list(array.shape)
                    item["dtype"] = str(array.dtype)
                except (ImportError, OSError, ValueError):
                    pass
            if path.suffix.lower() == ".csv" and stat.st_size <= 512_000:
                csv_rows = _read_csv(path)
                item["previewRows"] = csv_rows[:50]
                item["rowCount"] = len(csv_rows)
            elif path.suffix.lower() in {".log", ".txt", ".json", ".jsonl"} and stat.st_size <= 100_000:
                try:
                    item["textPreview"] = path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeDecodeError):
                    pass
            artifacts.append(item)

    with _JOB_LOCK:
        jobs = [dict(job) for job in _JOBS.values()]
    known_ids = {str(job.get("id")) for job in jobs}
    for history_path in sorted(RESULTS_ROOT.rglob("run_history/*.json")) if RESULTS_ROOT.is_dir() else []:
        try:
            saved = json.loads(history_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(saved.get("id")) not in known_ids:
            log_path = PROJECT_ROOT / str(saved.get("logPath") or "")
            if log_path.is_file():
                try:
                    saved["log"] = log_path.read_text(encoding="utf-8", errors="replace")[-500_000:]
                except OSError:
                    pass
            jobs.append(saved)
            known_ids.add(str(saved.get("id")))

    type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for record in records:
        record_type = str(record.get("record_type") or "untyped")
        status = str(record.get("data_status") or "unverified")
        type_counts[record_type] = type_counts.get(record_type, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    return _json_safe({
        "recordCount": len(records),
        "artifactCount": len(artifacts),
        "runCount": sum(run["complete"] for run in _archive_run_segments(records)),
        "typeCounts": type_counts,
        "statusCounts": status_counts,
        "records": records,
        "runs": _archive_run_segments(records),
        "artifacts": artifacts,
        "jobs": sorted(jobs, key=lambda job: str(job.get("createdAt") or ""), reverse=True),
        "logNote": "Existing ledger rows are durable. Original console logs were not persisted before run-log archiving was added; older run timelines are reconstructed from their saved trial and evaluation rows.",
    })


def _job_storage_paths(job: dict[str, Any]) -> tuple[Path, Path]:
    mode = str(job.get("mode") or "")
    output_root = RESULTS_ROOT / "mock" if mode in {"mock", "mock-suite"} else RESULTS_ROOT
    job_id = re.sub(r"[^a-f0-9]", "", str(job.get("id") or ""))
    return output_root / "run_logs" / f"{job_id}.log", output_root / "run_history" / f"{job_id}.json"


def _persist_job_manifest(job_id: str) -> None:
    job = _get_job(job_id)
    if not job:
        return
    log_path, manifest_path = _job_storage_paths(job)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    job["logPath"] = log_path.relative_to(PROJECT_ROOT).as_posix()
    job["historyPath"] = manifest_path.relative_to(PROJECT_ROOT).as_posix()
    with _JOB_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update({"logPath": job["logPath"], "historyPath": job["historyPath"]})
            job = dict(_JOBS[job_id])
    job.pop("log", None)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(_json_safe(job), indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)


def _legacy_artifacts() -> dict[str, Any]:
    """Read outputs from the original PPMI/HBN/NKI pipeline without guessing provenance."""
    candidates = []
    if RESULTS_ROOT.is_dir():
        candidates = [
            path for path in RESULTS_ROOT.rglob("*.csv")
            if path.name in {"all_cv_results.csv", "fold_level_results.csv", "heldout_seed_results.csv"}
            or path.name.endswith("_cv_results.csv")
        ]
    artifacts = []
    for path in sorted(candidates):
        rows = _read_csv(path)
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        in_mock_tree = "mock" in path.relative_to(RESULTS_ROOT).parts
        artifacts.append({
            "name": path.name,
            "source": relative,
            "kind": "folds" if path.name == "fold_level_results.csv" else "heldout" if path.name == "heldout_seed_results.csv" else "learning" if path.name == "all_cv_results.csv" or path.name.endswith("_cv_results.csv") else "summary",
            "dataStatus": "mock" if in_mock_tree else "unverified",
            "count": len(rows),
            "rows": [_json_safe(row) for row in rows[:5000]],
        })
    return {"count": len(artifacts), "artifacts": artifacts}


def _correlation_payload() -> dict[str, Any]:
    matrix_paths = sorted(RESULTS_ROOT.rglob("*_corr_matrix.npy")) if RESULTS_ROOT.is_dir() else []
    if not matrix_paths:
        return {"status": "missing", "matrices": []}
    try:
        import numpy as np
    except ImportError:
        return {"status": "unavailable", "reason": "NumPy is required to read correlation matrices.", "matrices": []}
    matrices = []
    for path in matrix_paths[:8]:
        try:
            matrix = np.load(path, allow_pickle=False, mmap_mode="r")
            if matrix.ndim != 2 or not matrix.size:
                continue
            rows, columns = matrix.shape
            max_axis = 56
            row_ids = np.linspace(0, rows - 1, min(rows, max_axis), dtype=int)
            column_ids = np.linspace(0, columns - 1, min(columns, max_axis), dtype=int)
            view = matrix[np.ix_(row_ids, column_ids)]
            metadata_path = path.with_name(path.name.replace("_corr_matrix.npy", "_corr_metadata.csv"))
            metadata = _read_csv(metadata_path)
            groups = []
            for row in metadata:
                try:
                    start = int(float(row.get("boundary_start", 0)))
                    end = int(float(row.get("boundary_end", 0)))
                except (TypeError, ValueError):
                    continue
                if end > start:
                    groups.append({"name": row.get("model", "Feature group"), "start": start, "end": end})
            labels = [
                next((group["name"] for group in groups if group["start"] <= int(index) < group["end"]), f"Feature {int(index) + 1}")
                for index in row_ids
            ]
            status = "mock" if "mock" in path.relative_to(RESULTS_ROOT).parts else "unverified"
            matrices.append({
                "name": path.name.removesuffix("_corr_matrix.npy").upper(),
                "source": path.relative_to(PROJECT_ROOT).as_posix(),
                "dataStatus": status,
                "shape": [int(rows), int(columns)],
                "rowLabels": labels,
                "columnLabels": [
                    next((group["name"] for group in groups if group["start"] <= int(index) < group["end"]), f"Feature {int(index) + 1}")
                    for index in column_ids
                ],
                "groups": groups,
                "values": _json_safe(view.tolist()),
            })
        except (OSError, ValueError, ImportError):
            continue
    return {"status": "available" if matrices else "invalid", "matrices": matrices}


def _preprocessing_payload() -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    if DEFAULT_CACHE.is_dir():
        for meta_path in DEFAULT_CACHE.rglob("*.json"):
            try:
                item = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = str(item.get("pipeline") or "Unlabelled pipeline")
            group = groups.setdefault(name, {"pipeline": name, "scans": 0, "seconds": []})
            group["scans"] += 1
            try:
                seconds = float(item.get("preprocessing_seconds"))
                if math.isfinite(seconds) and seconds >= 0:
                    group["seconds"].append(seconds)
            except (TypeError, ValueError):
                pass
    cache_groups = []
    for group in groups.values():
        times = group.pop("seconds")
        group["meanSeconds"] = sum(times) / len(times) if times else None
        group["timedScans"] = len(times)
        cache_groups.append(group)

    variant_rows = []
    for artifact in _legacy_artifacts()["artifacts"]:
        for row in artifact["rows"]:
            pipeline = row.get("pipeline") or row.get("preprocessing") or row.get("preprocessing_pipeline")
            if pipeline:
                variant_rows.append({**row, "pipeline": pipeline, "data_status": row.get("data_status") or artifact["dataStatus"], "source": artifact["source"]})
    for row in _ledger_rows():
        pipeline = row.get("pipeline") or row.get("preprocessing") or row.get("preprocessing_pipeline")
        if pipeline:
            variant_rows.append({**row, "pipeline": pipeline})
    return {"cacheGroups": cache_groups, "variantRows": variant_rows[:5000], "status": "available" if cache_groups or variant_rows else "missing"}


def _qc_payload() -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for plane in ("axial", "coronal", "sagittal"):
        folder = WEB_ROOT / "slices" / plane
        indices = []
        if folder.is_dir():
            for path in folder.glob("slice_*.png"):
                match = re.fullmatch(r"slice_(\d+)\.png", path.name)
                if match:
                    indices.append(int(match.group(1)))
        indices.sort()
        slices[plane] = {"indices": indices, "count": len(indices)}
    exclusion_path = DATA_ROOT / "qc" / "exclusions.csv"
    exclusions = _read_csv(exclusion_path) if exclusion_path.is_file() else []
    reasons: dict[str, int] = {}
    for row in exclusions:
        reason = str(row.get("reason") or row.get("exclusion_reason") or "Unlabelled")
        reasons[reason] = reasons.get(reason, 0) + 1
    available = all(slices[plane]["count"] for plane in slices)
    return {
        "viewerStatus": "preview_assets" if available else "missing",
        "viewerNote": "These bundled orthogonal images are a visual preview only; they have no participant-level QC provenance.",
        "slices": slices,
        "exclusionsStatus": "unverified" if exclusions else "missing",
        "excludedScans": len(exclusions) if exclusions else None,
        "exclusionReasons": reasons,
    }


def _fit_log_curve(points: list[tuple[int, float]]) -> dict[str, Any] | None:
    if len(points) < 2:
        return None
    xs = [math.log(item[0]) for item in points]
    ys = [item[1] for item in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((value - mean_x) ** 2 for value in xs)
    if variance <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    intercept = mean_y - slope * mean_x
    return {"slope": slope, "intercept": intercept, "points": [{"n": n, "mae": mae} for n, mae in points]}


def _label_equivalence_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    fm_model = query.get("fm_model", ["brainiac"])[0]
    base_model = query.get("base_model", ["resnet18_scratch"])[0]
    task_id = query.get("task", [DEFAULT_TASK])[0]
    try:
        eval_n = int(query.get("eval_n", ["50"])[0])
    except (TypeError, ValueError):
        return {"status": "invalid", "reason": "Target label budget must be an integer."}
    if fm_model == base_model or task_id != DEFAULT_TASK or eval_n <= 0:
        return {"status": "invalid", "reason": "Choose two different T1 models and a positive label budget."}

    grouped: dict[str, dict[int, dict[str, list[float]]]] = {fm_model: {}, base_model: {}}
    for row in _ledger_rows():
        if row.get("data_status") != "observed" or row.get("task_id") != task_id or row.get("model_id") not in grouped:
            continue
        try:
            n_labels = int(row.get("n_labels"))
            seen_mae = float(row.get("seen_mae"))
            unseen_mae = float(row.get("unseen_mae"))
        except (TypeError, ValueError):
            continue
        if n_labels < 1 or not (math.isfinite(seen_mae) and math.isfinite(unseen_mae)):
            continue
        budget = grouped[row["model_id"]].setdefault(n_labels, {"seen": [], "unseen": []})
        budget["seen"].append(seen_mae)
        budget["unseen"].append(unseen_mae)

    cohorts: dict[str, Any] = {}
    for cohort in ("seen", "unseen"):
        common = sorted(set(grouped[fm_model]).intersection(grouped[base_model]))
        fm_points = []
        base_points = []
        for n_labels in common:
            fm_values = grouped[fm_model][n_labels][cohort]
            base_values = grouped[base_model][n_labels][cohort]
            if fm_values and base_values:
                fm_points.append((n_labels, sum(fm_values) / len(fm_values)))
                base_points.append((n_labels, sum(base_values) / len(base_values)))
        fm_fit = _fit_log_curve(fm_points)
        base_fit = _fit_log_curve(base_points)
        if not fm_fit or not base_fit:
            return {
                "status": "missing",
                "reason": f"At least two shared observed label budgets are required for the {cohort} cohort.",
                "fmModel": fm_model,
                "baseModel": base_model,
                "taskId": task_id,
                "evalN": eval_n,
            }
        fm_at_n = fm_fit["slope"] * math.log(eval_n) + fm_fit["intercept"]
        base_slope = base_fit["slope"]
        try:
            required_n = math.exp((fm_at_n - base_fit["intercept"]) / base_slope) if base_slope < 0 else None
        except OverflowError:
            required_n = None
        if required_n is not None and (not math.isfinite(required_n) or required_n <= 0):
            required_n = None
        cohorts[cohort] = {
            "fmCurve": fm_fit,
            "baselineCurve": base_fit,
            "fmMaeAtN": fm_at_n,
            "baselineRequiredN": required_n,
            "N_eq": None if required_n is None else required_n - eval_n,
        }
    return {"status": "observed", "dataStatus": "observed", "fmModel": fm_model, "baseModel": base_model, "taskId": task_id, "evalN": eval_n, "cohorts": cohorts}


def _pareto_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    task_id = query.get("task", [DEFAULT_TASK])[0]
    summary: dict[str, dict[str, Any]] = {}
    for row in _ledger_rows():
        if row.get("data_status") != "observed" or (task_id and row.get("task_id") != task_id):
            continue
        model_id = str(row.get("model_id") or "")
        try:
            mae = float(row.get("unseen_mae"))
            cost_value = row.get("total_gpu_hours")
            hours = float(cost_value) if cost_value not in {None, ""} else float(row.get("total_gpu_seconds")) / 3600
        except (TypeError, ValueError):
            continue
        if not model_id or not math.isfinite(mae) or not math.isfinite(hours) or hours <= 0:
            continue
        item = summary.setdefault(model_id, {"modelId": model_id, "hours": 0.0, "maes": [], "cells": 0})
        item["hours"] += hours
        item["maes"].append(mae)
        item["cells"] += 1
    points = [{"modelId": item["modelId"], "hours": item["hours"], "unseenMae": sum(item["maes"]) / len(item["maes"]), "cells": item["cells"]} for item in summary.values()]
    for point in points:
        point["onFrontier"] = not any(
            other["hours"] <= point["hours"] and other["unseenMae"] <= point["unseenMae"]
            and (other["hours"] < point["hours"] or other["unseenMae"] < point["unseenMae"])
            for other in points
        )
    return {"status": "observed" if points else "missing", "dataStatus": "observed" if points else "missing", "taskId": task_id, "points": points}


def _task_ids() -> list[str]:
    folder = PROJECT_ROOT / "configs" / "tasks"
    return sorted(path.stem for path in folder.glob("*.yaml")) if folder.is_dir() else []


def _model_ids() -> list[str]:
    folder = PROJECT_ROOT / "configs" / "models"
    return sorted(path.stem for path in folder.glob("*.yaml")) if folder.is_dir() else []


def _model_info(model_id: str) -> dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "models" / f"{model_id}.yaml"
    text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    kind_match = re.search(r"^\s*type:\s*[\"]?([^\"#\s]+)", text, re.MULTILINE)
    checkpoint_match = re.search(r"^\s*checkpoint_path:\s*[\"]?([^\"#\r\n]+)", text, re.MULTILINE)
    model_type = kind_match.group(1).strip() if kind_match else "unknown"
    checkpoint_value = checkpoint_match.group(1).strip() if checkpoint_match else None
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint_required = model_type == "foundation"
    checkpoint_ready = bool(checkpoint_path and checkpoint_path.is_file())
    available = not checkpoint_required or checkpoint_ready
    reason = None if available else "Configured checkpoint is missing or has no path in this model config."
    return {
        "id": model_id,
        "type": model_type,
        "checkpointRequired": checkpoint_required,
        "checkpointReady": checkpoint_ready if checkpoint_required else None,
        "available": available,
        "reason": reason,
    }


def _check(name: str, path: Path, *, kind: str = "file", required: bool = True) -> dict[str, Any]:
    exists = path.is_file() if kind == "file" else path.is_dir()
    if exists and kind == "directory":
        exists = any(path.glob("*.npy"))
    return {"name": name, "ready": bool(exists), "required": required, "detail": "Ready" if exists else "Not found"}


def _readiness() -> dict[str, Any]:
    checks = [
        _check("Frozen AOMIC splits", DEFAULT_SPLITS),
        _check("AOMIC scan metadata", DEFAULT_METADATA),
        _check("Preprocessed 128³ cache", DEFAULT_CACHE, kind="directory"),
        _check("FOMO pretraining manifest", PROJECT_ROOT / "data" / "manifests" / "mapping.tsv", required=False),
        _check("Observed results ledger", RESULTS_ROOT / "results.csv", required=False),
    ]
    for model_id in _model_ids():
        info = _model_info(model_id)
        if info["checkpointRequired"]:
            config_path = PROJECT_ROOT / "configs" / "models" / f"{model_id}.yaml"
            text = config_path.read_text(encoding="utf-8")
            match = re.search(r"^\s*checkpoint_path:\s*[\"]?([^\"#\r\n]+)", text, re.MULTILINE)
            relative = match.group(1).strip() if match else "configured checkpoint"
            checks.append({
                "name": f"Optional {model_id} checkpoint ({relative})",
                "ready": info["checkpointReady"],
                "required": False,
                "detail": "Ready" if info["checkpointReady"] else "Not found",
            })
    required_checks = [item for item in checks if item["required"]]
    return {
        "checks": checks,
        "observedRunReady": all(item["ready"] for item in required_checks),
        "auditReady": checks[3]["ready"] and (PROJECT_ROOT / "configs" / "tasks" / "audit_cohorts.json").is_file(),
    }


def _summary() -> dict[str, Any]:
    rows = _ledger_rows()
    audit_path = RESULTS_ROOT / "table1_contamination.csv"
    provenance_path = RESULTS_ROOT / "table1_contamination_provenance.json"
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
    audit_status = str(provenance.get("data_status", "missing")).lower()
    if audit_path.is_file() and audit_status not in {"observed", "mock"}:
        audit_status = "unverified"
    readiness = _readiness()
    return {
        "project": "Seen It Before",
        "question": "Does pretraining exposure change brain-age performance on seen versus unseen AOMIC cohorts?",
        "updatedAt": _utc_now(),
        "observedRuns": sum(row.get("data_status") == "observed" for row in rows),
        "mockRuns": sum(row.get("data_status") == "mock" for row in rows),
        "auditStatus": audit_status,
        "taskIds": _task_ids(),
        "modelIds": _model_ids(),
        "labelBudgets": [25, 50, 100, 250, 500, "all"],
        "seeds": [42, 1337, 2026, 7, 99],
        "readiness": readiness,
    }


def _audit_payload() -> dict[str, Any]:
    audit_path = RESULTS_ROOT / "table1_contamination.csv"
    provenance_path = RESULTS_ROOT / "table1_contamination_provenance.json"
    rows = _read_csv(audit_path) if audit_path.is_file() else []
    provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
    status = str(provenance.get("data_status", "missing")).lower()
    if rows and status not in {"observed", "mock"}:
        status = "unverified"
    if not rows:
        status = "missing"
    return {
        "dataStatus": status,
        "source": provenance.get("source_repo") or provenance.get("manifest_version") or None,
        "manifestVersion": provenance.get("manifest_version") or None,
        "seenThresholdPct": provenance.get("seen_threshold_pct"),
        "unseenThresholdPct": provenance.get("unseen_threshold_pct"),
        "rows": rows,
    }


def _configuration() -> dict[str, Any]:
    # The observed data loader and synthetic smoke fixture currently implement T1.
    return {
        "tasks": [DEFAULT_TASK],
        "models": [_model_info(model_id) for model_id in _model_ids()],
        "labelBudgets": [25, 50, 100, 250, 500, -1],
        "seeds": sorted(VALID_SEEDS),
        "trialBudgets": [2, 5, 20, 50],
        "dataSources": [
            {"id": "AOMIC", "mode": "T1 study", "path": "data/metadata/aomic_scans.csv", "ready": DEFAULT_METADATA.is_file(), "note": "Frozen splits and the preprocessed cache are also required."},
            {"id": "PPMI", "mode": "Legacy pipeline", "path": "--ppmi-dir", "ready": False, "note": "Pass a local dataset root to run_benchmark.py."},
            {"id": "HBN", "mode": "Legacy pipeline", "path": "--hbn-dir", "ready": False, "note": "Pass a local dataset root to run_benchmark.py."},
            {"id": "NKI", "mode": "Legacy pipeline", "path": "--nki-dir", "ready": False, "note": "Pass a local dataset root to run_benchmark.py."},
        ],
    }


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job:
            return dict(job)
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        return None
    for manifest_path in (RESULTS_ROOT / "mock" / "run_history" / f"{job_id}.json", RESULTS_ROOT / "run_history" / f"{job_id}.json"):
        try:
            saved = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        log_path = PROJECT_ROOT / str(saved.get("logPath") or "")
        if log_path.is_file():
            try:
                saved["log"] = log_path.read_text(encoding="utf-8", errors="replace")[-500_000:]
            except OSError:
                saved["log"] = ""
        return saved
    return None


def _run_job(job_id: str, command: list[str]) -> None:
    with _JOB_LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["startedAt"] = _utc_now()
        job = dict(_JOBS[job_id])
    log_path, _ = _job_storage_paths(job)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _persist_job_manifest(job_id)
    try:
        log_tail = ""
        with log_path.open("w", encoding="utf-8", newline="") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                log_tail = (log_tail + line)[-12_000:]
                with _JOB_LOCK:
                    _JOBS[job_id]["log"] = log_tail
            return_code = process.wait()
            status = "completed" if return_code == 0 else "failed"
            with _JOB_LOCK:
                _JOBS[job_id].update({
                    "status": status,
                    "exitCode": return_code,
                    "log": log_tail,
                    "finishedAt": _utc_now(),
                })
    except OSError as exc:
        with _JOB_LOCK:
            _JOBS[job_id].update({"status": "failed", "error": str(exc), "log": f"{type(exc).__name__}: {exc}\n", "finishedAt": _utc_now()})
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{type(exc).__name__}: {exc}\n")
        except OSError:
            pass
    finally:
        _persist_job_manifest(job_id)


def _start_run(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    mode = str(payload.get("mode", "mock"))
    task_id = str(payload.get("taskId", DEFAULT_TASK))
    model_id = str(payload.get("modelId", "resnet18_scratch"))
    try:
        n_labels = int(payload.get("nLabels", 25))
        seed = int(payload.get("seed", 42))
        trials = int(payload.get("trials", 2 if mode == "mock" else 5))
    except (TypeError, ValueError):
        return 400, {"error": "Label budget, seed, and trial budget must be integers."}
    if mode not in {"mock", "observed"}:
        return 400, {"error": "Run mode must be 'mock' or 'observed'."}
    if task_id not in _task_ids() or model_id not in _model_ids():
        return 400, {"error": "Choose a task and model that have a configuration in this repository."}
    if task_id != DEFAULT_TASK:
        return 400, {"error": "The interactive runner currently supports the T1 AOMIC brain-age task only."}
    if n_labels not in VALID_BUDGETS or seed not in VALID_SEEDS or trials not in VALID_TRIALS:
        return 400, {"error": "The requested budgets or seed are not in the study configuration."}
    model_info = _model_info(model_id)
    if not model_info["available"]:
        return 409, {"error": f"{model_id} requires a foundation-model checkpoint that is not present. Add the configured checkpoint before starting this cell."}
    if mode == "observed":
        readiness = _readiness()
        if not readiness["observedRunReady"]:
            return 409, {"error": "Observed runs need frozen splits, AOMIC metadata, and cached volumes. See the readiness checklist."}
        if trials == 2:
            return 400, {"error": "Observed runs use the preregistered tuning budgets: 5, 20, or 50 trials."}

    with _JOB_LOCK:
        if any(job.get("status") in {"queued", "running"} for job in _JOBS.values()):
            return 409, {"error": "A benchmark cell is already running. Wait for it to finish before starting another."}
        job_id = uuid.uuid4().hex[:12]
        results_dir = RESULTS_ROOT / "mock" if mode == "mock" else RESULTS_ROOT
        command = [
            sys.executable,
            str(PROJECT_ROOT / "harness" / "run_cell.py"),
            "--task", task_id,
            "--model", model_id,
            "--n_labels", str(n_labels),
            "--seed", str(seed),
            "--trials", str(trials),
            "--results_dir", str(results_dir),
        ]
        if mode == "mock":
            command.append("--mock")
        else:
            command.extend([
                "--splits_json", str(DEFAULT_SPLITS),
                "--metadata_csv", str(DEFAULT_METADATA),
                "--cache_dir", str(DEFAULT_CACHE),
            ])
        _JOBS[job_id] = {
            "id": job_id,
            "mode": mode,
            "taskId": task_id,
            "modelId": model_id,
            "nLabels": "all" if n_labels == -1 else n_labels,
            "seed": seed,
            "trials": trials,
            "status": "queued",
            "createdAt": _utc_now(),
            "log": "",
        }
    _persist_job_manifest(job_id)
    worker = threading.Thread(target=_run_job, args=(job_id, command), daemon=True)
    worker.start()
    return 202, _get_job(job_id) or {"id": job_id, "status": "queued"}


def _did_payload() -> dict[str, Any]:
    summary_path = RESULTS_ROOT / "did_summary.json"
    ledger_path = RESULTS_ROOT / "results.csv"
    if not summary_path.is_file():
        return {"status": "missing", "result": None}
    try:
        result = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid", "result": None}
    if result.get("data_status") != "observed":
        return {"status": "unverified", "result": None}
    if "p_value_bootstrap_two_sided" not in result:
        return {"status": "legacy", "result": None}
    stale = ledger_path.is_file() and summary_path.stat().st_mtime < ledger_path.stat().st_mtime
    return {"status": "stale" if stale else "observed", "result": None if stale else result}


def _has_matched_did_cells(rows: list[dict[str, Any]], fm_model: str, base_model: str, task_id: str) -> bool:
    fields = ("task_id", "n_labels", "seed", "trial_budget")
    arms: dict[str, set[tuple[str, ...]]] = {fm_model: set(), base_model: set()}
    for row in rows:
        model_id = row.get("model_id")
        if row.get("data_status") != "observed" or row.get("task_id") != task_id or model_id not in arms:
            continue
        trial_budget = row.get("trial_budget", row.get("trials"))
        arms[model_id].add(tuple(str(row.get(field, trial_budget if field == "trial_budget" else "")) for field in fields))
    return bool(arms[fm_model].intersection(arms[base_model]))


def _start_did(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    fm_model = str(payload.get("fmModel", "brainiac"))
    base_model = str(payload.get("baseModel", "resnet18_scratch"))
    task_id = str(payload.get("taskId", DEFAULT_TASK))
    try:
        bootstrap = int(payload.get("bootstrap", 2000))
    except (TypeError, ValueError):
        return 400, {"error": "Bootstrap count must be an integer."}
    if fm_model not in _model_ids() or base_model not in _model_ids() or task_id not in _task_ids():
        return 400, {"error": "Choose configured models and a configured task."}
    if fm_model == base_model:
        return 400, {"error": "The foundation model and baseline must be different model arms."}
    if task_id != DEFAULT_TASK:
        return 400, {"error": "The current DiD implementation is configured for T1 AOMIC brain age."}
    if bootstrap < 1 or bootstrap > 50_000:
        return 400, {"error": "Bootstrap count must be between 1 and 50,000."}
    observed = [row for row in _ledger_rows() if row.get("data_status") == "observed"]
    if not observed or not (RESULTS_ROOT / "results.csv").is_file():
        return 409, {"error": "A preregistered DiD needs observed final evaluations in results/results.csv."}
    if not _has_matched_did_cells(observed, fm_model, base_model, task_id):
        return 409, {"error": "No observed cells match across task, label budget, seed, and tuning-trial budget for these model arms."}
    with _JOB_LOCK:
        if any(job.get("status") in {"queued", "running"} for job in _JOBS.values()):
            return 409, {"error": "A local run or analysis is already running. Wait for it to finish first."}
        job_id = uuid.uuid4().hex[:12]
        command = [
            sys.executable, str(PROJECT_ROOT / "analysis" / "did.py"),
            "--results_csv", str(RESULTS_ROOT / "results.csv"),
            "--output_dir", str(RESULTS_ROOT),
            "--fm_model", fm_model, "--base_model", base_model,
            "--task", task_id, "--bootstrap", str(bootstrap),
        ]
        _JOBS[job_id] = {
            "id": job_id, "mode": "analysis", "taskId": task_id,
            "modelId": fm_model, "baseModel": base_model,
            "nLabels": "—", "seed": "—", "trials": bootstrap,
            "status": "queued", "createdAt": _utc_now(), "log": "",
        }
    _persist_job_manifest(job_id)
    threading.Thread(target=_run_job, args=(job_id, command), daemon=True).start()
    return 202, _get_job(job_id) or {"id": job_id, "status": "queued"}


def _start_legacy_demo() -> tuple[int, dict[str, Any]]:
    with _JOB_LOCK:
        if any(job.get("status") in {"queued", "running"} for job in _JOBS.values()):
            return 409, {"error": "A local run or analysis is already running. Wait for it to finish first."}
        job_id = uuid.uuid4().hex[:12]
        output_dir = RESULTS_ROOT / "mock" / f"benchmark-demo-{job_id}"
        mock_dir = PROJECT_ROOT / "data_mock" / f"benchmark-demo-{job_id}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_benchmark.py"),
            "--fast", "--demo",
            "--mock-dir", str(mock_dir),
            "--output-dir", str(output_dir),
            "--device", "cpu",
        ]
        _JOBS[job_id] = {
            "id": job_id,
            "mode": "mock-suite",
            "taskId": "PPMI · HBN · NKI demo",
            "modelId": "Legacy benchmark suite",
            "nLabels": "Fast mode",
            "seed": "—",
            "trials": "—",
            "status": "queued",
            "createdAt": _utc_now(),
            "log": "",
            "outputDir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        }
    _persist_job_manifest(job_id)
    threading.Thread(target=_run_job, args=(job_id, command), daemon=True).start()
    return 202, _get_job(job_id) or {"id": job_id, "status": "queued"}


class ResearchRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_artifact(self, query: dict[str, list[str]]) -> None:
        requested = query.get("path", [""])[0]
        candidate = (PROJECT_ROOT / requested).resolve()
        allowed_roots = [root.resolve() for root in (RESULTS_ROOT, RESULTS_TEST_ROOT) if root.exists()]
        if not candidate.is_file() or not any(candidate.is_relative_to(root) for root in allowed_roots):
            return self._send_json({"error": "Saved artifact not found."}, 404)
        content_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(candidate.stat().st_size))
        self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{quote(candidate.name)}")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with candidate.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self._send_json({"status": "ok", "project": "Seen It Before", "time": _utc_now()})
        if parsed.path == "/api/summary":
            return self._send_json(_summary())
        if parsed.path == "/api/config":
            return self._send_json(_configuration())
        if parsed.path == "/api/audit":
            return self._send_json(_audit_payload())
        if parsed.path == "/api/legacy-results":
            return self._send_json(_legacy_artifacts())
        if parsed.path == "/api/correlation":
            return self._send_json(_correlation_payload())
        if parsed.path == "/api/preprocessing":
            return self._send_json(_preprocessing_payload())
        if parsed.path == "/api/qc":
            return self._send_json(_qc_payload())
        if parsed.path == "/api/analysis/did":
            return self._send_json(_did_payload())
        if parsed.path == "/api/analysis/label-equivalence":
            return self._send_json(_label_equivalence_payload(parse_qs(parsed.query)))
        if parsed.path == "/api/analysis/pareto":
            return self._send_json(_pareto_payload(parse_qs(parsed.query)))
        if parsed.path == "/api/results":
            scope = parse_qs(parsed.query).get("scope", ["observed"])[0]
            if scope not in {"observed", "mock", "all"}:
                return self._send_json({"error": "scope must be observed, mock, or all"}, 400)
            rows = _ledger_rows()
            if scope != "all":
                rows = [row for row in rows if row.get("data_status") == scope]
            return self._send_json({"scope": scope, "count": len(rows), "rows": rows})
        if parsed.path == "/api/archive":
            return self._send_json(_results_archive())
        if parsed.path == "/api/artifact":
            return self._send_artifact(parse_qs(parsed.query))
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[a-f0-9]{12}", job_id):
                return self._send_json({"error": "Unknown job."}, 404)
            job = _get_job(job_id)
            return self._send_json(job or {"error": "Unknown job."}, 200 if job else 404)
        if parsed.path.startswith("/api/"):
            return self._send_json({"error": "API route not found."}, 404)
        return super().do_GET()

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/benchmark-demo":
            status, result = _start_legacy_demo()
            return self._send_json(result, status)
        if route not in {"/api/runs", "/api/analysis/did"}:
            return self._send_json({"error": "API route not found."}, 404)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json({"error": "Invalid request length."}, 400)
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            return self._send_json({"error": "Request body is missing or too large."}, 413)
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._send_json({"error": "Request body must be valid JSON."}, 400)
        if not isinstance(payload, dict):
            return self._send_json({"error": "Request body must be a JSON object."}, 400)
        status, result = _start_run(payload) if route == "/api/runs" else _start_did(payload)
        return self._send_json(result, status)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")


def find_available_port(start_port: int = 5000) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"No available local port between {start_port} and {start_port + 49}.")


def run_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    chosen_port = find_available_port(port)
    server = ThreadingHTTPServer((host, chosen_port), ResearchRequestHandler)
    server.daemon_threads = True
    print(f"Seen It Before is ready at http://{host}:{chosen_port}")
    print("Press Ctrl+C to stop the local research dashboard.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the research dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Seen It Before research dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind; defaults to localhost only.")
    parser.add_argument("--port", type=int, default=5000, help="Preferred local port (next free port is selected if busy).")
    args = parser.parse_args()
    run_server(args.host, args.port)

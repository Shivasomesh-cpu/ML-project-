"""
harness/ledger.py
Thread-safe, append-only results ledger.
Records every experimental trial and finalized cell run to results/results.parquet.
Falls back to CSV format if parquet serialization is unavailable.
"""

import os
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_LEDGER_LOCK = threading.Lock()


class ResultsLedger:
    def __init__(self, ledger_dir: Path = Path("results")):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.ledger_dir / "results.parquet"
        self.csv_path = self.ledger_dir / "results.csv"

    def append_result(self, record: Dict[str, Any]):
        """
        Appends a completed evaluation cell result into results.parquet and results.csv.
        """
        with _LEDGER_LOCK:
            record["timestamp"] = pd.Timestamp.now().isoformat()
            df_new = pd.DataFrame([record])

            # 1. Update CSV. Reconcile evolving columns before writing so a new field
            # never shifts values beneath an older header.
            if self.csv_path.exists():
                df_existing_csv = pd.read_csv(self.csv_path)
                df_csv = pd.concat([df_existing_csv, df_new], ignore_index=True, sort=False)
                tmp_csv = self.csv_path.with_suffix(".csv.tmp")
                df_csv.to_csv(tmp_csv, index=False)
                tmp_csv.replace(self.csv_path)
            else:
                df_new.to_csv(self.csv_path, index=False)

            # 2. Update Parquet
            try:
                if self.parquet_path.exists():
                    df_existing = pd.read_parquet(self.parquet_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_combined = df_new
                df_combined.to_parquet(self.parquet_path, index=False)
            except Exception as e:
                logger.debug(f"Parquet write skipped or failed ({e}), CSV record preserved.")

            logger.info(
                f"[LEDGER] Appended cell: Task={record.get('task_id')} | "
                f"Model={record.get('model_id')} | N={record.get('n_labels')} | "
                f"Seed={record.get('seed')} | Metric={record.get('primary_metric_name')}={record.get('primary_metric_val', 0.0):.4f}"
            )

    def load_all_results(self) -> pd.DataFrame:
        """Loads all recorded results into a DataFrame."""
        if self.parquet_path.exists():
            try:
                return pd.read_parquet(self.parquet_path)
            except Exception:
                pass
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()

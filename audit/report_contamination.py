"""
audit/report_contamination.py
Generates Table 1 (Contamination Table) in Markdown and CSV formats,
plus summary statistics and publication-ready figures.
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit.overlap import run_full_audit


def generate_table1_markdown(df: pd.DataFrame, manifest_version: str = "FOMO50K (April 2026)", query_date: str = None) -> str:
    if query_date is None:
        query_date = datetime.now().strftime("%Y-%m-%d")

    md = []
    md.append("# Table 1: Benchmark Pretraining Contamination Audit")
    md.append(f"**Manifest Version:** {manifest_version} | **Audit Date:** {query_date}\n")
    md.append("| Evaluation Cohort | Total Subjects ($N$) | Found in Pretraining ($N_{\\text{seen}}$) | Contamination Rate (%) | Pretraining Status |")
    md.append("| :--- | :---: | :---: | :---: | :--- |")

    for _, row in df.iterrows():
        status_badge = f"**{row['status']}**"
        md.append(
            f"| {row['cohort']} | {row['total_subjects']:,} | {row['pretraining_seen_count']:,} | {row['contamination_pct']:.1f}% | {status_badge} |"
        )

    md.append("\n*Note: AOMIC PIOP1 and PIOP2 are collected on the identical Philips Achieva 3T scanner at the same LAB neuroimaging center as AOMIC ID1000, establishing the controlled natural experiment.*")
    return "\n".join(md)


def plot_contamination_bar(df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    
    colors = ["#e74c3c" if "SEEN" in s else ("#2ecc71" if "UNSEEN" in s else "#f39c12") for s in df["status"]]
    
    ax = sns.barplot(
        x="contamination_pct",
        y="cohort",
        data=df,
        hue="cohort",
        legend=False,
        palette=colors
    )
    
    plt.title("Pretraining Data Contamination Rate Across Evaluation Benchmarks", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Percentage of Subjects in Pretraining Corpus (%)", fontsize=11, weight="semibold")
    plt.ylabel("Evaluation Benchmark Cohort", fontsize=11, weight="semibold")
    plt.xlim(0, 105)

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.1f}%",
            (width + 1.5, p.get_y() + p.get_height() / 2),
            ha="left", va="center", fontsize=10, weight="bold"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Contamination Table 1.")
    parser.add_argument("--manifest_path", type=str, default="data/manifests/mapping.tsv")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cohorts_json", type=str, required=True, help="JSON mapping cohort names to actual subject IDs")
    parser.add_argument("--manifest_version", type=str, default="FOMO50K (v1.0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(args.manifest_path)
    df_audit = run_full_audit(manifest_file, Path(args.cohorts_json))

    provenance_path = manifest_file.parent / "manifest_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {"data_status": "unverified", "source_repo": "unknown"}
    provenance["cohorts_json"] = str(Path(args.cohorts_json).resolve())
    (out_dir / "table1_contamination_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Save CSV
    csv_path = out_dir / "table1_contamination.csv"
    df_audit.to_csv(csv_path, index=False)

    # Save Markdown
    md_content = generate_table1_markdown(df_audit, manifest_version=args.manifest_version)
    md_content = f"**Data status:** `{provenance['data_status']}` | **Manifest source:** `{provenance['source_repo']}`\n\n" + md_content
    md_path = out_dir / "table1_contamination.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Plot Figure
    fig_path = out_dir / "figure1_contamination.png"
    plot_contamination_bar(df_audit, fig_path)

    print(f"\n[OK] Table 1 Markdown saved to: {md_path}")
    print(f"[OK] Table 1 CSV saved to: {csv_path}")
    print(f"[OK] Contamination Plot saved to: {fig_path}\n")
    print(md_content)


if __name__ == "__main__":
    main()

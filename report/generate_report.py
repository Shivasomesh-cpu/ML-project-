"""Generate a transparent report that distinguishes observed results from missing work."""

import argparse
import json
from pathlib import Path


def _read(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _analysis_section(title: str, result: dict | None, figure: str, details: str) -> str:
    if not result or result.get("data_status") != "observed":
        return f"## {title}\n\nNot yet available from observed data. {details}\n"
    lines = [f"## {title}\n", "Results below were computed from observed ledger records."]
    if "did_point_estimate" in result:
        boot_p = result.get("p_value_bootstrap_two_sided")
        sign_p = result.get("p_value_sign_flip_secondary", result.get("p_value_sign_flip"))
        lines.extend([
            f"- DiD point estimate: `{result['did_point_estimate']:+.3f}` years",
            f"- 95% bootstrap CI: `[{result['ci_95_lower']:+.3f}, {result['ci_95_upper']:+.3f}]` years",
            f"- Two-sided bootstrap p-value: `{boot_p:.4f}`" if boot_p is not None else "",
            f"- Sign-flip p-value (secondary): `{sign_p:.4e}`" if sign_p is not None else "",
            f"- Paired cells: `{result['n_paired_cells']}`",
            f"- Preregistered H1 supported: `{'Yes' if result.get('h1_supported') else 'No'}`",
        ])
        lines = [line for line in lines if line]
    else:
        for cohort in ("seen_cohort", "unseen_cohort"):
            value = result.get(cohort, {}).get("N_eq")
            lines.append(f"- {cohort.replace('_', ' ').title()} N_eq: `{value if value is not None else 'not estimable'}`")
    lines.append(f"\n![{title}](../results/{figure})")
    return "\n".join(lines) + "\n"


def generate_full_report(results_dir: Path, output_md: Path) -> str:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    table = _read(results_dir / "table1_contamination.md", "Contamination table not yet compiled from a real pretraining manifest.")
    did = _load_json(results_dir / "did_summary.json")
    neq = _load_json(results_dir / "label_equivalence_summary.json")
    pareto = results_dir / "pareto_summary.csv"
    audit_provenance = _load_json(results_dir / "table1_contamination_provenance.json")
    audit_observed = bool(audit_provenance and audit_provenance.get("data_status") == "observed")
    if not audit_observed:
        table = "Contamination table not yet compiled from a verified observed manifest and actual cohort subject IDs."
    audit_warning = "" if audit_observed else "\n**Status:** not yet available; unverified or mock contamination outputs are excluded.\n"
    pareto_provenance = _load_json(results_dir / "pareto_provenance.json")
    pareto_observed = bool(pareto_provenance and pareto_provenance.get("data_status") == "observed")
    report = [
        "# Seen It Before: contamination-audited brain-MRI evaluation\n",
        "## Status\n",
        "This report does not infer experimental findings from mock data, missing results, or illustrative profiles. Run the manifest audit and matched observed evaluation cells before interpreting a conclusion.\n",
        "## Contamination audit\n",
        audit_warning,
        table,
        "\n![Contamination audit](../results/figure1_contamination.png)\n" if audit_observed and (results_dir / "figure1_contamination.png").exists() else "",
        _analysis_section("Amsterdam Difference-in-Differences", did, "figure2_amsterdam_did.png", "Run BrainIAC and the baseline at matching task, N, seed, and trial budget."),
        _analysis_section("Label equivalence", neq, "figure3_label_equivalence.png", "Run at least two shared label budgets for the foundation model and baseline."),
        "## Compute-matched Pareto frontier\n",
        "Observed Pareto summary:\n" + (f"\n```csv\n{pareto.read_text(encoding='utf-8')}\n```\n![Pareto frontier](../results/figure4_pareto_frontier.png)" if pareto_observed and pareto.exists() else "\nNot yet available from observed results. Mock runs and representative profiles are excluded.\n"),
        "\n## Limitations\n",
        "- A subject-level audit is only as strong as the supplied pretraining manifest and source-specific ID normalization.\n- Foundation checkpoints must be verified compatible with the adapter; a missing or incompatible checkpoint now stops the run.\n- Task 1 has an explicit frozen-split scan-to-cache loader; equivalent loaders for Tasks 2 and 3 remain to be implemented.\n",
    ]
    text = "\n".join(report)
    output_md.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a transparent research report.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_file", default="report/report.md")
    args = parser.parse_args()
    generate_full_report(Path(args.results_dir), Path(args.output_file))
    print(f"Report written to {args.output_file}")


if __name__ == "__main__":
    main()

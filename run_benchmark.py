#!/usr/bin/env python
"""
Command-line runner for the structural MRI foundation model benchmark.
Evaluates AnatCL, BrainIAC, SwinBrain, 3D-Neuro-SimCLR, and CNN3D baselines
against FreeSurfer references across PPMI, HBN, and NKI datasets.

Usage examples:
  1. Quick self-contained demo test:
     python run_benchmark.py --demo --fast

  2. Full benchmark with mock data:
     python run_benchmark.py --demo

  3. Custom real data benchmark:
     python run_benchmark.py --ppmi-dir /path/to/ppmi --hbn-dir /path/to/hbn --nki-dir /path/to/nki --weights-dir /path/to/checkpoints
"""

import os
import sys
import argparse
import torch

from src.data import generate_mock_dataset
from src.pipeline import run_benchmark

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark structural MRI foundation models across clinical and demographic tasks."
    )
    parser.add_argument(
        "--demo", "--mock", action="store_true", default=False,
        help="Generate synthetic/mock MRI data and run full pipeline end-to-end."
    )
    parser.add_argument(
        "--mock-n-subjects", type=int, default=16,
        help="Number of subjects per cohort when generating mock data (default: 16)."
    )
    parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Accelerated evaluation mode (2 seeds, 3 training fractions) for rapid verification."
    )
    parser.add_argument(
        "--mock-dir", type=str, default="./data_mock",
        help="Directory to store mock dataset (default: ./data_mock)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save output figures, tables, and CSVs (default: ./results)."
    )
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="Directory containing foundation model checkpoints (.ckpt, .pth, .tar)."
    )
    parser.add_argument(
        "--ppmi-dir", type=str, default=None,
        help="Root directory for PPMI dataset (if using real data)."
    )
    parser.add_argument(
        "--hbn-dir", type=str, default=None,
        help="Root directory for HBN dataset (if using real data)."
    )
    parser.add_argument(
        "--nki-dir", type=str, default=None,
        help="Root directory for NKI dataset (if using real data)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device: 'cpu' (deterministic default) or 'cuda'."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility settings
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    if torch.cuda.is_available() and args.device == "cuda":
        device = "cuda"
        print("Using CUDA device.")
    else:
        device = "cpu"
        print("Using CPU device.")

    if args.demo or (not args.ppmi_dir and not args.hbn_dir and not args.nki_dir):
        print("\n" + "="*70)
        print("GENERATING SYNTHETIC / MOCK NEUROIMAGING DATASETS")
        print("="*70)
        dataset_paths = generate_mock_dataset(args.mock_dir, n_subjects=args.mock_n_subjects)
    else:
        # Construct real dataset paths from provided directory arguments
        dataset_paths = {
            'PPMI': {
                'labels_path': os.path.join(args.ppmi_dir, "processed_cohort_with_mri.csv"),
                'cat12_dir': os.path.join(args.ppmi_dir, "cat12"),
                'turboprep_dir': os.path.join(args.ppmi_dir, "turboprep_output_ppmi")
            },
            'HBN': {
                'bids_dir': os.path.join(args.hbn_dir, "HBN_BIDS"),
                'demo_file': os.path.join(args.hbn_dir, "HBN_BIDS", "final_preprocessed_subjects_with_demographics.tsv"),
                'fs_dir': os.path.join(args.hbn_dir, "HBN_FreeSurfer", "freesurfer"),
                'turboprep_dir': os.path.join(args.hbn_dir, "turboprep_output")
            },
            'NKI': {
                'base_dir': args.nki_dir,
                'bids_dir': os.path.join(args.nki_dir, "NKI_BIDS"),
                'demo_file': os.path.join(args.nki_dir, "final_demographics.txt"),
                'fs_dir': os.path.join(args.nki_dir, "NKI_FreeSurfer", "freesurfer"),
                'turboprep_dir': os.path.join(args.nki_dir, "turboprep_NKI")
            }
        }

    print("\n" + "="*70)
    print("STARTING BENCHMARK PIPELINE")
    print(f"Output directory : {args.output_dir}")
    print(f"Device           : {device}")
    print(f"Fast mode        : {args.fast}")
    print("="*70)

    all_cv_results = run_benchmark(
        dataset_paths=dataset_paths,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        device=device,
        fast_mode=args.fast
    )

    print("\n" + "="*70)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*70)
    print(f"Total CV experimental runs saved: {len(all_cv_results)}")
    print(f"Results located at: {os.path.abspath(args.output_dir)}")
    print("="*70)

if __name__ == "__main__":
    main()

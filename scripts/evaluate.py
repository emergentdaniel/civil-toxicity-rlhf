#!/usr/bin/env python3
"""Evaluate a DPO checkpoint with full analysis pipeline.

Runs: metrics, slice analysis, calibration, cost-sensitive analysis,
failure modes. Saves all results as CSVs and figures.

Usage:
    python scripts/evaluate.py runs/dpo_beta_stability/<run_dir>/
    python scripts/evaluate.py runs/dpo_beta_stability/<run_dir>/ --samples 10000
    python scripts/evaluate.py runs/dpo_beta_stability/<run_dir>/ --sft-results results/sft_results.csv
    python scripts/evaluate.py runs/dpo_beta_stability/<run_dir>/ --skip-inference

Output in <run_dir>/results/:
    dpo_results.csv          Raw scores per example
    metrics.yaml             PR-AUC, ROC-AUC, operating points
    slices.csv               Per-slice precision/recall
    slice_comparison.csv     SFT vs DPO slice comparison (if SFT results available)
    cost_analysis.csv        Optimal thresholds per cost ratio
    calibration.csv          Calibration curve data
    false_positives.csv      Top false positives
    false_negatives.csv      Top false negatives

Output in <run_dir>/figures/:
    pr_roc_curves.png
    calibration.png
    score_distribution.png
    slice_comparison.png     (if SFT results available)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.scoring import (
    evaluate_model, compute_metrics, get_confusion_at_threshold,
    slice_analysis, slice_analysis_comparison,
    cost_analysis, failure_analysis,
    plot_curves_with_operating_points,
)
from src.training import load_dpo_model, clear_memory


def load_config(run_dir: Path):
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        print(f"ERROR: No config.yaml found in {run_dir}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    return Config(cfg)


def save_calibration_data(df: pd.DataFrame, output_path: Path, n_bins: int = 20):
    probs = 1 / (1 + np.exp(-df['score'].values))
    y_true = df['true_label'].values
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
    cal_df = pd.DataFrame({'predicted_probability': prob_pred, 'actual_frequency': prob_true})
    cal_df.to_csv(output_path, index=False)
    return cal_df


def plot_score_distribution(df: pd.DataFrame, output_path: Path, beta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['score'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Toxicity Score (log-odds)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Score Distribution (beta={beta})')
    axes[0].set_yscale('log')

    toxic = df[df['true_label'] == 1]['score']
    nontoxic = df[df['true_label'] == 0]['score']
    axes[1].hist(nontoxic, bins=50, alpha=0.6, label='Not Toxic', color='tab:green')
    axes[1].hist(toxic, bins=50, alpha=0.6, label='Toxic', color='tab:red')
    axes[1].set_xlabel('Toxicity Score (log-odds)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Score Distribution by Label (beta={beta})')
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_calibration(cal_df: pd.DataFrame, output_path: Path, beta):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(cal_df['predicted_probability'], cal_df['actual_frequency'],
            's-', label=f'DPO (beta={beta})', markersize=8)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Frequency')
    ax.set_title(f'Calibration Curve (beta={beta})')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_slice_comparison(slice_df: pd.DataFrame, output_path: Path, beta):
    if slice_df is None or len(slice_df) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(slice_df))
    width = 0.35

    axes[0].bar(x - width/2, slice_df['sft_precision'], width, label='SFT', alpha=0.8)
    axes[0].bar(x + width/2, slice_df['dpo_precision'], width, label=f'DPO (beta={beta})', alpha=0.8)
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision by Slice')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(slice_df['slice'], rotation=30, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x - width/2, slice_df['sft_recall'], width, label='SFT', alpha=0.8)
    axes[1].bar(x + width/2, slice_df['dpo_recall'], width, label=f'DPO (beta={beta})', alpha=0.8)
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall by Slice')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(slice_df['slice'], rotation=30, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Full evaluation of a DPO run")
    parser.add_argument('run_dir', type=Path, help='Path to run directory')
    parser.add_argument('--samples', type=int, default=None, help='Override test samples')
    parser.add_argument('--sft-results', type=Path, default=None,
                        help='Path to SFT results CSV (enables slice comparison)')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip model inference, use existing dpo_results.csv')
    args = parser.parse_args()

    config = load_config(args.run_dir)
    if args.samples:
        config.test_samples = args.samples

    results_dir = Path(config.results_dir)
    figures_dir = Path(config.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    beta = getattr(config, 'dpo_beta', 'N/A')
    print(f"\n{'='*60}")
    print(f"EVALUATING: beta={beta}, seed={getattr(config, 'seed', 'N/A')}")
    print(f"{'='*60}")

    # ── 1. Model inference ──
    dpo_results_path = results_dir / 'dpo_results.csv'

    if args.skip_inference and dpo_results_path.exists():
        print(f"\n[1/8] Loading existing results from {dpo_results_path}")
        df = pd.read_csv(dpo_results_path)
    else:
        from datasets import load_dataset
        test = load_dataset('json',
                            data_files=f'{config.data_dir}/civil_comments_test.jsonl',
                            split='train')
        n_samples = min(config.test_samples, len(test))
        print(f"\n[1/8] Running inference on {n_samples:,} samples...")

        model, tokenizer = load_dpo_model(config)
        df = evaluate_model(model, tokenizer, test, max_samples=config.test_samples)
        df.to_csv(dpo_results_path, index=False)

        del model
        clear_memory()

    # ── 2. Metrics ──
    print(f"\n[2/8] Computing metrics...")
    metrics = compute_metrics(df)

    metrics_out = {
        'beta': float(beta) if beta != 'N/A' else None,
        'pr_auc': round(float(metrics['pr_auc']), 4),
        'roc_auc': round(float(metrics['roc_auc']), 4),
        'kid_safe_threshold': round(float(metrics['kid_safe_threshold']), 4),
        'adult_threshold': round(float(metrics['adult_threshold']), 4),
        'n_samples': len(df),
        'n_toxic': int(df['true_label'].sum()),
        'toxic_rate': round(float(df['true_label'].mean()), 4),
    }

    for mode, key in [('kid_safe', 'kid_safe_threshold'), ('adult', 'adult_threshold')]:
        conf = get_confusion_at_threshold(df, metrics[key])
        metrics_out[f'{mode}_precision'] = round(float(conf['precision']), 4)
        metrics_out[f'{mode}_recall'] = round(float(conf['recall']), 4)

    with open(results_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics_out, f, default_flow_style=False)

    print(f"  PR-AUC:  {metrics_out['pr_auc']}")
    print(f"  ROC-AUC: {metrics_out['roc_auc']}")
    print(f"  Kid-safe: P={metrics_out['kid_safe_precision']:.1%}, R={metrics_out['kid_safe_recall']:.1%}")
    print(f"  Adult:    P={metrics_out['adult_precision']:.1%}, R={metrics_out['adult_recall']:.1%}")

    # ── 3. PR/ROC curves ──
    print(f"\n[3/8] Generating PR/ROC curves...")
    fig = plot_curves_with_operating_points(metrics, title_prefix=f'DPO (beta={beta}) ')
    fig.savefig(figures_dir / 'pr_roc_curves.png', dpi=150)
    plt.close(fig)

    # ── 4. Score distributions ──
    print(f"[4/8] Generating score distributions...")
    plot_score_distribution(df, figures_dir / 'score_distribution.png', beta)

    # ── 5. Calibration ──
    print(f"[5/8] Computing calibration...")
    cal_df = save_calibration_data(df, results_dir / 'calibration.csv')
    plot_calibration(cal_df, figures_dir / 'calibration.png', beta)

    # ── 6. Slice analysis ──
    print(f"\n[6/8] Running slice analysis...")
    slices = slice_analysis(df, metrics['kid_safe_threshold'])
    slices.to_csv(results_dir / 'slices.csv', index=False)
    print(slices.to_string(index=False))

    # Comparison with SFT
    sft_path = args.sft_results
    if sft_path is None:
        for candidate in [Path('results/sft_results.csv'),
                          Path(config.results_dir) / '..' / '..' / '..' / 'results' / 'sft_results.csv']:
            if candidate.exists():
                sft_path = candidate
                break

    if sft_path is not None and sft_path.exists():
        print(f"\n  Comparing slices against SFT ({sft_path})...")
        df_sft = pd.read_csv(sft_path)
        metrics_sft = compute_metrics(df_sft)
        slice_comp = slice_analysis_comparison(
            df_sft, df, metrics_sft['kid_safe_threshold'], metrics['kid_safe_threshold']
        )
        slice_comp.to_csv(results_dir / 'slice_comparison.csv', index=False)
        plot_slice_comparison(slice_comp, figures_dir / 'slice_comparison.png', beta)

        print("\n  SFT vs DPO slice comparison:")
        print(slice_comp[['slice', 'precision_delta', 'recall_delta']].to_string(index=False))
    else:
        print("  (Skipping SFT comparison — pass --sft-results or place in results/sft_results.csv)")

    # ── 7. Cost analysis ──
    print(f"\n[7/8] Running cost analysis...")
    cost_ratios = [(10, 1), (5, 1), (1, 1), (1, 5)]
    cost_rows = []
    for fn_cost, fp_cost in cost_ratios:
        _, opt = cost_analysis(df, fn_cost, fp_cost)
        cost_rows.append({
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'optimal_threshold': round(float(opt['threshold']), 4),
            'total_cost': int(opt['cost']),
            'false_negatives': int(opt['fn']),
            'false_positives': int(opt['fp']),
        })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(results_dir / 'cost_analysis.csv', index=False)
    print(cost_df.to_string(index=False))

    # ── 8. Failure analysis ──
    print(f"\n[8/8] Running failure analysis...")
    fp, fn = failure_analysis(df, metrics['kid_safe_threshold'], n=10)
    fp.to_csv(results_dir / 'false_positives.csv', index=False)
    fn.to_csv(results_dir / 'false_negatives.csv', index=False)

    # ── Done ──
    print(f"\n{'='*60}")
    print(f"COMPLETE: beta={beta}")
    print(f"{'='*60}")
    print(f"\nResults ({results_dir}):")
    for p in sorted(results_dir.iterdir()):
        print(f"  {p.name:30s} {p.stat().st_size / 1024:6.0f} KB")
    print(f"\nFigures ({figures_dir}):")
    for p in sorted(figures_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == '__main__':
    main()

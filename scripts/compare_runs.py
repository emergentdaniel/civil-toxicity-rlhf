#!/usr/bin/env python3
"""Compare DPO runs across beta values with full analysis.

Aggregates metrics, slice analysis, calibration, and cost-sensitive
results across runs. Generates comparison tables and plots.

Usage:
    python scripts/compare_runs.py runs/dpo_beta_stability/
    python scripts/compare_runs.py runs/dpo_beta_stability/ --plot
    python scripts/compare_runs.py runs/dpo_beta_stability/ --output results/beta_comparison.csv

Expects each run to have been evaluated with scripts/evaluate.py first.
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


def load_run(run_dir: Path) -> dict:
    """Load all available analysis artifacts from a single run."""
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_dir = run_dir / 'results'
    run_data = {
        'beta': config.get('dpo_beta'),
        'seed': config.get('seed'),
        'run': run_dir.name,
    }

    # Metrics
    metrics_path = results_dir / 'metrics.yaml'
    if metrics_path.exists():
        with open(metrics_path) as f:
            run_data['metrics'] = yaml.safe_load(f)
    else:
        return None

    # Slices
    slices_path = results_dir / 'slices.csv'
    if slices_path.exists():
        run_data['slices'] = pd.read_csv(slices_path)

    # Slice comparison (SFT vs DPO)
    slice_comp_path = results_dir / 'slice_comparison.csv'
    if slice_comp_path.exists():
        run_data['slice_comparison'] = pd.read_csv(slice_comp_path)

    # Calibration
    cal_path = results_dir / 'calibration.csv'
    if cal_path.exists():
        run_data['calibration'] = pd.read_csv(cal_path)

    # Cost analysis
    cost_path = results_dir / 'cost_analysis.csv'
    if cost_path.exists():
        run_data['cost'] = pd.read_csv(cost_path)

    return run_data


def print_metrics_table(runs: list[dict]):
    """Print main metrics comparison table."""
    rows = []
    for r in runs:
        m = r['metrics']
        rows.append({
            'beta': r['beta'],
            'PR-AUC': m['pr_auc'],
            'ROC-AUC': m['roc_auc'],
            'KS Prec': m.get('kid_safe_precision', ''),
            'KS Recall': m.get('kid_safe_recall', ''),
            'Adult Prec': m.get('adult_precision', ''),
            'Adult Recall': m.get('adult_recall', ''),
        })
    df = pd.DataFrame(rows).sort_values('beta')

    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False, float_format='{:.4f}'.format))

    best = df.loc[df['PR-AUC'].idxmax()]
    print(f"\nBest PR-AUC: beta={best['beta']} → {best['PR-AUC']:.4f}")
    return df


def print_slice_table(runs: list[dict]):
    """Print per-slice precision across beta values."""
    # Use slice_comparison if available, otherwise raw slices
    slice_runs = [r for r in runs if 'slices' in r]
    if not slice_runs:
        return None

    print("\n" + "=" * 80)
    print("SLICE ANALYSIS ACROSS BETAS (at kid-safe threshold)")
    print("=" * 80)

    # Gather all slice names
    all_slices = set()
    for r in slice_runs:
        all_slices.update(r['slices']['slice'].tolist())

    # Build precision table
    rows = []
    for slice_name in sorted(all_slices):
        row = {'slice': slice_name}
        for r in sorted(slice_runs, key=lambda x: x['beta']):
            s = r['slices']
            match = s[s['slice'] == slice_name]
            if len(match) > 0:
                row[f"P@{r['beta']}"] = match.iloc[0]['precision']
                row[f"R@{r['beta']}"] = match.iloc[0]['recall']
            else:
                row[f"P@{r['beta']}"] = np.nan
                row[f"R@{r['beta']}"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Print precision columns only for readability
    p_cols = ['slice'] + [c for c in df.columns if c.startswith('P@')]
    print("\nPrecision by slice:")
    print(df[p_cols].to_string(index=False, float_format='{:.3f}'.format))

    r_cols = ['slice'] + [c for c in df.columns if c.startswith('R@')]
    print("\nRecall by slice:")
    print(df[r_cols].to_string(index=False, float_format='{:.3f}'.format))

    return df


def print_cost_table(runs: list[dict]):
    """Print cost comparison across beta values."""
    cost_runs = [r for r in runs if 'cost' in r]
    if not cost_runs:
        return None

    print("\n" + "=" * 80)
    print("COST ANALYSIS ACROSS BETAS")
    print("=" * 80)

    for fn_cost, fp_cost in [(10, 1), (5, 1), (1, 1), (1, 5)]:
        print(f"\n  FN:FP cost ratio {fn_cost}:{fp_cost}")
        for r in sorted(cost_runs, key=lambda x: x['beta']):
            match = r['cost'][(r['cost']['fn_cost'] == fn_cost) & (r['cost']['fp_cost'] == fp_cost)]
            if len(match) > 0:
                row = match.iloc[0]
                print(f"    beta={r['beta']:5.2f}: threshold={row['optimal_threshold']:7.3f}, "
                      f"cost={row['total_cost']:,}, FN={row['false_negatives']:,}, FP={row['false_positives']:,}")


def print_calibration_summary(runs: list[dict]):
    """Print calibration summary across betas."""
    cal_runs = [r for r in runs if 'calibration' in r]
    if not cal_runs:
        return

    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)

    for r in sorted(cal_runs, key=lambda x: x['beta']):
        cal = r['calibration']
        # Calibration error: mean absolute difference from diagonal
        mae = np.mean(np.abs(cal['actual_frequency'] - cal['predicted_probability']))
        max_err = np.max(np.abs(cal['actual_frequency'] - cal['predicted_probability']))
        print(f"  beta={r['beta']:5.2f}: mean_abs_error={mae:.3f}, max_error={max_err:.3f}")


def plot_comparison(runs: list[dict], output_dir: Path):
    """Generate comparison plots across beta values."""
    betas = sorted([r['beta'] for r in runs])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Row 1: Metrics ──
    pr_aucs = [next(r for r in runs if r['beta'] == b)['metrics']['pr_auc'] for b in betas]
    roc_aucs = [next(r for r in runs if r['beta'] == b)['metrics']['roc_auc'] for b in betas]

    axes[0, 0].plot(betas, pr_aucs, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('DPO Beta')
    axes[0, 0].set_ylabel('PR-AUC')
    axes[0, 0].set_title('PR-AUC vs Beta')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(betas, roc_aucs, 's-', linewidth=2, markersize=8, color='tab:orange')
    axes[0, 1].set_xlabel('DPO Beta')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('ROC-AUC vs Beta')
    axes[0, 1].grid(True, alpha=0.3)

    # Kid-safe precision/recall vs beta
    ks_prec = [next(r for r in runs if r['beta'] == b)['metrics'].get('kid_safe_precision', 0) for b in betas]
    ks_rec = [next(r for r in runs if r['beta'] == b)['metrics'].get('kid_safe_recall', 0) for b in betas]
    axes[0, 2].plot(betas, ks_prec, 'o-', label='Precision', linewidth=2, markersize=8)
    axes[0, 2].plot(betas, ks_rec, 's-', label='Recall', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('DPO Beta')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Kid-Safe Mode vs Beta')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # ── Row 2: Calibration, Cost, Slices ──

    # Calibration overlay
    cal_runs = [r for r in runs if 'calibration' in r]
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    for r in sorted(cal_runs, key=lambda x: x['beta']):
        cal = r['calibration']
        axes[1, 0].plot(cal['predicted_probability'], cal['actual_frequency'],
                        's-', label=f"beta={r['beta']}", markersize=5)
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Actual Frequency')
    axes[1, 0].set_title('Calibration Comparison')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])

    # Cost at 10:1 ratio vs beta
    cost_runs = [r for r in runs if 'cost' in r]
    if cost_runs:
        costs_10_1 = []
        for b in betas:
            r = next(r for r in cost_runs if r['beta'] == b)
            match = r['cost'][(r['cost']['fn_cost'] == 10) & (r['cost']['fp_cost'] == 1)]
            costs_10_1.append(match.iloc[0]['total_cost'] if len(match) > 0 else np.nan)
        axes[1, 1].plot(betas, costs_10_1, 'D-', linewidth=2, markersize=8, color='tab:red')
        axes[1, 1].set_xlabel('DPO Beta')
        axes[1, 1].set_ylabel('Total Cost')
        axes[1, 1].set_title('Cost at FN:FP=10:1 vs Beta')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No cost data', ha='center', va='center')

    # Slice precision heatmap
    slice_runs = [r for r in runs if 'slices' in r]
    if slice_runs:
        all_slices = sorted(set().union(*[set(r['slices']['slice']) for r in slice_runs]))
        heat_data = []
        for b in betas:
            r = next(r for r in slice_runs if r['beta'] == b)
            row = {}
            for s in all_slices:
                match = r['slices'][r['slices']['slice'] == s]
                row[s] = match.iloc[0]['precision'] if len(match) > 0 else np.nan
            heat_data.append(row)
        heat_df = pd.DataFrame(heat_data, index=[f"{b}" for b in betas])
        im = axes[1, 2].imshow(heat_df.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 2].set_xticks(range(len(all_slices)))
        axes[1, 2].set_xticklabels([s.replace('has_', '').replace('is_', '') for s in all_slices],
                                     rotation=30, ha='right')
        axes[1, 2].set_yticks(range(len(betas)))
        axes[1, 2].set_yticklabels([f"β={b}" for b in betas])
        axes[1, 2].set_title('Slice Precision by Beta')
        plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    else:
        axes[1, 2].text(0.5, 0.5, 'No slice data', ha='center', va='center')

    plt.tight_layout()
    fig.savefig(output_dir / 'beta_comparison.png', dpi=150)
    plt.close(fig)
    print(f"\nComparison plot saved to {output_dir / 'beta_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare DPO runs across beta values")
    parser.add_argument('run_dir', type=Path, help='Parent directory containing runs')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--output', type=str, default=None, help='Save metrics comparison CSV')
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"ERROR: Directory not found: {args.run_dir}")
        sys.exit(1)

    # Load all runs
    runs = []
    for run in sorted(args.run_dir.iterdir()):
        if not run.is_dir():
            continue
        data = load_run(run)
        if data is not None:
            runs.append(data)

    if not runs:
        print(f"No evaluated runs found in {args.run_dir}")
        print("Run scripts/evaluate.py on each run first.")
        sys.exit(1)

    print(f"Found {len(runs)} evaluated runs")

    # Print all comparison tables
    metrics_df = print_metrics_table(runs)
    print_calibration_summary(runs)
    slice_df = print_slice_table(runs)
    print_cost_table(runs)

    # Save
    if args.output:
        metrics_df.to_csv(args.output, index=False)
        print(f"\nMetrics saved to {args.output}")

    if args.plot and len(runs) > 1:
        plot_comparison(runs, args.run_dir)


if __name__ == '__main__':
    main()

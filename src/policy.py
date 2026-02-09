"""Decision policy simulation for content moderation.

Maps continuous risk scores to discrete moderation actions (ALLOW / BLOCK / ESCALATE)
under real-world constraints: fixed review budgets and maximum false-negative rates.

Usage from notebook:
    from src.policy import (
        simulate_policy,
        find_budget_constrained_threshold,
        find_risk_bounded_automation,
        run_budget_comparison,
        run_risk_bounded_comparison,
        run_bootstrap_stability,
        print_cost_impact,
    )
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_policy(scores: np.ndarray, labels: np.ndarray,
                    allow_below: float, block_above: float) -> dict:
    """Simulate a three-outcome moderation policy at fixed thresholds.

    Args:
        scores:      Model risk scores (higher = more toxic).
        labels:      Binary ground-truth labels (1 = toxic).
        allow_below: Content scored ≤ this is auto-allowed.
        block_above: Content scored ≥ this is auto-blocked.

    Returns:
        Dict with counts, rates, and precision for each decision bucket.
    """
    n = len(scores)
    allowed = scores <= allow_below
    blocked = scores >= block_above
    escalated = ~allowed & ~blocked

    n_allowed = int(allowed.sum())
    n_blocked = int(blocked.sum())
    n_escalated = int(escalated.sum())

    fn_in_allowed = int((allowed & (labels == 1)).sum())
    tp_in_blocked = int((blocked & (labels == 1)).sum())
    fp_in_blocked = int((blocked & (labels == 0)).sum())

    return {
        'n_allowed': n_allowed,
        'n_blocked': n_blocked,
        'n_escalated': n_escalated,
        'fn_in_allowed': fn_in_allowed,
        'fp_in_blocked': fp_in_blocked,
        'pct_allowed': n_allowed / n * 100,
        'pct_blocked': n_blocked / n * 100,
        'pct_escalated': n_escalated / n * 100,
        'fn_rate': fn_in_allowed / n_allowed if n_allowed > 0 else 0,
        'block_precision': tp_in_blocked / n_blocked if n_blocked > 0 else 0,
        'pct_auto': (n_allowed + n_blocked) / n * 100,
    }


# ---------------------------------------------------------------------------
# Budget-constrained threshold search
# ---------------------------------------------------------------------------

def find_budget_constrained_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    budget_frac: float,
    fn_fp_weight: float = 5.0,
    n_allow: int = 300,
    tolerance: float = 0.02,
) -> dict:
    """Find optimal allow/block thresholds under a fixed review budget.

    Constraints:
        Escalations must equal the budget within tolerance (not just ≤).
        Prevents gaming by underfilling the review queue.

    Objective (combined, not one-sided):
        minimize: FN_in_allowed + fn_fp_weight × FP_in_blocked

        fn_fp_weight controls the penalty for false blocks relative to
        missed toxicity. Default 5.0 means incorrectly blocking clean
        content is penalized, but missing toxic content is still weighted
        more heavily per-item (since clean items outnumber toxic ~6:1,
        the effective penalty is roughly balanced).

    Args:
        scores:        Model risk scores.
        labels:        Binary ground-truth (1 = toxic).
        budget_frac:   Fraction of total volume allocated to human review.
        fn_fp_weight:  Penalty weight for false blocks relative to false allows.
        n_allow:       Number of allow-threshold candidates to search.
        tolerance:     Fractional tolerance for escalation count matching.

    Returns:
        Dict with optimal thresholds, counts, rates, and combined cost.
        Returns {'cost': np.inf} if no feasible solution is found.
    """
    n = len(scores)
    target_escalated = int(n * budget_frac)
    esc_tolerance = max(int(n * tolerance * budget_frac), 50)

    # Wide search range — don't exclude solutions
    allow_candidates = np.linspace(
        np.percentile(scores, 0.5),
        np.percentile(scores, 98),
        n_allow,
    )

    best = {'cost': np.inf}

    for allow_t in allow_candidates:
        allowed_mask = scores <= allow_t
        n_allowed = allowed_mask.sum()
        if n_allowed == 0:
            continue

        remaining_mask = ~allowed_mask
        remaining_scores = scores[remaining_mask]
        n_remaining = int(remaining_mask.sum())
        if n_remaining == 0:
            continue

        # block_t chosen so that escalated ≈ target
        target_blocked = n_remaining - target_escalated
        if target_blocked < 0:
            continue

        remaining_sorted = np.sort(remaining_scores)
        if target_blocked == 0:
            block_t = remaining_sorted[-1] + 1
        elif target_blocked >= n_remaining:
            block_t = remaining_sorted[0] - 1
        else:
            idx = n_remaining - target_blocked
            block_t = remaining_sorted[max(0, min(idx, n_remaining - 1))]

        blocked_mask = remaining_mask & (scores >= block_t)
        escalated_mask = remaining_mask & (scores < block_t)
        n_escalated = int(escalated_mask.sum())

        # Enforce escalation equality within tolerance
        if abs(n_escalated - target_escalated) > esc_tolerance:
            continue

        fn_in_allowed = int((allowed_mask & (labels == 1)).sum())
        fp_in_blocked = int((blocked_mask & (labels == 0)).sum())
        cost = fn_in_allowed + fn_fp_weight * fp_in_blocked

        if cost < best['cost']:
            n_blocked = int(blocked_mask.sum())
            tp_in_blocked = int((blocked_mask & (labels == 1)).sum())
            best = {
                'cost': cost,
                'allow_t': float(allow_t),
                'block_t': float(block_t),
                'n_allowed': int(n_allowed),
                'n_blocked': n_blocked,
                'n_escalated': n_escalated,
                'fn_in_allowed': fn_in_allowed,
                'fp_in_blocked': fp_in_blocked,
                'fn_rate': fn_in_allowed / n_allowed,
                'block_precision': tp_in_blocked / n_blocked if n_blocked > 0 else 0,
                'pct_auto': (int(n_allowed) + n_blocked) / n * 100,
            }

    return best


# ---------------------------------------------------------------------------
# Risk-bounded automation search
# ---------------------------------------------------------------------------

def find_risk_bounded_automation(
    scores: np.ndarray,
    labels: np.ndarray,
    max_fn_rate: float,
    n_thresholds: int = 500,
) -> dict:
    """Find thresholds that maximize automation at a fixed safety constraint.

    Given a maximum acceptable FN rate on auto-allowed content,
    finds the threshold pair that maximizes the fraction of content
    handled without a human (auto-allow + auto-block).

    Args:
        scores:        Model risk scores.
        labels:        Binary ground-truth (1 = toxic).
        max_fn_rate:   Maximum fraction of auto-allowed content that may be toxic.
        n_thresholds:  Number of allow-threshold candidates to search.

    Returns:
        Dict with optimal thresholds, automation rate, and achieved FN rate.
        Returns {'pct_auto': 0} if no feasible solution is found.
    """
    n = len(scores)

    allow_candidates = np.linspace(
        np.percentile(scores, 0.5),
        np.percentile(scores, 98),
        n_thresholds,
    )

    best = {'pct_auto': 0}

    for allow_t in allow_candidates:
        allowed = scores <= allow_t
        n_allowed = int(allowed.sum())
        if n_allowed == 0:
            continue

        fn_in_allowed = int((allowed & (labels == 1)).sum())
        fn_rate = fn_in_allowed / n_allowed

        if fn_rate > max_fn_rate:
            continue

        # Maximize auto-block on remaining content
        remaining = scores[~allowed]
        if len(remaining) == 0:
            block_t = allow_t + 1
            n_blocked = 0
        else:
            block_candidates = np.linspace(
                np.percentile(remaining, 5),
                np.percentile(remaining, 95),
                200,
            )
            best_block_t = remaining.max() + 1
            best_n_blocked = 0
            for bt in block_candidates:
                curr = int((~allowed & (scores >= bt)).sum())
                if curr > best_n_blocked:
                    best_n_blocked = curr
                    best_block_t = bt
            block_t = best_block_t
            n_blocked = best_n_blocked

        blocked = ~allowed & (scores >= block_t)
        n_blocked = int(blocked.sum())
        pct_auto = (n_allowed + n_blocked) / n * 100

        if pct_auto > best['pct_auto']:
            n_escalated = n - n_allowed - n_blocked
            tp_in_blocked = int((blocked & (labels == 1)).sum())
            best = {
                'pct_auto': pct_auto,
                'allow_t': float(allow_t),
                'block_t': float(block_t),
                'fn_rate': fn_rate,
                'fn_in_allowed': fn_in_allowed,
                'n_allowed': n_allowed,
                'n_blocked': n_blocked,
                'n_escalated': n_escalated,
                'block_precision': tp_in_blocked / n_blocked if n_blocked > 0 else 0,
                'pct_escalated': n_escalated / n * 100,
            }

    return best


# ---------------------------------------------------------------------------
# Comparison runners (called from notebook)
# ---------------------------------------------------------------------------

def run_budget_comparison(
    df_sft: pd.DataFrame,
    df_dpo: pd.DataFrame,
    budget_fractions: Optional[list] = None,
    fn_fp_weight: float = 5.0,
) -> pd.DataFrame:
    """Compare SFT vs DPO at fixed review budgets.

    Args:
        df_sft: SFT results with 'score' and 'true_label' columns.
        df_dpo: DPO results with 'score' and 'true_label' columns.
        budget_fractions: List of budget fractions to test. Defaults to [0.05, 0.10, 0.15, 0.25].
        fn_fp_weight: Penalty weight for false blocks.

    Returns:
        DataFrame with one row per budget level.
    """
    if budget_fractions is None:
        budget_fractions = [0.05, 0.10, 0.15, 0.25]

    n_total = len(df_dpo)
    n_toxic = int(df_dpo['true_label'].sum())
    base_rate = n_toxic / n_total

    # Format objective string based on weight direction
    if fn_fp_weight >= 1.0:
        obj_str = f"FN_in_allowed + {fn_fp_weight:.0f} × FP_in_blocked"
    else:
        reciprocal = 1.0 / fn_fp_weight
        obj_str = f"{reciprocal:.0f} × FN_in_allowed + FP_in_blocked"

    print("=" * 110)
    print("BUDGET-CONSTRAINED ANALYSIS: SFT vs DPO at Fixed Review Capacity")
    print("=" * 110)
    print(f"\nDataset: {n_total:,} comments | {n_toxic:,} toxic ({base_rate:.1%})")
    print(f"Objective: minimize ({obj_str}) at fixed escalation budget")
    print(f"Question: At the same review budget, which model keeps users safer?\n")

    rows = []
    for budget in budget_fractions:
        target = int(n_total * budget)
        sft = find_budget_constrained_threshold(
            df_sft['score'].values, df_sft['true_label'].values.astype(int),
            budget, fn_fp_weight=fn_fp_weight,
        )
        dpo = find_budget_constrained_threshold(
            df_dpo['score'].values, df_dpo['true_label'].values.astype(int),
            budget, fn_fp_weight=fn_fp_weight,
        )

        if sft['cost'] == np.inf or dpo['cost'] == np.inf:
            print(f"  Budget {budget:.0%}: no feasible solution found, skipping.")
            continue

        rows.append({
            'Budget': f"{budget:.0%} ({target:,})",
            'SFT Esc': f"{sft['n_escalated']:,}",
            'DPO Esc': f"{dpo['n_escalated']:,}",
            'SFT FN%': f"{sft['fn_rate']:.2%}",
            'DPO FN%': f"{dpo['fn_rate']:.2%}",
            'SFT Missed': sft['fn_in_allowed'],
            'DPO Missed': dpo['fn_in_allowed'],
            'SFT Blk%': f"{sft['n_blocked']/n_total:.1%}",
            'DPO Blk%': f"{dpo['n_blocked']/n_total:.1%}",
            'SFT BlkPrec': f"{sft['block_precision']:.1%}",
            'DPO BlkPrec': f"{dpo['block_precision']:.1%}",
            'SFT FP Blk': sft['fp_in_blocked'],
            'DPO FP Blk': dpo['fp_in_blocked'],
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    print("Esc        = comments sent to human review (should match budget)")
    print("FN%        = toxic fraction of auto-allowed content")
    print("Missed     = absolute count of toxic items auto-allowed")
    print("Blk%       = fraction of all content auto-blocked")
    print("BlkPrec    = precision of auto-block (higher = fewer clean items silenced)")
    print("FP Blk     = clean items incorrectly auto-blocked")

    return table


def run_risk_bounded_comparison(
    df_sft: pd.DataFrame,
    df_dpo: pd.DataFrame,
    fn_constraints: Optional[dict] = None,
) -> pd.DataFrame:
    """Compare SFT vs DPO at fixed safety constraints.

    Args:
        df_sft: SFT results with 'score' and 'true_label' columns.
        df_dpo: DPO results with 'score' and 'true_label' columns.
        fn_constraints: Dict mapping mode names to max FN rates.
            Defaults to Kid Safe (1%), Default (2%), Permissive (5%).

    Returns:
        DataFrame with one row per constraint level.
    """
    if fn_constraints is None:
        fn_constraints = {
            'Kid Safe (FN ≤ 1%)': 0.01,
            'Default (FN ≤ 2%)': 0.02,
            'Permissive (FN ≤ 5%)': 0.05,
        }

    n_total = len(df_dpo)
    n_toxic = int(df_dpo['true_label'].sum())
    base_rate = n_toxic / n_total

    print("=" * 105)
    print("RISK-BOUNDED AUTOMATION: SFT vs DPO at Fixed Safety Constraints")
    print("=" * 105)
    print(f"\nDataset: {n_total:,} comments | {n_toxic:,} toxic ({base_rate:.1%})")
    print(f"Question: At the same safety guarantee, which model automates more?\n")

    rows = []
    for mode_name, max_fn in fn_constraints.items():
        sft = find_risk_bounded_automation(
            df_sft['score'].values, df_sft['true_label'].values.astype(int), max_fn,
        )
        dpo = find_risk_bounded_automation(
            df_dpo['score'].values, df_dpo['true_label'].values.astype(int), max_fn,
        )

        delta = dpo['pct_auto'] - sft['pct_auto']
        delta_str = f"+{delta:.1f}pp" if delta >= 0 else f"{delta:.1f}pp"

        rows.append({
            'Policy': mode_name,
            'SFT Auto-Rate': f"{sft['pct_auto']:.1f}%",
            'DPO Auto-Rate': f"{dpo['pct_auto']:.1f}%",
            'Δ Auto': delta_str,
            'SFT Escalated': f"{sft['pct_escalated']:.1f}%",
            'DPO Escalated': f"{dpo['pct_escalated']:.1f}%",
            'SFT FN (actual)': f"{sft['fn_rate']:.2%}",
            'DPO FN (actual)': f"{dpo['fn_rate']:.2%}",
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print()
    print("Auto-Rate = % of content decided without a human")
    print("Δ Auto = DPO improvement in automation (positive = DPO automates more)")
    print("Escalated = % sent to human review")
    print("FN (actual) = achieved FN rate (must stay ≤ policy constraint)")

    return table


# ---------------------------------------------------------------------------
# Bootstrap stability check
# ---------------------------------------------------------------------------

def run_bootstrap_stability(
    df_sft: pd.DataFrame,
    df_dpo: pd.DataFrame,
    budget_frac: float = 0.10,
    n_bootstrap: int = 300,
    seed: int = 42,
    fn_fp_weight: float = 5.0,
) -> dict:
    """Bootstrap comparison of SFT vs DPO at a fixed review budget.

    Resamples the evaluation set and re-runs the budget-constrained optimizer
    to measure stability of the SFT vs DPO difference.

    Args:
        df_sft: SFT results.
        df_dpo: DPO results.
        budget_frac: Review budget fraction to test.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        fn_fp_weight: Penalty weight for false blocks.

    Returns:
        Dict with delta arrays, summary statistics, and win counts.
    """
    rng = np.random.RandomState(seed)
    n_total = len(df_dpo)

    sft_scores = df_sft['score'].values
    sft_labels = df_sft['true_label'].values.astype(int)
    dpo_scores = df_dpo['score'].values
    dpo_labels = df_dpo['true_label'].values.astype(int)

    delta_missed = []
    delta_fn_rate = []

    print(f"Running {n_bootstrap} bootstrap resamples at {budget_frac:.0%} budget...")

    for _ in range(n_bootstrap):
        idx = rng.choice(n_total, size=n_total, replace=True)

        sft_boot = find_budget_constrained_threshold(
            sft_scores[idx], sft_labels[idx], budget_frac,
            fn_fp_weight=fn_fp_weight, n_allow=100, tolerance=0.02,
        )
        dpo_boot = find_budget_constrained_threshold(
            dpo_scores[idx], dpo_labels[idx], budget_frac,
            fn_fp_weight=fn_fp_weight, n_allow=100, tolerance=0.02,
        )

        if sft_boot['cost'] == np.inf or dpo_boot['cost'] == np.inf:
            continue

        delta_missed.append(dpo_boot['fn_in_allowed'] - sft_boot['fn_in_allowed'])
        delta_fn_rate.append(dpo_boot['fn_rate'] - sft_boot['fn_rate'])

    delta_missed = np.array(delta_missed)
    delta_fn_rate = np.array(delta_fn_rate)
    n_valid = len(delta_missed)

    dpo_wins = int((delta_missed < 0).sum())
    ties = int((delta_missed == 0).sum())
    sft_wins = int((delta_missed > 0).sum())

    print(f"\n{'=' * 70}")
    print(f"BOOTSTRAP STABILITY CHECK (n={n_valid} resamples, budget={budget_frac:.0%})")
    print(f"{'=' * 70}")
    print(f"\nΔ Toxic Missed (DPO − SFT):")
    print(f"  Median: {np.median(delta_missed):+.1f}")
    print(f"  Mean:   {np.mean(delta_missed):+.1f}")
    print(f"  95% CI: [{np.percentile(delta_missed, 2.5):+.1f}, "
          f"{np.percentile(delta_missed, 97.5):+.1f}]")
    print(f"  Std:    {np.std(delta_missed):.1f}")
    print(f"\nΔ FN Rate (DPO − SFT):")
    print(f"  Median: {np.median(delta_fn_rate):+.4f}")
    print(f"  95% CI: [{np.percentile(delta_fn_rate, 2.5):+.4f}, "
          f"{np.percentile(delta_fn_rate, 97.5):+.4f}]")
    print(f"\nDPO wins {dpo_wins}/{n_valid} resamples ({dpo_wins/n_valid:.0%})")
    print(f"Tie      {ties}/{n_valid} resamples ({ties/n_valid:.0%})")
    print(f"SFT wins {sft_wins}/{n_valid} resamples ({sft_wins/n_valid:.0%})")

    median_delta = np.median(delta_missed)
    ci_low = np.percentile(delta_missed, 2.5)
    ci_high = np.percentile(delta_missed, 97.5)

    if abs(median_delta) <= 5 and ci_low <= 0 <= ci_high:
        print(f"\n→ No significant difference under strict review budgets.")
        print(f"  Both models achieve comparable safety at this review capacity.")
    elif median_delta < 0:
        print(f"\n→ DPO misses fewer toxic items at the same review budget.")
    else:
        print(f"\n→ SFT misses fewer toxic items at the same review budget.")

    return {
        'delta_missed': delta_missed,
        'delta_fn_rate': delta_fn_rate,
        'n_valid': n_valid,
        'dpo_wins': dpo_wins,
        'ties': ties,
        'sft_wins': sft_wins,
        'median_delta': float(median_delta),
        'ci_95': (float(ci_low), float(ci_high)),
    }


# ---------------------------------------------------------------------------
# Cost impact (appendix / illustrative)
# ---------------------------------------------------------------------------

def print_cost_impact(
    df_dpo: pd.DataFrame,
    fn_constraints: Optional[dict] = None,
    daily_second_stage: int = 250_000,
    cost_low: float = 0.25,
    cost_high: float = 1.00,
) -> None:
    """Print illustrative cost impact based on risk-bounded automation rates.

    Uses second-stage volume (after upstream heuristic filters), not total
    platform volume, to produce credible estimates.

    Args:
        df_dpo: DPO results with 'score' and 'true_label' columns.
        fn_constraints: Dict mapping mode names to max FN rates.
        daily_second_stage: Daily comments reaching this second-stage model.
        cost_low: Low estimate for human review cost per comment.
        cost_high: High estimate for human review cost per comment.
    """
    if fn_constraints is None:
        fn_constraints = {
            'Kid Safe (FN ≤ 1%)': 0.01,
            'Default (FN ≤ 2%)': 0.02,
            'Permissive (FN ≤ 5%)': 0.05,
        }

    print("=" * 70)
    print("ILLUSTRATIVE COST IMPACT (upper-bound estimates)")
    print("=" * 70)
    print(f"\nSecond-stage volume: {daily_second_stage:,} comments/day")
    print(f"Review cost: ${cost_low:.2f}–${cost_high:.2f} per comment\n")

    scores = df_dpo['score'].values
    labels = df_dpo['true_label'].values.astype(int)

    for mode_name, max_fn in fn_constraints.items():
        result = find_risk_bounded_automation(scores, labels, max_fn)

        auto_rate = result['pct_auto'] / 100
        daily_auto = int(daily_second_stage * auto_rate)
        daily_esc = daily_second_stage - daily_auto

        saved_low = daily_auto * cost_low
        saved_high = daily_auto * cost_high

        print(f"{mode_name}:")
        print(f"  {daily_auto:,} auto-resolved / {daily_esc:,} escalated per day")
        print(f"  Review savings: ${saved_low:,.0f}–${saved_high:,.0f}/day "
              f"(${saved_low * 365:,.0f}–${saved_high * 365:,.0f}/year)")
        print(f"  Safety: FN rate {result['fn_rate']:.2%} on auto-allowed content")
        print()

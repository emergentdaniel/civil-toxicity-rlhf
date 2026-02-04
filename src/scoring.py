# src/scoring.py
"""
Scoring, evaluation, and visualization functions for content moderation RLHF.

Usage in notebook:
    from src.scoring import (
        evaluate_model, compute_metrics, 
        plot_top_token_probs, plot_curves_with_operating_points,
        slice_analysis, run_full_analysis
    )
    
    # Evaluate single model
    df = evaluate_model(model, tokenizer, val_dataset, max_samples=1000)
    metrics = compute_metrics(df)
    
    # Plot token distribution (before/after SFT)
    plot_top_token_probs(base_model, tokenizer, sample_prompt, title="Base Model")
    plot_top_token_probs(sft_model, tokenizer, sample_prompt, title="After SFT")
    
    # Plot curves with operating points
    plot_curves_with_operating_points(metrics, title_prefix="DPO ")
    
    # Compare all models
    plot_all_curves(metrics_base, metrics_sft, metrics_dpo)
    
    # Slice analysis
    slices = slice_analysis(df, threshold=metrics['kid_safe_threshold'])
    
    # Full analysis pipeline
    results = run_full_analysis(
        {'base': base_model, 'sft': sft_model, 'dpo': dpo_model},
        tokenizer, val_dataset, max_samples=10000
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc,
    precision_score,
    recall_score,
    confusion_matrix
)
from tqdm import tqdm

# ============================================================
# SCORING
# ============================================================

@torch.inference_mode()
def label_logprob(model, tokenizer, prompt: str, label: str) -> float:
    """Get log probability of a label given a prompt."""
    enc_prompt = tokenizer(prompt, return_tensors='pt').to(model.device)
    enc_label = tokenizer(label, return_tensors='pt', add_special_tokens=False).to(model.device)
    input_ids = torch.cat([enc_prompt['input_ids'], enc_label['input_ids']], dim=1)
    attn_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:]
    label_len = enc_label['input_ids'].shape[1]
    start = target.shape[1] - label_len
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return token_logprobs[:, start:].sum().item()


def toxicity_score(model, tokenizer, prompt: str) -> float:
    """Returns log P(toxic) - log P(not toxic). Higher = more toxic."""
    lp_toxic = label_logprob(model, tokenizer, prompt, ' toxic')
    lp_nt = label_logprob(model, tokenizer, prompt, ' not toxic')
    return lp_toxic - lp_nt


def get_label_probs(model, tokenizer, prompt):
    """
    Get P(toxic) and P(not) for checking probability concentration.
    
    Usage:
        p_toxic, p_not = get_label_probs(model, tokenizer, prompt)
        print(f"P(toxic) + P(not) = {p_toxic + p_not:.3f}")  # ≈1 after SFT
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
    toxic_id = tokenizer.encode(' toxic', add_special_tokens=False)[0]
    not_id = tokenizer.encode(' not', add_special_tokens=False)[0]
    return probs[toxic_id].item(), probs[not_id].item()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, tokenizer, val_dataset, max_samples=None):
    """
    Score a model on validation set.
    
    Returns:
        DataFrame with columns: text, true_label, score
    
    Usage:
        df = evaluate_model(sft_model, tokenizer, val_dataset, max_samples=1000)
    """
    samples = val_dataset if max_samples is None else val_dataset.select(range(min(max_samples, len(val_dataset))))
    
    results = []
    for example in tqdm(samples, desc="Evaluating"):
        prompt = example['prompt']
        true_label = 1 if example['completion'].strip() == 'toxic' else 0
        score = toxicity_score(model, tokenizer, prompt)
        text = prompt.split('Comment: ')[1].split('\n\nAnswer:')[0]
        results.append({'text': text, 'true_label': true_label, 'score': score})
    
    return pd.DataFrame(results)


def compute_metrics(df, target_recall=0.90, target_precision=0.90):
    """
    Compute PR-AUC, ROC-AUC, curves, and operating points.
    
    Args:
        df: DataFrame with 'true_label' and 'score' columns
        target_recall: recall target for kid-safe threshold (default 0.90)
        target_precision: precision target for adult threshold (default 0.90)
    
    Returns dict with:
        - pr_auc, roc_auc
        - pr_curve: (precision, recall, thresholds)
        - roc_curve: (fpr, tpr, thresholds)
        - kid_safe_threshold (at target_recall)
        - adult_threshold (at target_precision)
        - target_recall, target_precision (for reference)
    
    Usage:
        metrics = compute_metrics(df)  # defaults: 90% recall, 90% precision
        metrics = compute_metrics(df, target_recall=0.95, target_precision=0.80)
    """
    y_true = df['true_label'].values
    scores = df['score'].values
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    
    # Target recall threshold (kid-safe)
    idx_target_recall = np.argmin(np.abs(recall - target_recall))
    kid_safe_threshold = pr_thresholds[min(idx_target_recall, len(pr_thresholds)-1)]
    
    # Target precision threshold (adult)
    idx_target_prec = np.where(precision >= target_precision)[0]
    if len(idx_target_prec) > 0:
        best_idx = idx_target_prec[np.argmax(recall[idx_target_prec])]
        adult_threshold = pr_thresholds[min(best_idx, len(pr_thresholds)-1)]
    else:
        adult_threshold = np.max(scores)
    
    return {
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, tpr),
        'pr_curve': (precision, recall, pr_thresholds),
        'roc_curve': (fpr, tpr, roc_thresholds),
        'kid_safe_threshold': kid_safe_threshold,
        'adult_threshold': adult_threshold,
        'target_recall': target_recall,
        'target_precision': target_precision
    }


def get_confusion_at_threshold(df, threshold):
    """
    Get precision, recall, confusion matrix at a specific threshold.
    
    Usage:
        conf = get_confusion_at_threshold(df, metrics['kid_safe_threshold'])
        print(f"Precision: {conf['precision']:.1%}")
    """
    y_true = df['true_label'].values
    y_pred = (df['score'].values >= threshold).astype(int)
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def get_summary_table(df_base, df_sft, df_dpo):
    """
    Print a summary table with comparisons and confusion matirices.
    
    Args:
        df_base: DataFrame with base model scores
        df_sft: DataFrame with SFT model scores  
        df_dpo: DataFrame with DPO model scores
        
    """
    metrics_base = compute_metrics(df_base)
    metrics_sft = compute_metrics(df_sft)
    metrics_dpo = compute_metrics(df_dpo)
    
    # Summary table
    print("\n" + "-"*40)
    print("MODEL COMPARISON SUMMARY")
    print("-"*40)
    print(f"{'Model':<10} {'PR-AUC':>10} {'ROC-AUC':>10}")
    print(f"{'Base':<10} {metrics_base['pr_auc']:>10.4f} {metrics_base['roc_auc']:>10.4f}")
    print(f"{'SFT':<10} {metrics_sft['pr_auc']:>10.4f} {metrics_sft['roc_auc']:>10.4f}")
    print(f"{'DPO':<10} {metrics_dpo['pr_auc']:>10.4f} {metrics_dpo['roc_auc']:>10.4f}")
    
    print(f"\nSFT improvement over Base: PR-AUC +{metrics_sft['pr_auc']-metrics_base['pr_auc']:.4f}")
    print(f"DPO improvement over SFT:  PR-AUC +{metrics_dpo['pr_auc']-metrics_sft['pr_auc']:.4f}")
    
    # Operating points for DPO
    print("\n" + "-"*40)
    print("OPERATING POINTS (DPO Model)")
    print("-"*40)
    
    kid_safe = get_confusion_at_threshold(df_dpo, metrics_dpo['kid_safe_threshold'])
    adult = get_confusion_at_threshold(df_dpo, metrics_dpo['adult_threshold'])
    
    print(f"\nKid-Safe Mode (target: 90% recall)")
    print(f"  Threshold: {metrics_dpo['kid_safe_threshold']:.4f}")
    print(f"  Precision: {kid_safe['precision']:.4f}")
    print(f"  Recall:    {kid_safe['recall']:.4f}")
    print(f"  Confusion Matrix:\n{kid_safe['confusion_matrix']}")
    
    print(f"\nAdult Mode (target: 90% precision)")
    print(f"  Threshold: {metrics_dpo['adult_threshold']:.4f}")
    print(f"  Precision: {adult['precision']:.4f}")
    print(f"  Recall:    {adult['recall']:.4f}")
    print(f"  Confusion Matrix:\n{adult['confusion_matrix']}")

def compare_operating_points(metrics_base, metrics_sft, metrics_dpo=None, 
                             target_recalls=[0.90, 0.80, 0.70],
                             target_precisions=[0.90, 0.80, 0.70]):
    """
    Compare precision at fixed recall (and vice versa) across models.
    Shows how much less you sacrifice after training.
    """
    
    models = [('Base', metrics_base), ('SFT', metrics_sft)]
    if metrics_dpo is not None:
        models.append(('DPO', metrics_dpo))
    
    # Precision at fixed recall levels
    print("=" * 60)
    print("PRECISION AT FIXED RECALL")
    print("=" * 60)
    print(f"{'Recall Target':<15}", end="")
    for name, _ in models:
        print(f"{name:>12}", end="")
    print()
    print("-" * 60)
    
    for target_recall in target_recalls:
        print(f"{target_recall:<15.0%}", end="")
        for name, m in models:
            precision, recall, _ = m['pr_curve']
            idx = np.argmin(np.abs(recall - target_recall))
            print(f"{precision[idx]:>12.1%}", end="")
        print()
    
    # Recall at fixed precision levels
    print("\n" + "=" * 60)
    print("RECALL AT FIXED PRECISION")
    print("=" * 60)
    print(f"{'Precision Target':<15}", end="")
    for name, _ in models:
        print(f"{name:>12}", end="")
    print()
    print("-" * 60)
    
    for target_precision in target_precisions:
        print(f"{target_precision:<15.0%}", end="")
        for name, m in models:
            precision, recall, _ = m['pr_curve']
            idx = np.where(precision >= target_precision)[0]
            if len(idx) > 0:
                best_recall = recall[idx].max()
                print(f"{best_recall:>12.1%}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()
    
    # Summary of improvement
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    # At 90% recall, how much precision gained?
    prec_b, rec_b, _ = metrics_base['pr_curve']
    prec_s, rec_s, _ = metrics_sft['pr_curve']
    
    idx_b = np.argmin(np.abs(rec_b - 0.90))
    idx_s = np.argmin(np.abs(rec_s - 0.90))
    
    print(f"At 90% recall: SFT gains {prec_s[idx_s] - prec_b[idx_b]:+.1%} precision over Base")
    
    if metrics_dpo:
        prec_d, rec_d, _ = metrics_dpo['pr_curve']
        idx_d = np.argmin(np.abs(rec_d - 0.90))
        print(f"At 90% recall: DPO gains {prec_d[idx_d] - prec_s[idx_s]:+.1%} precision over SFT")
    
    # At 90% precision, how much recall gained?
    idx_b = np.where(prec_b >= 0.90)[0]
    idx_s = np.where(prec_s >= 0.90)[0]
    
    rec_at_90_base = rec_b[idx_b].max() if len(idx_b) > 0 else 0
    rec_at_90_sft = rec_s[idx_s].max() if len(idx_s) > 0 else 0
    
    print(f"At 90% precision: SFT gains {rec_at_90_sft - rec_at_90_base:+.1%} recall over Base")
    
    if metrics_dpo:
        idx_d = np.where(prec_d >= 0.90)[0]
        rec_at_90_dpo = rec_d[idx_d].max() if len(idx_d) > 0 else 0
        print(f"At 90% precision: DPO gains {rec_at_90_dpo - rec_at_90_sft:+.1%} recall over SFT")


def analyze_failure_probs(model, tokenizer, df_failures):
    """Check both probabilities for failure cases."""
    for _, row in df_failures.iterrows():
        prompt = f"Is this comment toxic? Answer only 'toxic' or 'not toxic'.\n\nComment: {row['text']}\n\nAnswer:"
        p_toxic, p_not = get_label_probs(model, tokenizer, prompt)
        
        ratio = p_toxic / p_not if p_not > 1e-10 else float('inf')
        
        print(f"\nScore: {row['score']:.2f}")
        print(f"P(toxic): {p_toxic:.6f} | P(not toxic): {p_not:.6f} | Ratio: {ratio:.2f}")
        print(f"   {row['text'][:100]}...")
# ============================================================
# PLOTTING
# ============================================================

def plot_top_token_probs(model, tokenizer, prompt, top_k=9, title="Next Token Distribution"):
    """
    Plot top-k token probabilities as horizontal bar chart.
    Shows probability concentration before/after SFT.
    
    Usage:
        plot_top_token_probs(base_model, tokenizer, prompt, title="Base Model")
        plt.show()
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        probs = F.softmax(model(**inputs).logits[0, -1, :], dim=-1)
    
    top_probs, top_indices = torch.topk(probs, top_k)
    tokens = [tokenizer.decode(idx).strip() or repr(tokenizer.decode(idx)) for idx in top_indices]
    probs_np = top_probs.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['g' if ('toxic' in tok or tok == 'not') else 'r' for tok in tokens]
    ax.barh(range(len(tokens)), probs_np, color=colors)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title(title)
    
    for i, prob in enumerate(probs_np):
        ax.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontsize=9)
    
    ax.set_xlim(0, max(probs_np) * 1.2)
    ax.legend(handles=[Patch(facecolor='g', label='Target'), Patch(facecolor='r', label='Other')], 
              loc='lower right', fontsize=8)
    plt.tight_layout()
    return fig


def plot_curves_with_operating_points(metrics, title_prefix="DPO ", color ="tab:green"):
    """
    Plot PR and ROC curves with operating points marked.
    Uses target_recall and target_precision from metrics (set in compute_metrics).
    
    Usage:
        metrics = compute_metrics(df, target_recall=0.95, target_precision=0.80)
        fig = plot_curves_with_operating_points(metrics, title_prefix="DPO ")
        plt.savefig('figures/dpo_curves.png')
    """
    precision, recall, pr_thresh = metrics['pr_curve']
    fpr, tpr, roc_thresh = metrics['roc_curve']
    
    target_recall = metrics.get('target_recall', 0.90)
    target_precision = metrics.get('target_precision', 0.90)
    
    # Find operating points on PR curve
    idx_recall = np.argmin(np.abs(recall - target_recall))
    idx_prec = np.where(precision >= target_precision)[0]
    idx_prec = idx_prec[np.argmax(recall[idx_prec])] if len(idx_prec) > 0 else 0
    
    # Find corresponding ROC points using thresholds
    kid_thresh = pr_thresh[min(idx_recall, len(pr_thresh)-1)]
    adult_thresh = pr_thresh[min(idx_prec, len(pr_thresh)-1)]
    roc_kid_idx = np.argmin(np.abs(roc_thresh - kid_thresh))
    roc_adult_idx = np.argmin(np.abs(roc_thresh - adult_thresh))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR Curve
    axes[0].plot(recall, precision, c=color,linewidth=2,label=f"PR-AUC={metrics['pr_auc']:.3f}")
    axes[0].scatter([recall[idx_recall]], [precision[idx_recall]], s=100, c='blue', marker='o', 
                    zorder=5, label=f'Kid Safe Mode: Recall={target_recall:.0%}, Precision={precision[idx_recall]:.0%}')
    axes[0].annotate(f"Score Threshold={kid_thresh:.2f}",xy=(recall[idx_recall], precision[idx_recall]),xytext=(-130, -5),textcoords="offset points",fontsize=9)
    axes[0].scatter([recall[idx_prec]], [precision[idx_prec]], s=100, c='red', marker='s', 
                    zorder=5, label=f'Adult Mode: Precision={target_precision:.0%}, Recall={recall[idx_prec]:.0%}')
    axes[0].annotate(f"Score Threshold={adult_thresh:.2f}",xy=(recall[idx_prec], precision[idx_prec]),xytext=(10, 0),textcoords="offset points",fontsize=9)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'{title_prefix}Precision-Recall Curve')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    axes[1].plot(fpr, tpr, c=color, linewidth=2, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].scatter([fpr[roc_kid_idx]], [tpr[roc_kid_idx]], s=100, c='blue', marker='o', 
                    zorder=5, label=f'Kid Safe Mode: Recall={target_recall:.0%}, False Positive Rate={fpr[roc_kid_idx]:.0%}')
    axes[1].annotate(f"Score Threshold ={kid_thresh:.2f}",xy=(fpr[roc_kid_idx], tpr[roc_kid_idx]),xytext=(13, -5),textcoords="offset points",fontsize=9)
    axes[1].scatter([fpr[roc_adult_idx]], [tpr[roc_adult_idx]], s=100, c='red', marker='s', 
                    zorder=5, label=f'Adult Mode: Precision={target_precision:.0%}, False Positive Rate={fpr[roc_adult_idx]:.0%}')
    axes[1].annotate(f"Score Threshold ={adult_thresh:.2f}",xy=(fpr[roc_adult_idx], tpr[roc_adult_idx]),xytext=(8, 10),textcoords="offset points",fontsize=9)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{title_prefix}ROC Curve')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_all_curves(metrics_base, metrics_sft, metrics_dpo=None):
    """
    Plot PR and ROC curves for all models on same plot.
    
    Usage:
        fig = plot_all_curves(metrics_base, metrics_sft, metrics_dpo)
        plt.savefig('figures/all_curves.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = [('Base', metrics_base, '--'), ('SFT', metrics_sft, '-')]
    if metrics_dpo is not None:
        models.append(('DPO', metrics_dpo, '-'))
    
    for name, m, ls in models:
        prec, rec, _ = m['pr_curve']
        fpr, tpr, _ = m['roc_curve']
        axes[0].plot(rec, prec, linestyle=ls, linewidth=2, label=f"{name} (AUC={m['pr_auc']:.3f})")
        axes[1].plot(fpr, tpr, linestyle=ls, linewidth=2, label=f"{name} (AUC={m['roc_auc']:.3f})")
    
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve: Base → SFT → DPO')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve: Base → SFT → DPO')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_score_distributions(df_base, df_sft, df_dpo):
    """
    Plot score distributions and shift analysis.
    
    Usage:
        fig = plot_score_distributions(df_base, df_sft, df_dpo)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Score distributions - all three models
    axes[0].hist(df_base['score'], bins=50, alpha=0.5, label='Base', density=True)
    axes[0].hist(df_sft['score'], bins=50, alpha=0.5, label='SFT', density=True)
    axes[0].hist(df_dpo['score'], bins=50, alpha=0.5, label='SFT + DPO', density=True)
    axes[0].set_xlabel('Toxicity Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distributions')
    axes[0].legend()
    
    # Score shift histogram (SFT → DPO)
    merged = df_sft[['text', 'score']].merge(df_dpo[['text', 'score']], on='text', suffixes=('_sft', '_dpo'))
    merged['shift'] = merged['score_dpo'] - merged['score_sft']
    axes[1].hist(merged['shift'], bins=50, color='purple', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='No change')
    axes[1].set_xlabel('Score Change (DPO - SFT)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('How DPO Shifted Predictions')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    # Shift by true label
    merged_labels = df_sft[['text', 'true_label', 'score']].merge(
        df_dpo[['text', 'score']], on='text', suffixes=('_sft', '_dpo'))
    merged_labels['shift'] = merged_labels['score_dpo'] - merged_labels['score_sft']
    
    toxic_shift = merged_labels[merged_labels['true_label'] == 1]['shift']
    nontoxic_shift = merged_labels[merged_labels['true_label'] == 0]['shift']
    
    axes[2].boxplot([toxic_shift, nontoxic_shift], labels=['Toxic', 'Not Toxic'])
    axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Score Change (DPO - SFT)')
    axes[2].set_title('Score Shift by True Label')

    merged = df_sft[['text', 'true_label', 'score']].merge(
        df_dpo[['text', 'score']], on='text', suffixes=('_sft', '_dpo')
    )
    merged['score_diff'] = merged['score_dpo'] - merged['score_sft']

    toxic = merged[merged['true_label'] == 1]['score_diff']
    not_toxic = merged[merged['true_label'] == 0]['score_diff']

    print(f"Mean shift for TOXIC comments:     {toxic.mean():+.4f}")
    print(f"Mean shift for NOT TOXIC comments: {not_toxic.mean():+.4f}")
    print(f"\nIdeal: Toxic shifts positive, Not Toxic shifts negative")
    

    plt.tight_layout()
    return fig

def plot_calibration_curve(df, n_bins=10, title="Calibration Curve"):
    """
    Plot calibration curve - does predicted probability match actual frequency?
    
    Usage:
        fig = plot_calibration_curve(df_dpo, title="DPO Calibration")
    """
    from sklearn.calibration import calibration_curve
    
    # Convert log-odds score to probability
    probs = 1 / (1 + np.exp(-df['score'].values))  # sigmoid
    y_true = df['true_label'].values
    
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Model calibration
    ax.plot(prob_pred, prob_true, 's-', label='Model', markersize=8)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Frequency')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_all_calibration_curves(df_base, df_sft, df_dpo=None, n_bins=10):
    """
    Plot calibration curves for all models on same plot.
    
    Usage:
        fig = plot_all_calibration_curves(df_base, df_sft, df_dpo)
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    
    models = [('Base', df_base), ('SFT', df_sft)]
    if df_dpo is not None:
        models.append(('SFT + DPO', df_dpo))
    
    for name, df in models:
        probs = 1 / (1 + np.exp(-df['score'].values))
        y_true = df['true_label'].values
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
        ax.plot(prob_pred, prob_true, 's-', label=name, markersize=6)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Frequency')
    ax.set_title('Calibration Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def find_biggest_changes(df_sft, df_dpo, n=10):
    """Find examples where DPO changed predictions the most."""
    merged = df_sft[['text', 'true_label', 'score']].merge(
        df_dpo[['text', 'score']], on='text', suffixes=('_sft', '_dpo')
    )
    merged['score_diff'] = merged['score_dpo'] - merged['score_sft']
    merged['abs_diff'] = merged['score_diff'].abs()
    
    print("\n" + "="*60)
    print(f"TOP {n} EXAMPLES WHERE DPO CHANGED MOST")
    print("="*60)
    
    top_changes = merged.nlargest(n, 'abs_diff')
    
    for _, row in top_changes.iterrows():
        direction = "↑ more toxic" if row['score_diff'] > 0 else "↓ less toxic"
        correct = "✓" if row['true_label'] == 1 and row['score_diff'] > 0 or \
                        row['true_label'] == 0 and row['score_diff'] < 0 else "✗"
        
        print(f"\n{correct} True: {'toxic' if row['true_label'] else 'not toxic'} | "
              f"SFT: {row['score_sft']:.2f} → DPO: {row['score_dpo']:.2f} ({direction})")
        print(f"   {row['text'][:150]}...")
    
    return top_changes

# ============================================================
# FAILURE MODE ANALYSIS
# ============================================================

def failure_analysis(df, threshold, n=5):
    """
    Find high-confidence errors: false positives and false negatives.
    
    Usage:
        fp, fn = failure_analysis(df_dpo, metrics_dpo['kid_safe_threshold'])
    """
    y_pred = (df['score'].values >= threshold).astype(int)
    df = df.copy()
    df['pred'] = y_pred
    
    # False positives: predicted toxic, actually not toxic (high confidence wrong)
    false_positives = df[(df['true_label'] == 0) & (df['pred'] == 1)].nlargest(n, 'score')
    
    # False negatives: predicted not toxic, actually toxic (confident miss)
    false_negatives = df[(df['true_label'] == 1) & (df['pred'] == 0)].nsmallest(n, 'score')
    
    print("\n" + "="*60)
    print("FAILURE ANALYSIS")
    print("="*60)
    
    print(f"\n--- FALSE POSITIVES (flagged but not toxic) ---")
    for _, row in false_positives.iterrows():
        print(f"\nScore: {row['score']:.2f}")
        print(f"   {row['text'][:150]}...")
    
    print(f"\n--- FALSE NEGATIVES (missed toxic content) ---")
    for _, row in false_negatives.iterrows():
        print(f"\nScore: {row['score']:.2f}")
        print(f"   {row['text'][:150]}...")
    
    return false_positives, false_negatives

# ============================================================
# SLICE ANALYSIS
# ============================================================

def slice_analysis(df, threshold):
    """
    Compute precision/recall per slice at given threshold.
    
    Usage:
        slices = slice_analysis(df, metrics['kid_safe_threshold'])
        print(slices)
    """
    df = df.copy()
    profanity = ['fuck', 'shit', 'damn', 'ass', 'bitch', 'hell']
    df['has_profanity'] = df['text'].str.lower().str.contains('|'.join(profanity), regex=True)
    df['has_quotes'] = df['text'].str.contains(r'["\'].*["\']', regex=True)
    df['has_laughter'] = df['text'].str.lower().str.contains(r'\b(lol|lmao|haha|rofl)\b', regex=True)
    df['is_long'] = df['text'].str.len() > 500
    df['is_short'] = df['text'].str.len() < 50
    
    y_pred = (df['score'].values >= threshold).astype(int)
    
    results = []
    for slice_name in ['has_profanity', 'has_quotes', 'has_laughter', 'is_long', 'is_short']:
        slice_df = df[df[slice_name]]
        if len(slice_df) < 10:
            continue
        y_true = slice_df['true_label'].values
        y_pred_slice = y_pred[df[slice_name].values]
        results.append({
            'slice': slice_name, 
            'n': len(slice_df), 
            'base_rate': y_true.mean(),
            'precision': precision_score(y_true, y_pred_slice, zero_division=0),
            'recall': recall_score(y_true, y_pred_slice, zero_division=0)
        })
    return pd.DataFrame(results)


# ============================================================
#  COST-SENSITIVE ANALYSIS
# ============================================================

def cost_analysis(df, fn_cost=10, fp_cost=1):
    """Find optimal threshold given asymmetric costs."""
    y_true = df['true_label'].values
    scores = df['score'].values
    
    thresholds = np.percentile(scores, np.linspace(1, 99, 100))
    
    costs = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        total_cost = fn * fn_cost + fp * fp_cost
        costs.append({'threshold': t, 'cost': total_cost, 'fn': fn, 'fp': fp})
    
    costs_df = pd.DataFrame(costs)
    optimal_idx = costs_df['cost'].idxmin()
    optimal = costs_df.iloc[optimal_idx]
    
    return costs_df, optimal

def compare_costs(df_sft, df_dpo, cost_ratios=[(10, 1), (5, 1), (1, 1), (1, 5)]):
    """Compare optimal thresholds at different cost ratios."""
    print("\n" + "="*60)
    print("COST-SENSITIVE THRESHOLD ANALYSIS")
    print("="*60)
    
    results = []
    for fn_cost, fp_cost in cost_ratios:
        _, opt_sft = cost_analysis(df_sft, fn_cost, fp_cost)
        _, opt_dpo = cost_analysis(df_dpo, fn_cost, fp_cost)
        
        print(f"\nFN:FP cost ratio {fn_cost}:{fp_cost}")
        print(f"  SFT optimal: threshold={opt_sft['threshold']:.3f}, cost={opt_sft['cost']:.0f}")
        print(f"  DPO optimal: threshold={opt_dpo['threshold']:.3f}, cost={opt_dpo['cost']:.0f}")
        print(f"  Cost reduction: {opt_sft['cost'] - opt_dpo['cost']:.0f} ({(opt_sft['cost'] - opt_dpo['cost'])/opt_sft['cost']*100:.1f}%)")
        
        results.append({
            'fn_cost': fn_cost,
            'fp_cost': fp_cost,
            'sft_threshold': opt_sft['threshold'],
            'sft_cost': opt_sft['cost'],
            'dpo_threshold': opt_dpo['threshold'],
            'dpo_cost': opt_dpo['cost'],
        })
    
    return pd.DataFrame(results)


# ============================================================
# FULL ANALYSIS PIPELINE
# ============================================================

def run_full_analysis(models_dict, tokenizer, val_dataset, max_samples=1000, save_dir='results'):
    """
    Run complete analysis on multiple models.
    
    Args:
        models_dict: {'base': model, 'sft': model, 'dpo': model}
        tokenizer: shared tokenizer
        val_dataset: validation data with 'prompt' and 'completion' columns
        max_samples: number of samples to evaluate
        save_dir: where to save results
    
    Returns:
        dict with 'df' and 'metrics' for each model
    
    Usage:
        results = run_full_analysis(
            {'base': base_model, 'sft': sft_model, 'dpo': dpo_model},
            tokenizer, val_dataset, max_samples=10000
        )
        
        # Access results
        print(results['dpo']['metrics']['pr_auc'])
        results['sft']['df'].to_csv('sft_predictions.csv')
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    for name, model in models_dict.items():
        print(f"\n{'='*50}\nEvaluating {name.upper()}\n{'='*50}")
        df = evaluate_model(model, tokenizer, val_dataset, max_samples)
        metrics = compute_metrics(df)
        results[name] = {'df': df, 'metrics': metrics}
        print(f"{name}: PR-AUC={metrics['pr_auc']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")
        df.to_csv(f'{save_dir}/{name}_predictions.csv', index=False)
    
    # Plot all curves
    model_names = [k for k in ['base', 'sft', 'dpo'] if k in results]
    if len(model_names) >= 2:
        metrics_list = [results[k]['metrics'] for k in model_names]
        fig = plot_all_curves(*metrics_list[:3] if len(metrics_list) >= 3 else (*metrics_list, None))
        fig.savefig(f'{save_dir}/curves.png', dpi=150)
        plt.close()
    
    # Operating points for best model
    best = 'dpo' if 'dpo' in results else 'sft'
    m = results[best]['metrics']
    df = results[best]['df']
    
    print(f"\n{'='*50}\nOPERATING POINTS ({best.upper()})\n{'='*50}")
    for mode, thresh in [('Kid-Safe (90% recall)', m['kid_safe_threshold']), 
                          ('Adult (90% precision)', m['adult_threshold'])]:
        conf = get_confusion_at_threshold(df, thresh)
        print(f"\n{mode}: threshold={thresh:.2f}")
        print(f"  Precision={conf['precision']:.1%}, Recall={conf['recall']:.1%}")
    
    # Slice analysis
    slices = slice_analysis(df, m['kid_safe_threshold'])
    slices.to_csv(f'{save_dir}/slice_analysis.csv', index=False)
    print(f"\n{'='*50}\nSLICE ANALYSIS\n{'='*50}")
    print(slices.to_string(index=False))
    
    # Score shift analysis if DPO exists
    if 'sft' in results and 'dpo' in results:
        fig = plot_score_distributions(results['sft']['df'], results['dpo']['df'])
        fig.savefig(f'{save_dir}/score_analysis.png', dpi=150)
        plt.close()
        
        # Print shift summary
        merged = results['sft']['df'][['text', 'true_label', 'score']].merge(
            results['dpo']['df'][['text', 'score']], on='text', suffixes=('_sft', '_dpo'))
        merged['shift'] = merged['score_dpo'] - merged['score_sft']
        
        toxic_shift = merged[merged['true_label'] == 1]['shift'].mean()
        nontoxic_shift = merged[merged['true_label'] == 0]['shift'].mean()
        
        print(f"\n{'='*50}\nDPO SCORE SHIFT\n{'='*50}")
        print(f"Toxic comments mean shift:     {toxic_shift:.2f}")
        print(f"Non-toxic comments mean shift: {nontoxic_shift:.2f}")
    
    return results

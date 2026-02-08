import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc, 
    confusion_matrix,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset

# ============================================================
# 1. SCORING FUNCTIONS
# ============================================================

@torch.inference_mode()
def label_logprob(model, tokenizer, prompt: str, label: str) -> float:
    enc_prompt = tokenizer(prompt, return_tensors='pt').to(model.device)
    enc_label  = tokenizer(label,  return_tensors='pt', add_special_tokens=False).to(model.device)
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
    """Returns logprob(toxic) - logprob(not toxic). Higher = more toxic."""
    lp_toxic = label_logprob(model, tokenizer, prompt, ' toxic')
    lp_nt    = label_logprob(model, tokenizer, prompt, ' not toxic')
    return lp_toxic - lp_nt

# ============================================================
# 2. EVALUATE A MODEL ON VALIDATION SET
# ============================================================

def evaluate_model(model, tokenizer, val_dataset, max_samples=None):
    results = []
    samples = val_dataset if max_samples is None else val_dataset.select(range(min(max_samples, len(val_dataset))))
    
    print(f"Starting evaluation on {len(samples)} samples...")
    
    for i, example in enumerate(samples):
        if i % 50 == 0:
            print(f"Processing {i}/{len(samples)}...")
        
        prompt = example['prompt']
        true_label = 1 if example['completion'].strip() == 'toxic' else 0
        score = toxicity_score(model, tokenizer, prompt)
        
        text = prompt.split('Comment: ')[1].split('\n\nAnswer:')[0]
        
        results.append({
            'prompt': prompt,
            'text': text,
            'true_label': true_label,
            'score': score
        })
    
    print("Done!")
    return pd.DataFrame(results)

# ============================================================
# 3. CORE METRICS: PR CURVE, ROC CURVE, OPERATING POINTS
# ============================================================

def compute_metrics(df):
    """Compute PR-AUC, ROC-AUC, and find operating points."""
    y_true = df['true_label'].values
    scores = df['score'].values
    
    # PR Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Operating points
    # Kid-safe: 90% recall (catch almost all toxic)
    idx_90_recall = np.argmin(np.abs(recall - 0.90))
    kid_safe_threshold = pr_thresholds[min(idx_90_recall, len(pr_thresholds)-1)]
    
    # Adult mode: 95% precision (minimize false flags)  
    high_prec_idx = np.where(precision >= 0.95)[0]
    if len(high_prec_idx) > 0:
        # Pick the one with highest recall among those with 95%+ precision
        best_idx = high_prec_idx[np.argmax(recall[high_prec_idx])]
        adult_threshold = pr_thresholds[min(best_idx, len(pr_thresholds)-1)]
    else:
        adult_threshold = np.max(scores)  # fallback
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'pr_curve': (precision, recall, pr_thresholds),
        'roc_curve': (fpr, tpr, roc_thresholds),
        'kid_safe_threshold': kid_safe_threshold,
        'adult_threshold': adult_threshold
    }

def get_confusion_at_threshold(df, threshold):
    """Get precision, recall, confusion matrix at a specific threshold."""
    y_true = df['true_label'].values
    y_pred = (df['score'].values >= threshold).astype(int)
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

# ============================================================
# 4. PLOTTING
# ============================================================

def plot_pr_curves(metrics_base, metrics_sft):
    """Plot PR curves for base vs SFT model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    prec_b, rec_b, _ = metrics_base['pr_curve']
    prec_s, rec_s, _ = metrics_sft['pr_curve']
    
    ax.plot(rec_b, prec_b, label=f"Base (AUC={metrics_base['pr_auc']:.3f})", linestyle='--')
    ax.plot(rec_s, prec_s, label=f"SFT (AUC={metrics_sft['pr_auc']:.3f})")
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve: Base vs SFT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pr_curve.png', dpi=150)
    plt.show()

def plot_roc_curves(metrics_base, metrics_sft):
    """Plot ROC curves for base vs SFT model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr_b, tpr_b, _ = metrics_base['roc_curve']
    fpr_s, tpr_s, _ = metrics_sft['roc_curve']
    
    ax.plot(fpr_b, tpr_b, label=f"Base (AUC={metrics_base['roc_auc']:.3f})", linestyle='--')
    ax.plot(fpr_s, tpr_s, label=f"SFT (AUC={metrics_sft['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k:', label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Base vs SFT')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.show()

# ============================================================
# 5. SLICE-BASED ERROR ANALYSIS
# ============================================================

def add_slices(df):
    """Add slice columns for error analysis."""
    df = df.copy()
    
    # Profanity (simple check - expand as needed)
    profanity_words = ['fuck', 'shit', 'damn', 'ass', 'bitch', 'hell']
    df['has_profanity'] = df['text'].str.lower().str.contains('|'.join(profanity_words), regex=True)
    
    # Quoted text (might contain quoted profanity)
    df['has_quotes'] = df['text'].str.contains(r'["\'].*["\']', regex=True)
    
    # Sarcasm/laughter indicators
    df['has_laughter'] = df['text'].str.lower().str.contains(r'\b(lol|lmao|haha|hehe|rofl)\b', regex=True)
    
    # All caps (more than 50% uppercase, at least 10 chars)
    df['is_shouty'] = df['text'].apply(
        lambda x: len(x) >= 10 and sum(c.isupper() for c in x) / max(len(x.replace(' ', '')), 1) > 0.5
    )
    
    # Long comments
    df['is_long'] = df['text'].str.len() > 500
    
    # Short comments
    df['is_short'] = df['text'].str.len() < 50
    
    return df

def slice_analysis(df, threshold):
    """Compute precision/recall per slice at given threshold."""
    df = add_slices(df)
    y_pred = (df['score'].values >= threshold).astype(int)
    df['pred'] = y_pred
    
    slices = ['has_profanity', 'has_quotes', 'has_laughter', 'is_shouty', 'is_long', 'is_short']
    
    results = []
    for slice_name in slices:
        slice_df = df[df[slice_name]]
        if len(slice_df) < 10:
            continue
        
        y_true_slice = slice_df['true_label'].values
        y_pred_slice = slice_df['pred'].values
        
        results.append({
            'slice': slice_name,
            'n': len(slice_df),
            'base_rate': y_true_slice.mean(),
            'precision': precision_score(y_true_slice, y_pred_slice, zero_division=0),
            'recall': recall_score(y_true_slice, y_pred_slice, zero_division=0)
        })
    
    return pd.DataFrame(results)

# ============================================================
# 6. QUALITATIVE COMPARISON TABLE
# ============================================================

def qualitative_comparison(df_base, df_sft, threshold_base, threshold_sft, n_examples=20):
    """Generate side-by-side comparison of base vs SFT predictions."""
    # Merge on text
    merged = df_base[['text', 'true_label', 'score']].merge(
        df_sft[['text', 'score']], 
        on='text', 
        suffixes=('_base', '_sft')
    )
    
    merged['pred_base'] = (merged['score_base'] >= threshold_base).astype(int)
    merged['pred_sft'] = (merged['score_sft'] >= threshold_sft).astype(int)
    
    # Find interesting examples: where models disagree or where one is wrong
    disagreements = merged[merged['pred_base'] != merged['pred_sft']]
    
    # Sample: some disagreements, some agreements
    if len(disagreements) >= n_examples // 2:
        sample_disagree = disagreements.sample(n=min(n_examples // 2, len(disagreements)), random_state=42)
    else:
        sample_disagree = disagreements
    
    agreements = merged[merged['pred_base'] == merged['pred_sft']]
    sample_agree = agreements.sample(n=min(n_examples - len(sample_disagree), len(agreements)), random_state=42)
    
    comparison = pd.concat([sample_disagree, sample_agree])
    
    # Format for display
    comparison['true'] = comparison['true_label'].map({0: 'not toxic', 1: 'toxic'})
    comparison['base_pred'] = comparison['pred_base'].map({0: 'not toxic', 1: 'toxic'})
    comparison['sft_pred'] = comparison['pred_sft'].map({0: 'not toxic', 1: 'toxic'})
    
    return comparison[['text', 'true', 'score_base', 'base_pred', 'score_sft', 'sft_pred']]

# ============================================================
# 7. MAIN ANALYSIS PIPELINE
# ============================================================

def run_full_analysis(base_model, sft_model, tokenizer, val_dataset, max_samples=1000):
    """
    Run complete analysis pipeline.
    
    Args:
        base_model: The base LLaMA model (no fine-tuning)
        sft_model: The SFT fine-tuned model
        tokenizer: Shared tokenizer
        val_dataset: Validation dataset with 'prompt' and 'completion' columns
        max_samples: Number of samples to evaluate (for speed)
    """
    print("=" * 60)
    print("EVALUATING BASE MODEL")
    print("=" * 60)
    df_base = evaluate_model(base_model, tokenizer, val_dataset, max_samples)
    
    print("\n" + "=" * 60)
    print("EVALUATING SFT MODEL")
    print("=" * 60)
    df_sft = evaluate_model(sft_model, tokenizer, val_dataset, max_samples)
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)
    metrics_base = compute_metrics(df_base)
    metrics_sft = compute_metrics(df_sft)
    
    # Print summary
    print(f"\nBase Model:  PR-AUC = {metrics_base['pr_auc']:.4f}, ROC-AUC = {metrics_base['roc_auc']:.4f}")
    print(f"SFT Model:   PR-AUC = {metrics_sft['pr_auc']:.4f}, ROC-AUC = {metrics_sft['roc_auc']:.4f}")
    
    # Operating points
    print("\n" + "-" * 40)
    print("OPERATING POINTS (SFT Model)")
    print("-" * 40)
    
    kid_safe = get_confusion_at_threshold(df_sft, metrics_sft['kid_safe_threshold'])
    adult = get_confusion_at_threshold(df_sft, metrics_sft['adult_threshold'])
    
    print(f"\nKid-Safe Mode (target: 90% recall)")
    print(f"  Threshold: {metrics_sft['kid_safe_threshold']:.4f}")
    print(f"  Precision: {kid_safe['precision']:.4f}")
    print(f"  Recall:    {kid_safe['recall']:.4f}")
    print(f"  Confusion Matrix:\n{kid_safe['confusion_matrix']}")
    
    print(f"\nAdult Mode (target: 95% precision)")
    print(f"  Threshold: {metrics_sft['adult_threshold']:.4f}")
    print(f"  Precision: {adult['precision']:.4f}")
    print(f"  Recall:    {adult['recall']:.4f}")
    print(f"  Confusion Matrix:\n{adult['confusion_matrix']}")
    
    # Plot curves
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_pr_curves(metrics_base, metrics_sft)
    plot_roc_curves(metrics_base, metrics_sft)
    
    # Slice analysis
    print("\n" + "=" * 60)
    print("SLICE-BASED ERROR ANALYSIS (SFT Model, Kid-Safe threshold)")
    print("=" * 60)
    slice_results = slice_analysis(df_sft, metrics_sft['kid_safe_threshold'])
    print(slice_results.to_string(index=False))
    
    # Qualitative comparison
    print("\n" + "=" * 60)
    print("QUALITATIVE COMPARISON (20 examples)")
    print("=" * 60)
    comparison = qualitative_comparison(
        df_base, df_sft, 
        metrics_base['kid_safe_threshold'], 
        metrics_sft['kid_safe_threshold'],
        n_examples=20
    )
    print(comparison.to_string(index=False))
    
    # Save results
    df_sft.to_csv('sft_predictions.csv', index=False)
    slice_results.to_csv('slice_analysis.csv', index=False)
    comparison.to_csv('qualitative_comparison.csv', index=False)
    
    return {
        'df_base': df_base,
        'df_sft': df_sft,
        'metrics_base': metrics_base,
        'metrics_sft': metrics_sft,
        'slice_results': slice_results,
        'comparison': comparison
    }


"""
# Load your models
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# SFT model (load adapter on top of base)
sft_model = PeftModel.from_pretrained(base_model, "path/to/your/sft_checkpoint")

# Load validation data
val_dataset = load_dataset('json', data_files='civil_comments_sft_eval.jsonl', split='train')

# Run analysis
results = run_full_analysis(base_model, sft_model, tokenizer, val_dataset, max_samples=1000)
"""
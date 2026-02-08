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
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

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
# 2. EVALUATE A MODEL
# ============================================================

def evaluate_model(model, tokenizer, val_dataset, max_samples=None):
    """Returns DataFrame with columns: prompt, text, true_label, score"""
    results = []
    samples = val_dataset if max_samples is None else val_dataset.select(range(min(max_samples, len(val_dataset))))
    
    print(f"Starting evaluation on {len(samples)} samples...")
    
    for i, example in enumerate(tqdm(samples, desc="Scoring")):
        if i % 100 == 0:
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
# 3. CORE METRICS
# ============================================================

def compute_metrics(df):
    """Compute PR-AUC, ROC-AUC, and find operating points."""
    y_true = df['true_label'].values
    scores = df['score'].values
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Kid-safe: 90% recall
    idx_90_recall = np.argmin(np.abs(recall - 0.90))
    kid_safe_threshold = pr_thresholds[min(idx_90_recall, len(pr_thresholds)-1)]
    
    # Adult mode: 95% precision
    high_prec_idx = np.where(precision >= 0.95)[0]
    if len(high_prec_idx) > 0:
        best_idx = high_prec_idx[np.argmax(recall[high_prec_idx])]
        adult_threshold = pr_thresholds[min(best_idx, len(pr_thresholds)-1)]
    else:
        adult_threshold = np.max(scores)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'pr_curve': (precision, recall, pr_thresholds),
        'roc_curve': (fpr, tpr, roc_thresholds),
        'kid_safe_threshold': kid_safe_threshold,
        'adult_threshold': adult_threshold
    }

def get_confusion_at_threshold(df, threshold):
    y_true = df['true_label'].values
    y_pred = (df['score'].values >= threshold).astype(int)
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

# ============================================================
# 4. THREE-WAY PLOTTING (Base → SFT → DPO)
# ============================================================

def plot_pr_curves_three(metrics_base, metrics_sft, metrics_dpo):
    """Plot PR curves for base vs SFT vs DPO."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    prec_b, rec_b, _ = metrics_base['pr_curve']
    prec_s, rec_s, _ = metrics_sft['pr_curve']
    prec_d, rec_d, _ = metrics_dpo['pr_curve']
    
    ax.plot(rec_b, prec_b, label=f"Base (AUC={metrics_base['pr_auc']:.3f})", linestyle=':', linewidth=2)
    ax.plot(rec_s, prec_s, label=f"SFT (AUC={metrics_sft['pr_auc']:.3f})", linestyle='--', linewidth=2)
    ax.plot(rec_d, prec_d, label=f"DPO (AUC={metrics_dpo['pr_auc']:.3f})", linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Base → SFT → DPO', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pr_curve_all.png', dpi=150)
    plt.show()

def plot_roc_curves_three(metrics_base, metrics_sft, metrics_dpo):
    """Plot ROC curves for base vs SFT vs DPO."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    fpr_b, tpr_b, _ = metrics_base['roc_curve']
    fpr_s, tpr_s, _ = metrics_sft['roc_curve']
    fpr_d, tpr_d, _ = metrics_dpo['roc_curve']
    
    ax.plot(fpr_b, tpr_b, label=f"Base (AUC={metrics_base['roc_auc']:.3f})", linestyle=':', linewidth=2)
    ax.plot(fpr_s, tpr_s, label=f"SFT (AUC={metrics_sft['roc_auc']:.3f})", linestyle='--', linewidth=2)
    ax.plot(fpr_d, tpr_d, label=f"DPO (AUC={metrics_dpo['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Base → SFT → DPO', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curve_all.png', dpi=150)
    plt.show()

# ============================================================
# 5. DPO-SPECIFIC ANALYSIS
# ============================================================

def compare_score_distributions(df_base, df_sft, df_dpo):
    """Compare how score distributions change across training stages."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Score distributions
    axes[0].hist(df_base['score'], bins=50, alpha=0.6, label='Base', density=True)
    axes[0].hist(df_sft['score'], bins=50, alpha=0.6, label='SFT', density=True)
    axes[0].hist(df_dpo['score'], bins=50, alpha=0.6, label='DPO', density=True)
    axes[0].set_xlabel('Toxicity Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distributions')
    axes[0].legend()
    
    # SFT vs DPO shift
    merged = df_sft[['text', 'true_label', 'score']].merge(
        df_dpo[['text', 'score']], on='text', suffixes=('_sft', '_dpo')
    )
    merged['score_diff'] = merged['score_dpo'] - merged['score_sft']
    
    axes[1].hist(merged['score_diff'], bins=50, color='purple', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='No change')
    axes[1].set_xlabel('Score Change (DPO - SFT)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('How DPO Shifted Predictions')
    axes[1].legend()
    
    # Score diff by true label
    toxic = merged[merged['true_label'] == 1]['score_diff']
    not_toxic = merged[merged['true_label'] == 0]['score_diff']
    
    axes[2].boxplot([toxic, not_toxic], labels=['Toxic', 'Not Toxic'])
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_ylabel('Score Change (DPO - SFT)')
    axes[2].set_title('Score Shift by True Label')
    
    plt.tight_layout()
    plt.savefig('dpo_score_analysis.png', dpi=150)
    plt.show()
    
    # Print insights
    print("\n" + "="*60)
    print("DPO SCORE SHIFT ANALYSIS")
    print("="*60)
    print(f"Mean shift for TOXIC comments:     {toxic.mean():+.4f}")
    print(f"Mean shift for NOT TOXIC comments: {not_toxic.mean():+.4f}")
    print(f"\nIdeal: Toxic shifts positive, Not Toxic shifts negative")
    
    return merged

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
# 6. CALIBRATION ANALYSIS
# ============================================================

def plot_calibration_three(df_base, df_sft, df_dpo, n_bins=10):
    """Compare calibration across all three models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for df, name, style in [(df_base, 'Base', ':'), 
                            (df_sft, 'SFT', '--'), 
                            (df_dpo, 'DPO', '-')]:
        probs = torch.sigmoid(torch.tensor(df['score'].values)).numpy()
        true_freq, pred_prob = calibration_curve(df['true_label'], probs, n_bins=n_bins, strategy='uniform')
        ax.plot(pred_prob, true_freq, style, marker='o', label=name, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title('Calibration Curve: Base → SFT → DPO', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_all.png', dpi=150)
    plt.show()

# ============================================================
# 7. SLICE-BASED ERROR ANALYSIS
# ============================================================

def add_slices(df):
    """Add slice columns for error analysis."""
    df = df.copy()
    
    profanity_words = ['fuck', 'shit', 'damn', 'ass', 'bitch', 'hell', 'crap']
    df['has_profanity'] = df['text'].str.lower().str.contains('|'.join(profanity_words), regex=True)
    df['has_quotes'] = df['text'].str.contains(r'["\'].*["\']', regex=True)
    df['has_laughter'] = df['text'].str.lower().str.contains(r'\b(lol|lmao|haha|hehe|rofl)\b', regex=True)
    df['is_shouty'] = df['text'].apply(
        lambda x: len(x) >= 10 and sum(c.isupper() for c in x) / max(len(x.replace(' ', '')), 1) > 0.5
    )
    df['is_long'] = df['text'].str.len() > 500
    df['is_short'] = df['text'].str.len() < 50
    
    return df

def slice_analysis_comparison(df_sft, df_dpo, threshold_sft, threshold_dpo):
    """Compare slice performance between SFT and DPO."""
    df_sft = add_slices(df_sft)
    df_dpo = add_slices(df_dpo)
    
    df_sft['pred'] = (df_sft['score'].values >= threshold_sft).astype(int)
    df_dpo['pred'] = (df_dpo['score'].values >= threshold_dpo).astype(int)
    
    slices = ['has_profanity', 'has_quotes', 'has_laughter', 'is_shouty', 'is_long', 'is_short']
    
    results = []
    for slice_name in slices:
        sft_slice = df_sft[df_sft[slice_name]]
        dpo_slice = df_dpo[df_dpo[slice_name]]
        
        if len(sft_slice) < 10:
            continue
        
        results.append({
            'slice': slice_name,
            'n': len(sft_slice),
            'base_rate': sft_slice['true_label'].mean(),
            'sft_precision': precision_score(sft_slice['true_label'], sft_slice['pred'], zero_division=0),
            'sft_recall': recall_score(sft_slice['true_label'], sft_slice['pred'], zero_division=0),
            'dpo_precision': precision_score(dpo_slice['true_label'], dpo_slice['pred'], zero_division=0),
            'dpo_recall': recall_score(dpo_slice['true_label'], dpo_slice['pred'], zero_division=0),
        })
    
    results_df = pd.DataFrame(results)
    results_df['precision_delta'] = results_df['dpo_precision'] - results_df['sft_precision']
    results_df['recall_delta'] = results_df['dpo_recall'] - results_df['sft_recall']
    
    return results_df

# ============================================================
# 8. COST-SENSITIVE ANALYSIS
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
# 9. MAIN ANALYSIS PIPELINE
# ============================================================

def run_full_dpo_analysis(df_base, df_sft, df_dpo):
    """
    Run complete three-way analysis from pre-computed dataframes.
    
    Args:
        df_base: DataFrame with base model scores
        df_sft: DataFrame with SFT model scores  
        df_dpo: DataFrame with DPO model scores
    """
    print("="*60)
    print("COMPUTING METRICS")
    print("="*60)
    
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
    
    print(f"\nAdult Mode (target: 95% precision)")
    print(f"  Threshold: {metrics_dpo['adult_threshold']:.4f}")
    print(f"  Precision: {adult['precision']:.4f}")
    print(f"  Recall:    {adult['recall']:.4f}")
    print(f"  Confusion Matrix:\n{adult['confusion_matrix']}")
    
    # Plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    plot_pr_curves_three(metrics_base, metrics_sft, metrics_dpo)
    plot_roc_curves_three(metrics_base, metrics_sft, metrics_dpo)
    plot_calibration_three(df_base, df_sft, df_dpo)
    
    # DPO-specific analysis
    print("\n" + "="*60)
    print("DPO-SPECIFIC ANALYSIS")
    print("="*60)
    
    merged = compare_score_distributions(df_base, df_sft, df_dpo)
    top_changes = find_biggest_changes(df_sft, df_dpo, n=10)
    
    # Slice analysis
    print("\n" + "="*60)
    print("SLICE-BASED ERROR ANALYSIS (SFT vs DPO)")
    print("="*60)
    
    slice_results = slice_analysis_comparison(
        df_sft, df_dpo,
        metrics_sft['kid_safe_threshold'],
        metrics_dpo['kid_safe_threshold']
    )
    print(slice_results.to_string(index=False))
    
    # Cost analysis
    cost_results = compare_costs(df_sft, df_dpo)
    
    # Save results
    df_dpo.to_csv('dpo_predictions.csv', index=False)
    slice_results.to_csv('slice_analysis_dpo.csv', index=False)
    cost_results.to_csv('cost_analysis.csv', index=False)
    top_changes.to_csv('dpo_biggest_changes.csv', index=False)
    
    return {
        'metrics_base': metrics_base,
        'metrics_sft': metrics_sft,
        'metrics_dpo': metrics_dpo,
        'slice_results': slice_results,
        'cost_results': cost_results,
        'top_changes': top_changes
    }


# ============================================================
# 10. USAGE
# ============================================================

"""
# Option A: If you have all three models loaded
# ---------------------------------------------
df_base = evaluate_model(base_model, tokenizer, val_dataset, max_samples=1000)
df_sft = evaluate_model(sft_model, tokenizer, val_dataset, max_samples=1000)
df_dpo = evaluate_model(dpo_model, tokenizer, val_dataset, max_samples=1000)

results = run_full_dpo_analysis(df_base, df_sft, df_dpo)


# Option B: Sequential evaluation (saves VRAM)
# --------------------------------------------
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
val_dataset = load_dataset('json', data_files='civil_comments_sft_eval.jsonl', split='train')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 1. Evaluate Base
print("Evaluating BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
df_base = evaluate_model(base_model, tokenizer, val_dataset, max_samples=1000)
df_base.to_csv('base_results.csv', index=False)
del base_model; gc.collect(); torch.cuda.empty_cache()

# 2. Evaluate SFT
print("Evaluating SFT model...")
sft_base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
sft_model = PeftModel.from_pretrained(sft_base, 'sft_lora_adapter')
df_sft = evaluate_model(sft_model, tokenizer, val_dataset, max_samples=1000)
df_sft.to_csv('sft_results.csv', index=False)
del sft_base, sft_model; gc.collect(); torch.cuda.empty_cache()

# 3. Evaluate DPO
print("Evaluating DPO model...")
dpo_base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
dpo_model = PeftModel.from_pretrained(dpo_base, 'dpo_lora_adapter')  # Your DPO adapter path
df_dpo = evaluate_model(dpo_model, tokenizer, val_dataset, max_samples=1000)
df_dpo.to_csv('dpo_results.csv', index=False)
del dpo_base, dpo_model; gc.collect(); torch.cuda.empty_cache()

# 4. Run full analysis
df_base = pd.read_csv('base_results.csv')
df_sft = pd.read_csv('sft_results.csv')
df_dpo = pd.read_csv('dpo_results.csv')

results = run_full_dpo_analysis(df_base, df_sft, df_dpo)
"""

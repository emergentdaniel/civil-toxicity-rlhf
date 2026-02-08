# Content Moderation with RLHF (SFT + DPO)

Fine-tuning LLaMA 3.2-1B for toxicity detection using Supervised Fine-Tuning and Direct Preference Optimization.

**Constraints:** Single RTX 4070 (8GB) using 4-bit quantization + LoRA

## Results

| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| Base  | 0.223  | 0.648   |
| + SFT | 0.729  | 0.924   |
| + DPO | 0.752  | 0.931   |

**Precision at Fixed Recall**

| Recall | Base | SFT | DPO |
|--------|------|-----|-----|
| 95%    | 17%  | 33% | 35% |
| 90%    | 17%  | 41% | 44% |
| 80%    | 19%  | 52% | 55% |

**Recall at Fixed Precision**

| Precision | Base | SFT | DPO |
|-----------|------|-----|-----|
| 95%       | 0%   | 17% | 21% |
| 90%       | 0%   | 26% | 31% |
| 80%       | 0%   | 46% | 51% |

## Motivation

Content moderation is not binary. Different platforms need different thresholds: a kids' platform wants to catch everything; an adult platform wants to minimize false flags. The core problem is ranking risk under uncertainty, then letting the platform decide where to draw the line.

This project trains a continuous toxicity scorer that supports multiple moderation policies via score thresholding, without retraining.

## System Context

In a production moderation pipeline, it is not feasible or necessary to score every comment with a large language model. Inexpensive heuristics and lightweight classifiers handle the majority of content.

This model is intended as a second-stage moderation component, applied only to ambiguous or high-risk cases that pass initial filters. Its goal is reducing human review load while controlling false-negative risk on borderline content.

The training pipeline has two stages:

1. **SFT** teaches the task format and concentrates probability mass onto the labels "toxic" and "not toxic". After SFT, `P("toxic"|x) + P("not toxic"|x) ≈ 1`, so the log-probability difference approximates a log-odds score.
2. **DPO** refines the ranking by maximizing the log-odds of a preferred response relative to a rejected one. This turns what was approximately a classifier into a ranker, making the log-probability difference a natural scoring function.

The model outputs `log P("toxic"|x) - log P("not toxic"|x)` rather than a hard classification. Positive scores indicate toxic, negative scores indicate non-toxic, and magnitude reflects confidence.

## Dataset and Assumptions

[Civil Comments](https://huggingface.co/datasets/google/civil_comments): 1.8M comments from news sites (2015-2017) with toxicity annotations (2018-2019). Each comment has a continuous toxicity score (0-1) representing the fraction of annotators who flagged it rather than a hard ground-truth label. A threshold of 0.3 binarizes labels for training.

The model learns an approximation of historical annotator judgments, not an objective notion of toxicity. Model behavior is shaped by the quality, source, and era of the training data.

## Evaluation

**PR-AUC over accuracy.** The dataset is 85% non-toxic. A model predicting "not toxic" for everything achieves 85% accuracy. PR-AUC measures ranking quality across all thresholds and is not inflated by class imbalance.

**Log-odds scoring.** DPO optimizes the log-probability difference between responses. Using the same metric for evaluation aligns training and inference and provides a natural confidence measure. Scores should be interpreted as decision margins, not calibrated probabilities—DPO sacrifices calibration for better ranking.

## Operating Modes

The same model supports different policies by moving a single score threshold:

1. **Kid Safe Mode (90% recall):** Catches 90% of toxic comments at 44% precision. Appropriate for platforms where missed toxicity is expensive.
2. **Adult Mode (90% precision):** 90% of flags are correct, catching 31% of toxic content. Appropriate for platforms where false positives damage user experience.

The score threshold gap between modes is ~35 log-probability units, meaning the model requires ~10^15× higher confidence before flagging in adult mode versus kid-safe mode.

**Cost analysis** across FN:FP ratios (10:1, 5:1, 1:1, 1:5) shows DPO reduces expected cost at every operating point, with the largest gains (6.5%) when missed toxicity is expensive.

## Slice-Based Error Analysis

Aggregate metrics hide important behavioral differences. DPO does not uniformly shift model confidence—it changes behavior asymmetrically across linguistic patterns:

**Where DPO stays cautious:** profanity, ALL CAPS (treated as aggression signal), sarcastic laughter ("lol you're an idiot"). No confidence boost even when benign, which is appropriate since these correlate with adversarial or ambiguous content.

**Where DPO gains confidence:** short comments (less text = less ambiguity, largest shift toward non-toxic) and quoted speech (better handling of reported vs. authored toxicity).

Recall stays stable across all slices. DPO refines false positives without sacrificing false-negative coverage.

## Failure Modes

Most false positives at kid-safe threshold contain clear insults that annotators scored below 0.3—the model may be outperforming the labels. False negatives cluster at 0.30–0.40 (the noisy boundary region) and tend to be condescending or contextually mild.

DPO produces overconfident predictions (P ≈ 0 or 1, rarely in between). This is expected and acceptable for ranking/thresholding, but means raw probabilities should not be interpreted as calibrated confidence.



## Project Structure

```
civil-toxicity-rlhf/
├── src/
│   └── scoring.py              # Evaluation and visualization utilities
├── scripts/
│   ├── train_dpo.py            # DPO training with CLI args
│   ├── evaluate.py             # Evaluation script
│   └── compare_runs.py         # Beta sweep comparison
├── config/
│   └── config.yaml             # Training configuration
├── data/                       # Processed datasets (generated)
├── figures/                    # Generated plots
├── rlhf_content_moderation.ipynb
├── Dockerfile
├── requirements.txt
└── README.md

civil-toxicity-artifacts/       # Outside repo, not tracked in git
├── checkpoints/
│   ├── sft/final/              # SFT adapter weights
│   └── dpo/final/              # DPO adapter weights (includes SFT)
└── results/                    # Evaluation CSVs (base, sft, dpo)
```

Checkpoints and evaluation results are stored outside the repo in `../civil-toxicity-artifacts/` to keep the repository lightweight and avoid committing large binary files.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/emergentdaniel/civil-toxicity-rlhf.git
cd civil-toxicity-rlhf
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook rlhf_content_moderation.ipynb
```

The notebook runs top-to-bottom and will download the Civil Comments dataset, train SFT and DPO adapters, evaluate all three model stages, and generate analysis plots.

### 3. DPO hyperparameter sweeps

```bash
python scripts/train_dpo.py --beta 0.1
python scripts/evaluate.py runs/dpo_beta_stability/<run_dir>/
python scripts/compare_runs.py runs/dpo_beta_stability/ --plot
```

### Docker

```bash
docker build -t civil-toxicity-rlhf .
docker run --gpus all -p 8888:8888 civil-toxicity-rlhf
```

## Design Rationale & System Tradeoffs

**Ranking vs. Classification.** The system outputs a continuous log-odds score rather than a binary label. This means a single model supports multiple policies (kid-safe, adult) via threshold adjustment, but it also means raw scores are not calibrated probabilities. SFT produces well-calibrated probabilities; DPO sacrifices that calibration to improve ranking quality. For deployment, scores should be treated as decision margins for thresholding, not as literal risk estimates.

**Log-odds scoring.** DPO optimizes log-probability difference between responses. Using the same metric for evaluation aligns training and inference, and provides a natural confidence measure.

**Memory management.** Training on 8GB VRAM requires 4-bit NF4 quantization (chosen over FP4 because model weights are approximately Gaussian and NF4 has higher resolution near zero), LoRA rank 16 on attention projections, gradient accumulation (effective batch size 16 with per-device batch size 1), and explicit cache clearing between model loads. Evaluation runs in float16 without quantization, using ~5.5GB VRAM.

**Modular evaluation.** `src/scoring.py` separates evaluation logic from the notebook, making it reusable and testable.

**Containerization.** Experiments use a CUDA-enabled PyTorch Docker image for reproducibility across GPU environments. Dependencies are managed via pip to avoid conda-in-Docker complexity.

## Future Work

- Beta sweep analysis across DPO hyperparameters
- Multi-label toxicity scoring (threat, insult, obscenity) via weighted scores
- Comparison against a classification head baseline

## Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA 12.1+
- 8GB VRAM (uses 4-6GB during training)
- About ~2 hours for training + ~3 hours for evaluation on RTX 4070

## License

MIT

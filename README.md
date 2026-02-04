# Content Moderation with RLHF (SFT + DPO)

Fine-tuning LLaMA 3.2-1B for toxicity detection using Supervised Fine-Tuning and Direct Preference Optimization.

**Constraints:** Single RTX 4070 (8GB) using 4-bit quantization + LoRA

## Overview

Content moderation is not binary. Different platforms need different thresholds. A kids' platform wants to catch everything; an adult platform wants to avoid false flags.

This project trains a continuous toxicity scorer that supports multiple moderation policies via thresholding, without retraining:

1. **SFT** teaches the task and concentrates probability mass onto "toxic" / "not toxic" labels
2. **DPO** refines the ranking, optimizing the log-odds difference between preferred and rejected responses

The model outputs a log-odds score rather than a hard classification, enabling flexible deployment across different policy regimes.

Because RLHF changes model behavior asymmetrically, we rely on slice-based evaluation to understand how DPO affects specific linguistic phenomena rather than just aggregate metrics.

## Dataset and Assumptions

[Civil Comments](https://huggingface.co/datasets/google/civil_comments): 1.8M comments from news sites (2015-2017) with toxicity annotations (2018-2019). Each comment has a continuous toxicity score (0-1) representing the fraction of annotators who flagged it, rather than a hard ground-truth label. A threshold of 0.3 binarizes labels for training.

The model learns an approximation of historical annotator judgments, not an objective notion of toxicity. Model behavior is shaped by the quality, source, and era of the training data.


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

## Slice-Based Error Analysis

Aggregate metrics hide important behavioral differences. To understand how DPO affects specific linguistic patterns, we evaluate performance on targeted slices (profanity, ALL CAPS, laughter, quoted speech, length).

DPO does not uniformly shift model confidence. It remains conservative on stylistic signals associated with borderline toxicity (profanity, ALL CAPS, sarcastic laughter), while becoming more confident on short, unambiguous comments. This matches real-world moderation needs, where such cues often correlate with adversarial or ambiguous content.



## Project Structure

```
civil-toxicity-rlhf/
├── src/
│   └── scoring.py          # Evaluation and visualization utilities
├── config/
│   └── config.yaml         # Training configuration
├── figures/                 # Generated plots
├── data/                    # Processed datasets (generated)
├── checkpoints/             # Model adapters (generated)
├── results/                 # Evaluation outputs (generated)
├── rlhf-content-moderation.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/civil-toxicity-rlhf.git
cd civil-toxicity-rlhf
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook rlhf-content-moderation.ipynb
```

The notebook runs top-to-bottom and will:
- Download the Civil Comments dataset
- Train SFT and DPO adapters
- Evaluate all models
- Generate analysis plots

### Docker

```bash
docker build -t civil-toxicity-rlhf .
docker run --gpus all -p 8888:8888 civil-toxicity-rlhf
```

## Key Design Decisions

**PR-AUC over accuracy.** The dataset is 85% non-toxic. A model predicting "not toxic" for everything achieves 85% accuracy.

**Log-odds scoring.** DPO optimizes log-probability difference between responses. Using the same metric for evaluation aligns training and inference, and provides a natural confidence measure.

**Memory management.** Training on 8GB VRAM requires explicit cache clearing between model loads, 4-bit NF4 quantization with double quantization, and gradient accumulation (effective batch size 16 with batch size 1).

**Modular evaluation.** `src/scoring.py` separates evaluation logic from the notebook, making it reusable and testable.

**Containerization.** Experiments use a CUDA-enabled PyTorch Docker image for reproducibility across GPU environments. Dependencies are managed via pip to avoid conda-in-Docker complexity.


## Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA 12.1+
- 8GB VRAM (uses 4-6GB during training)
- About 2 hours for training + 3 hours for evaluation on RTX 4070

## License

MIT

# Content Moderation with RLHF (SFT + DPO)

A second-stage moderation system that uses an RLHF-trained scorer and explicit decision policies to reduce human review load while bounding false-negative risk.

## What This System Does

Content moderation decisions have asymmetric costs. Missed toxicity causes user harm; false flags erode trust and inflate review queues. A single accuracy number cannot capture this tradeoff because different platforms weigh these errors differently.

This system produces a continuous risk score for each comment, then applies an explicit decision policy that maps scores to actions. The score ranks risk; the policy decides what to do about it. Separating these two concerns means a single model supports multiple operating modes: from aggressive filtering to permissive defaults without retraining.

The non-trivial part is not building the scorer. It is designing evaluation around operational constraints (review budgets, safety ceilings) rather than aggregate metrics, and showing where model improvements actually change system behavior versus where they don't.

## System Architecture

This model is not a first-line filter. In a production pipeline, cheap heuristics and lightweight classifiers handle the majority of content. This model runs as a **second-stage scorer** applied only to the ambiguous or high-risk fraction that passes initial filters.

```
All traffic → Keyword/regex filters → Lightweight classifier → This model → Decision policy
                    ↓ (clear spam)         ↓ (obvious cases)       ↓              ↓
                  Auto-block             Auto-allow/block      Risk score    Allow / Block / Escalate
                                                                                       ↓
                                                                                 Human reviewer
```

The model sees pre-filtered traffic, not raw volume. An LLM-based scorer is acceptable at this stage because volume is reduced and the comments that reach it are the ones where context matters—sarcasm, quoted speech, borderline insults which simpler models get wrong.

Three possible outcomes per comment:

- **Allow**: passes automatically, no human sees it.
- **Block**: removed automatically, no human needed.
- **Escalate**: model confidence is insufficient; routed to a human reviewer.

Escalation is a first-class outcome, not a failure mode. The system's job is to minimize how often humans need to intervene while keeping false-negative risk on auto-allowed content within a declared bound.

## Decision Policy and Operating Modes

The decision policy maps continuous scores to discrete actions using two thresholds (allow and block). Everything between the thresholds gets escalated.

The system defines three operating modes by varying the maximum tolerable false-negative rate on auto-allowed content:

| Mode | Max FN Rate | Intent |
|------|-------------|--------|
| Kid Safe | ≤ 1% | Minimize missed toxicity. Accept higher escalation and false blocks. |
| Default | ≤ 2% | Balanced. Standard content platform. |
| Permissive | ≤ 5% | Minimize false flags. Accept more missed toxicity. |

Switching modes means moving thresholds, not retraining. The scorer is the same in all cases; only the policy changes.

Two parameters control the entire system: the labeling threshold used during training (0.3 toxicity score, which defines what "toxic" means to the model) and the score thresholds used at inference (which define what the platform does about it). Both are interpretable and auditable.

## Model Overview

**Base model:** LLaMA 3.2-1B-Instruct, chosen for the constraint of a single RTX 4070 (8GB VRAM).

**Training pipeline:**

1. **SFT** teaches the output format. After SFT, probability mass concentrates on two tokens ("toxic" / "not toxic") such that `P("toxic"|x) + P("not toxic"|x) ≈ 1`. This is not explicitly enforced, it emerges from training on binary labels. The model now behaves as a binary classifier whose log-probability difference approximates log-odds.

2. **DPO** refines the ranking. It optimizes the log-probability difference between preferred and rejected responses, which is the same quantity used for scoring at inference. Without SFT first, DPO would waste capacity learning the output format instead of refining decision margins.

**Output:** `score = log P("toxic"|x) − log P("not toxic"|x)`. Positive = toxic, negative = not toxic, magnitude = decision margin.

SFT produces well-calibrated probabilities (expected calibration error is low). DPO sacrifices calibration for better ranking. Predictions concentrate near P ≈ 0 and P ≈ 1 with little in between. This is acceptable for thresholding but means scores should be treated as decision margins, not literal risk estimates.

Hardware constraints required 4-bit NF4 quantization (NF4 over FP4 because model weights are approximately Gaussian, giving NF4 higher effective resolution near zero), LoRA rank 16 on attention projections, and gradient accumulation to simulate batch size 16 from per-device batch size 1. Evaluation runs in float16 without quantization (~5.5GB VRAM). Full training details are in the [notebook](rlhf_content_moderation.ipynb).

**Dataset:** [Civil Comments](https://huggingface.co/datasets/google/civil_comments): 1.8M news-site comments (2015-2017) with crowdsourced toxicity annotations (2018-2019). Each label is the fraction of annotators who flagged the comment, not a ground-truth determination. The model learns an approximation of historical annotator judgment, shaped by the source, era, and biases of that annotation pool.

## Evaluation Summary

The dataset is 85% non-toxic. A trivial "always not toxic" classifier scores 85% accuracy. PR-AUC is used instead because it measures ranking quality across all thresholds without inflation from class imbalance. DPO optimizes the same log-probability difference used for scoring, so training objective and evaluation metric are aligned.

| Model | PR-AUC | ROC-AUC |
|-------|--------|---------|
| Base  | 0.223  | 0.648   |
| + SFT | 0.729  | 0.924   |
| + DPO | 0.752  | 0.931   |

SFT accounts for the vast majority of the improvement (+0.506 PR-AUC). DPO's marginal gain (+0.023) initially appeared minor. The constraint-aware analysis below reveals that this small aggregate improvement concentrates exactly where it matters operationally.

## Decision Policy Simulation

Aggregate metrics measure scoring quality. They do not answer the operational question: given real constraints, what does the system *do*?

### Budget-Constrained: Fixed Review Capacity

The platform can review a fixed number of comments per day. The model triages into allow, block, and escalate to minimize a weighted cost of false negatives (toxicity leaked through auto-allow) and false positives (clean content auto-blocked). The cost weight is a policy parameter, not a model parameter.

| Budget | SFT FN% | DPO FN% | SFT Missed | DPO Missed | SFT Block Precision | DPO Block Precision |
|--------|---------|---------|------------|------------|---------------------|---------------------|
| 5%     | 2.70%   | 2.57%   | 781        | 753        | 52.2%               | 54.5%               |
| 10%    | 2.56%   | 2.34%   | 733        | 673        | 58.9%               | 60.8%               |
| 15%    | 2.52%   | 2.34%   | 717        | 673        | 68.1%               | 71.0%               |
| 25%    | 1.82%   | 1.70%   | 475        | 448        | 77.6%               | 80.3%               |

DPO misses fewer toxic items at every budget level. At 10% review budget, bootstrap resampling (n=300) shows DPO outperforms SFT in 99% of resamples (median reduction: 175 fewer missed toxic items, 95% CI: [14, 320]).

Cost-sensitive analysis across FN:FP ratios shows DPO reduces expected cost at every operating point, with the largest reduction (4.7%) at a 5:1 ratio the regime where policy decisions hinge on fine-grained ranking of borderline cases. At extreme ratios, thresholds are dominated by policy constraints and model ordering matters less.

### Risk-Bounded: Fixed Safety Requirements

The constraints flip. Safety is fixed; the question is how much content the system can auto-resolve.

| Policy | SFT Auto-Rate | DPO Auto-Rate | Δ | SFT FN (actual) | DPO FN (actual) |
|--------|---------------|---------------|---|-----------------|-----------------|
| Kid Safe (≤1%) | 97.7% | 97.8% | +0.1pp | 0.99% | 0.99% |
| Default (≤2%) | 98.3% | 98.4% | +0.1pp | 1.95% | 2.00% |
| Permissive (≤5%) | 99.2% | 99.2% | −0.0pp | 4.90% | 4.96% |

Under fixed safety ceilings, DPO adds ≤ 0.1 percentage points of automation—effectively zero. Both models auto-resolve 97–99% of content. This is not a failure of DPO; it is a property of the constraint. When operating in the extreme tails of the score distribution, both models have sufficient separation. DPO's ranking improvements live near the decision boundary, which only matters when the boundary is the binding constraint.

### What This Means

DPO's value is determined by which constraint binds:

- **Review capacity is the bottleneck** (platform scaling faster than it can hire T&S staff): DPO delivers measurable safety improvements per reviewer-hour. This is the regime where DPO justifies its training cost.
- **Safety policy is the bottleneck** (fixed FN ceiling): both models saturate. DPO adds negligible value.

This distinction is invisible in PR-AUC and only emerges through constraint-aware simulation. It also means the decision to use DPO should depend on the platform's operating regime, not on aggregate model metrics.

## Cost and Impact

Illustrative estimates for a second-stage volume of 250,000 comments/day at $0.15–$0.40 per human review:

| Mode | Auto-Resolved/Day | Escalated/Day | Review Savings/Year | Achieved FN Rate |
|------|-------------------|---------------|---------------------|------------------|
| Kid Safe | 244,437 | 5,563 | $13.4M–$35.7M | 0.99% |
| Default | 246,081 | 3,919 | $13.5M–$35.9M | 2.00% |
| Permissive | 247,912 | 2,088 | $13.6M–$36.2M | 4.96% |

These are upper-bound estimates. They assume all escalated volume would otherwise require human review and that the second-stage scorer is the only automated layer. Real savings depend on the existing pipeline and the marginal cost of adding this component versus alternatives (e.g., a classification head, which would be cheaper to serve).

The relevant comparison is not "model vs. no model" but "model vs. the next-cheapest option that meets the same safety bound." This project does not yet include that comparison (see Design Tradeoffs).

## Lifecycle and Feedback Loop

This system does not learn online. The model is trained offline and deployed as a fixed scorer. Adaptation happens through three mechanisms, ordered by cost:

1. **Threshold tuning.** If escalation volume drifts or reviewer override rates change, the platform adjusts allow/block thresholds. No retraining. This is the primary adaptation mechanism and the reason the scorer and policy are separated.

2. **Targeted data augmentation.** Addresses systematic slice failures identified through monitoring or the escalation queue. Disagreements between model decisions and reviewer overrides become candidate DPO preference pairs for the next training round. The pipeline supports this without architectural changes.

3. **Model replacement.** Reserved for distribution shifts that threshold tuning cannot absorb (e.g., new content formats, language drift). The budget-constrained and risk-bounded evaluation tables provide a direct A/B protocol: run the candidate model through the same simulation and compare safety-per-reviewer at matched budgets.

**Monitoring signals:** FN rate on auto-allowed content, escalation rate trend, block precision, reviewer override rate, and slice-level metric drift. Deviations trigger investigation, not automatic updates.

**What this system explicitly does not do:** online learning, automatic threshold adjustment, or self-supervised retraining. Each of these introduces feedback loops that are difficult to audit. The tradeoff is slower adaptation in exchange for predictable behavior.

## Design Tradeoffs and Limitations

**DPO sacrifices calibration for ranking.** SFT produces well-calibrated probabilities; DPO pushes predictions toward P ≈ 0 and P ≈ 1. This improves threshold-based decisions but means scores cannot be interpreted as risk probabilities. Any downstream system that needs calibrated confidence (e.g., for expected-value calculations) would need post-hoc recalibration.

**Single scorer vs. per-policy models.** A single model with threshold adjustment is simpler to maintain but means all operating modes share the same error distribution. A kid-safe deployment might benefit from a model specifically trained to minimize FN on the content types kids encounter. This project chose simplicity; the cost is that slice-level guarantees are weaker than aggregate guarantees.

**Annotation bias.** The model approximates 2018–2019 annotator judgments on 2015–2017 news comments. Toxicity norms shift. The slice analysis found the model outperforms labels on some clear insults (annotators scored below 0.3 on comments containing direct name-calling), which suggests label noise at the decision boundary. In deployment, the model's notion of toxicity will drift from current community standards without periodic relabeling and retraining.

**Static simulation vs. production dynamics.** The policy simulation uses a fixed test set and assumes stationary content distribution. In production, the escalation queue creates a feedback loop: content that gets escalated influences retraining data, which changes what future content gets escalated. This project does not model that loop.

**Missing baseline.** A classification head on the same base model would be cheaper to serve (no autoregressive generation) and might achieve comparable ranking quality. This project chose log-probability scoring because it aligns training and inference objectives for DPO, but the cost-performance tradeoff against a classification head is not yet quantified.

**Slice coverage.** DPO changes behavior asymmetrically: it increases confidence on short comments and quoted speech (reducing false positives) but stays cautious on profanity, ALL CAPS, and sarcastic laughter (no confidence gain even when benign). Recall is stable across all tested slices, but the slice set is not exhaustive—adversarial content, code-switching, and non-English text are not covered.

## Project Structure and Reproducibility

```
civil-toxicity-rlhf/
├── src/
│   ├── scoring.py              # Evaluation and visualization
│   ├── policy.py               # Decision policy simulation
│   └── training.py             # Model loading and quantization
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

civil-toxicity-artifacts/       # Outside repo, not tracked
├── checkpoints/
│   ├── sft/final/
│   └── dpo/final/              # Includes SFT adapter
└── results/                    # Evaluation CSVs
```

### Running

```bash
git clone https://github.com/emergentdaniel/civil-toxicity-rlhf.git
cd civil-toxicity-rlhf
pip install -r requirements.txt
jupyter notebook rlhf_content_moderation.ipynb
```

The notebook runs top-to-bottom: downloads data, trains SFT and DPO, evaluates all three model stages, and runs policy simulations. A fixed seed controls all data splitting and initialization.

```bash
# Docker
docker build -t civil-toxicity-rlhf .
docker run --gpus all -p 8888:8888 civil-toxicity-rlhf
```

**Hardware:** RTX 4070 (8GB VRAM). Training uses 4–6GB, evaluation uses ~5.5GB. Total runtime ~5 hours (2h training + 3h evaluation on 40k samples).

## TL;DR

- **Problem owned:** reducing human review load for content moderation while bounding false-negative risk under explicit operational constraints.
- **Decision the system makes:** auto-allow, auto-block, or escalate to a human. Controlled by two auditable thresholds that define the platform's risk tolerance.
- **Tradeoff accepted:** DPO improves safety-per-reviewer when review capacity is the bottleneck, but adds near-zero value when a safety ceiling is the binding constraint. The model also sacrifices probability calibration for better ranking, which is acceptable for thresholding but limits downstream use as a risk estimator.

## License

MIT

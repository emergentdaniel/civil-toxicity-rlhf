#!/usr/bin/env python3
"""Train DPO with specified beta. Reuses frozen SFT checkpoint.

Usage:
    python scripts/train_dpo.py --beta 0.1
    python scripts/train_dpo.py --beta 0.05 --seed 42

    # Sweep:
    for beta in 0.05 0.1 0.2 0.5; do
        python scripts/train_dpo.py --beta $beta
    done

Expects:
    - SFT adapter at checkpoints/sft/final/ (from main notebook)
    - DPO data at data/civil_comments_dpo_{train,eval}.jsonl (from main notebook)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

from src.training import load_sft_model, get_tokenizer, clear_memory


# Defaults match the notebook's Config dataclass
DEFAULTS = {
    'model_id': 'meta-llama/Llama-3.2-1B-Instruct',
    'seed': 67,
    'data_dir': 'data',
    'sft_adapter_path': 'checkpoints/sft/final',

    # Quantization
    'load_in_4bit': True,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,

    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],

    # DPO training
    'dpo_epochs': 1,
    'dpo_batch_size': 1,
    'dpo_gradient_accumulation': 16,
    'dpo_learning_rate': 5e-5,
    'dpo_max_length': 256,
    'dpo_max_prompt_length': 224,

    # Evaluation
    'test_samples': 90000,
}


def main():
    parser = argparse.ArgumentParser(description="Train DPO with specified beta")
    parser.add_argument('--beta', type=float, required=True, help='DPO beta (KL penalty)')
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'], help='Random seed')
    args = parser.parse_args()

    # Build run directory
    run_id = f"beta_{args.beta}_seed{args.seed}"
    run_dir = Path('runs/dpo_beta_stability') / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build config for this run
    cfg = {**DEFAULTS, 'dpo_beta': args.beta, 'seed': args.seed}
    cfg['results_dir'] = str(run_dir / 'results')
    cfg['figures_dir'] = str(run_dir / 'figures')
    cfg['dpo_output_dir'] = str(run_dir / 'checkpoints' / 'dpo')
    cfg['dpo_adapter_path'] = str(run_dir / 'checkpoints' / 'dpo' / 'final')

    # Create directories
    for d in [cfg['results_dir'], cfg['figures_dir'], cfg['dpo_output_dir']]:
        os.makedirs(d, exist_ok=True)

    # Save frozen config for reproducibility
    config_path = run_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Simple attribute access
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    config = Config(cfg)

    print(f"{'='*60}")
    print(f"DPO Training: beta={args.beta}, seed={args.seed}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}")

    # Verify SFT checkpoint exists
    sft_path = Path(config.sft_adapter_path)
    if not sft_path.exists():
        print(f"\nERROR: SFT adapter not found at {sft_path}")
        print("Run the main notebook first to train SFT.")
        sys.exit(1)

    # Load data
    train_dpo = load_dataset('json', data_files=f'{config.data_dir}/civil_comments_dpo_train.jsonl', split='train')
    eval_dpo = load_dataset('json', data_files=f'{config.data_dir}/civil_comments_dpo_eval.jsonl', split='train')
    print(f"DPO train: {len(train_dpo):,} | DPO eval: {len(eval_dpo):,}")

    # Load frozen SFT model
    model, tokenizer = load_sft_model(config, trainable=True)
    model.print_trainable_parameters()

    # DPO training
    dpo_config = DPOConfig(
        output_dir=config.dpo_output_dir,
        num_train_epochs=config.dpo_epochs,
        per_device_train_batch_size=config.dpo_batch_size,
        gradient_accumulation_steps=config.dpo_gradient_accumulation,
        learning_rate=config.dpo_learning_rate,
        beta=config.dpo_beta,
        max_length=config.dpo_max_length,
        max_prompt_length=config.dpo_max_prompt_length,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        gradient_checkpointing=False,
        report_to='none',
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dpo,
        eval_dataset=eval_dpo,
        processing_class=tokenizer,
    )

    print(f"\nStarting DPO training (beta={config.dpo_beta})...")
    dpo_trainer.train()

    # Save
    dpo_trainer.save_model(config.dpo_adapter_path)
    tokenizer.save_pretrained(config.dpo_adapter_path)
    print(f"\nSaved DPO adapter to {config.dpo_adapter_path}")
    print(f"Run complete: {run_dir}")
    print(f"\nNext: python scripts/evaluate.py {run_dir}")

    clear_memory()


if __name__ == '__main__':
    main()

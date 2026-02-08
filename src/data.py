"""Dataset formatting for SFT and DPO training."""

import os
from datasets import load_dataset


def format_sft_example(example: dict, toxic_threshold: float = 0.3) -> dict:
    """Convert a Civil Comments example to SFT format with binary label."""
    label = 'toxic' if example['toxicity'] >= toxic_threshold else 'not toxic'
    prompt = (
        f"Is this comment toxic? Answer only 'toxic' or 'not toxic'.\n\n"
        f"Comment: {example['text']}\n\n"
        f"Answer:"
    )
    return {'prompt': prompt, 'completion': f' {label}'}


def format_dpo_example(example: dict, toxic_threshold: float = 0.3) -> dict:
    """Convert a Civil Comments example to DPO format with chosen/rejected responses."""
    is_toxic = example['toxicity'] >= toxic_threshold
    prompt = (
        f"Is this comment toxic? Answer only 'toxic' or 'not toxic'.\n\n"
        f"Comment: {example['text']}\n\n"
        f"Answer:"
    )
    if is_toxic:
        return {'prompt': prompt, 'chosen': ' toxic', 'rejected': ' not toxic'}
    else:
        return {'prompt': prompt, 'chosen': ' not toxic', 'rejected': ' toxic'}


def prepare_datasets(config) -> None:
    """
    Prepare and save SFT, DPO, and test datasets.

    Writes JSONL files to config.data_dir:
        - civil_comments_sft_train.jsonl
        - civil_comments_sft_eval.jsonl
        - civil_comments_dpo_train.jsonl
        - civil_comments_dpo_eval.jsonl
        - civil_comments_test.jsonl
    """
    dataset = load_dataset(config.dataset_name)
    os.makedirs(config.data_dir, exist_ok=True)

    # SFT datasets
    train_sft = dataset['train'].shuffle(seed=config.seed).select(range(config.sft_train_samples))
    train_sft = train_sft.map(
        lambda x: format_sft_example(x, config.toxic_threshold),
        remove_columns=dataset['train'].column_names,
    )
    eval_sft = dataset['validation'].shuffle(seed=config.seed).select(range(config.sft_eval_samples))
    eval_sft = eval_sft.map(
        lambda x: format_sft_example(x, config.toxic_threshold),
        remove_columns=dataset['validation'].column_names,
    )

    # DPO datasets
    train_dpo = dataset['train'].shuffle(seed=config.seed + 1).select(range(config.dpo_train_samples))
    train_dpo = train_dpo.map(
        lambda x: format_dpo_example(x, config.toxic_threshold),
        remove_columns=dataset['train'].column_names,
    )
    eval_dpo = dataset['validation'].shuffle(seed=config.seed + 2).select(range(config.dpo_eval_samples))
    eval_dpo = eval_dpo.map(
        lambda x: format_dpo_example(x, config.toxic_threshold),
        remove_columns=dataset['validation'].column_names,
    )

    # Test dataset (SFT format for evaluation)
    test = dataset['test'].shuffle(seed=config.seed).select(range(config.test_samples))
    test = test.map(
        lambda x: format_sft_example(x, config.toxic_threshold),
        remove_columns=dataset['test'].column_names,
    )

    # Save
    train_sft.to_json(os.path.join(config.data_dir, 'civil_comments_sft_train.jsonl'))
    eval_sft.to_json(os.path.join(config.data_dir, 'civil_comments_sft_eval.jsonl'))
    train_dpo.to_json(os.path.join(config.data_dir, 'civil_comments_dpo_train.jsonl'))
    eval_dpo.to_json(os.path.join(config.data_dir, 'civil_comments_dpo_eval.jsonl'))
    test.to_json(os.path.join(config.data_dir, 'civil_comments_test.jsonl'))

    print(f"SFT train: {len(train_sft):,} | SFT eval: {len(eval_sft):,}")
    print(f"DPO train: {len(train_dpo):,} | DPO eval: {len(eval_dpo):,}")
    print(f"Test: {len(test):,}")
    print(f"Saved to {config.data_dir}/")

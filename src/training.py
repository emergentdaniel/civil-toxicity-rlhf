"""Model loading, quantization, and training helpers."""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel


def clear_memory() -> None:
    """Clear GPU memory between training runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_bnb_config(config) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig from project config."""
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def get_lora_config(config) -> LoraConfig:
    """Create LoRA config from project config."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=list(config.lora_target_modules),
    )


def get_tokenizer(config) -> AutoTokenizer:
    """Load tokenizer with pad token set."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_quantized_model(config):
    """Load base model with 4-bit quantization."""
    bnb_config = get_bnb_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map='auto',
    )
    return model


def load_sft_model(config, trainable: bool = False):
    """
    Load SFT model (base + adapter).
    
    Args:
        config: project config with model_id and sft_adapter_path
        trainable: if True, prepare for further training (DPO)
    
    Returns:
        (model, tokenizer)
    """
    clear_memory()
    bnb_config = get_bnb_config(config)
    
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map='auto',
    )
    
    if trainable:
        base = prepare_model_for_kbit_training(base)
        model = PeftModel.from_pretrained(base, config.sft_adapter_path, is_trainable=True)
    else:
        model = PeftModel.from_pretrained(base, config.sft_adapter_path)
    
    tokenizer = get_tokenizer(config)
    return model, tokenizer


def load_dpo_model(config):
    """Load DPO model (base + DPO adapter)."""
    clear_memory()
    bnb_config = get_bnb_config(config)
    
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(base, config.dpo_adapter_path)
    tokenizer = get_tokenizer(config)
    return model, tokenizer

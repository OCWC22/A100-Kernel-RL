"""
Unified model loading for KernelForge training pipeline.

Two paths:
  Primary: Qwen3-Coder-Next (80B MoE, ~3.9B active) via HF AutoModelForCausalLM + peft LoraConfig
           Unsloth cannot do GPTQ-based QLoRA for MoE architectures.
  Fallback: Qwen2.5-Coder-7B-Instruct via unsloth.FastLanguageModel (dense model)

LoRA targets differ by architecture:
  MoE: attention + shared_expert only (not all 512 routed experts → ~207GB)
  Dense: all linear projection layers
"""
from __future__ import annotations

import os

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")

# Primary model (MoE)
PRIMARY_MODEL = os.getenv("KERNELFORGE_MODEL", "Qwen/Qwen3-Coder-Next")
# Fallback model (dense)
FALLBACK_MODEL = os.getenv("KERNELFORGE_FALLBACK_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

# LoRA constants
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

MOE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
]

DENSE_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LENGTH = 8192

# Singleton cache
_model = None
_tokenizer = None
_model_type = None  # "moe" or "dense"


def load_model_and_tokenizer(
    force_fallback: bool = False,
    checkpoint_path: str | None = None,
):
    """Load model and tokenizer with automatic fallback.

    Args:
        force_fallback: Skip primary model, go straight to Qwen2.5.
        checkpoint_path: Load from a fine-tuned checkpoint instead of base model.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    global _model, _tokenizer, _model_type
    if _model is not None:
        return _model, _tokenizer

    if checkpoint_path:
        _model, _tokenizer = _load_from_checkpoint(checkpoint_path)
        return _model, _tokenizer

    if not force_fallback:
        try:
            _model, _tokenizer = _load_primary()
            _model_type = "moe"
            return _model, _tokenizer
        except Exception as e:
            print(f"Primary model ({PRIMARY_MODEL}) failed: {e}")
            print(f"Falling back to {FALLBACK_MODEL}...")

    _model, _tokenizer = _load_fallback()
    _model_type = "dense"
    return _model, _tokenizer


def get_model_type() -> str | None:
    """Return 'moe' or 'dense' depending on which model loaded."""
    return _model_type


def _load_primary():
    """Load Qwen3-Coder-Next via HF AutoModelForCausalLM + peft LoraConfig."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Loading primary model: {PRIMARY_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(
        PRIMARY_MODEL,
        trust_remote_code=True,
    )

    # BitsAndBytes 4-bit config for MoE
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        PRIMARY_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=MOE_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Primary model loaded: {trainable:,} trainable / {total:,} total params "
          f"({trainable / total * 100:.2f}%)")

    return model, tokenizer


def _load_fallback():
    """Load Qwen2.5-Coder-7B-Instruct via Unsloth FastLanguageModel."""
    from unsloth import FastLanguageModel, PatchFastRL

    print(f"Loading fallback model: {FALLBACK_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FALLBACK_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=DENSE_LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_gradient_checkpointing="unsloth",
    )

    PatchFastRL("GRPO", FastLanguageModel)

    print(f"Fallback model loaded: {FALLBACK_MODEL}")
    return model, tokenizer


def _load_from_checkpoint(checkpoint_path: str):
    """Load a fine-tuned checkpoint (works for both MoE and dense)."""
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")

    # Try Unsloth first (works for both if saved via Unsloth)
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        print(f"Loaded checkpoint via Unsloth: {checkpoint_path}")
        return model, tokenizer
    except Exception:
        pass

    # Fall back to HF + PEFT
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Try loading PEFT adapter if present
    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded PEFT adapter from {checkpoint_path}")
    except Exception:
        print(f"No PEFT adapter found, using base model from {checkpoint_path}")

    return model, tokenizer

"""
training/lora_config.py

Defines all hyperparameters for the SigLIP 2 LoRA fine-tuning run.

All values are centralised here so neither the training script nor the Kaggle
notebook need any magic numbers scattered through their logic.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """
    LoRA adapter configuration for the SigLIP 2 model.

    Attributes:
        r: LoRA rank. Controls the number of trainable parameters per layer.
           Higher rank captures more fine-grained adaptation but uses more memory.
        lora_alpha: Scaling factor for LoRA updates. Effective learning rate for
                    LoRA layers scales as lora_alpha / r. Setting alpha = 2 * r
                    is a common and stable default.
        lora_dropout: Dropout applied to LoRA layers for regularisation.
        target_modules: Attention projection layer names to apply LoRA to.
                        Using q_proj and v_proj in both vision and text towers
                        is the standard approach — k_proj and out_proj
                        are left frozen to limit parameter count.
        bias: Whether to train bias terms. 'none' keeps base model biases frozen,
              which is the recommended setting when base weights are frozen.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class TrainingConfig:
    """
    Training loop configuration for the SigLIP 2 LoRA fine-tuning run.

    Attributes:
        model_name: HuggingFace model identifier for the base SigLIP 2 model.
        num_epochs: Number of complete passes over the training dataset.
                    Two caption variants are generated per product per epoch,
                    so the effective number of unique text inputs seen is 2x.
        batch_size: Number of image-text pairs per training step.
                    32 is safe for a T4 GPU (16 GB VRAM) with fp16 and
                    gradient checkpointing enabled.
        learning_rate: Peak learning rate for AdamW. The scheduler warms up
                       for warmup_ratio of total steps then decays on a cosine
                       curve back toward zero.
        warmup_ratio: Fraction of total training steps used for linear warmup.
        weight_decay: L2 regularisation coefficient applied to non-bias parameters.
        max_grad_norm: Maximum gradient norm for clipping. Guards against loss
                       spikes, which are more likely at the relatively high
                       learning rate used for LoRA.
        lr_scheduler_type: Shape of the learning rate decay curve after warmup.
                           Cosine decay outperforms linear for most fine-tuning
                           tasks and avoids an abrupt drop at the end of training.
        save_strategy: When to write checkpoints. 'epoch' guarantees at least
                       one checkpoint per epoch regardless of dataset size,
                       unlike step-based saving which can silently produce
                       nothing if total_steps < save_steps.
        evaluation_strategy: When to compute validation loss. Must match
                              save_strategy so the saved checkpoint corresponds
                              to an evaluated state.
        gradient_accumulation_steps: Number of forward passes before an
                                     optimiser step. Effective batch size =
                                     batch_size * gradient_accumulation_steps.
                                     Set > 1 if the physical batch OOMs on T4.
        fp16: Whether to run in 16-bit floating point. Required on T4 to fit
              the model and a useful batch size within VRAM limits.
        gradient_checkpointing: Re-computes activations during the backward
                                 pass instead of storing them. Reduces peak
                                 VRAM at the cost of ~20% throughput.
        logging_steps: Emit a training log line every this many optimiser steps.
        val_split_ratio: Fraction of the dataset held out for validation.
        output_dir: Directory where adapter checkpoints and the final adapter
                    weights are written.
        seed: Random seed for dataset splitting and weight initialisation.
    """

    model_name: str = "google/siglip2-base-patch16-224"
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 50
    val_split_ratio: float = 0.1
    output_dir: str = "training/checkpoints"
    seed: int = 42


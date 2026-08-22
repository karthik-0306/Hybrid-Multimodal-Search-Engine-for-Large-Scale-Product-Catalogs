"""
training/train.py

Fine-tunes google/siglip2-base-patch16-224 on the cleaned ABO product dataset
using LoRA adapters applied to the attention projection layers of both the vision
tower and the text tower.

The training objective is SigLIP's sigmoid loss: for every (image, caption) pair
in a batch, the model is trained to maximise the similarity of the pair and
minimise the similarity of all other in-batch cross-pairs. Unlike CLIP's softmax
loss, SigLIP's sigmoid formulation treats each pair independently, which makes
it more stable at smaller batch sizes.

Usage:
    python training/train.py
"""

import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from data_pipeline.schema import build_caption
from training.lora_config import LoRAConfig, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ABOProductDataset(Dataset):
    """
    PyTorch Dataset that serves (image, caption) pairs for SigLIP 2 training.

    For each product, two caption phrasings are generated: one that includes the
    brand name and one that omits it. On each epoch the dataset alternates between
    the two variants so the model sees both phrasings over the full training run.

    Args:
        records: List of product dicts containing at minimum 'main_image_path'
                 and the optional attribute fields used by build_caption.
        processor: The AutoProcessor loaded from the SigLIP 2 checkpoint.
        images_base_dir: Absolute path to the directory containing the image files.
        caption_variant: 0 for with-brand phrasing, 1 for without-brand phrasing.
    """

    def __init__(
        self,
        records: List[dict],
        processor: AutoProcessor,
        images_base_dir: Path,
        caption_variant: int = 0,
    ) -> None:
        self.records = records
        self.processor = processor
        self.images_base_dir = images_base_dir
        self.caption_variant = caption_variant

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single processed (image, caption) pair as model-ready tensors.

        If the image file cannot be opened, a grey placeholder is substituted
        silently to avoid crashing the training run on a single corrupt file.

        Args:
            idx: Index into the records list.

        Returns:
            Dict with keys 'pixel_values', 'input_ids', and 'attention_mask'.
        """
        record = self.records[idx]
        img_path = self.images_base_dir / record["main_image_path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logger.warning("Could not open image at %s, substituting placeholder", img_path)
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        include_brand = (self.caption_variant == 0)
        caption = build_caption(record, include_brand=include_brand)

        inputs = self.processor(
            text=[caption],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {key: val.squeeze(0) for key, val in inputs.items()}


def create_train_val_split(
    records: List[dict],
    val_split_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Shuffles and splits records into training and validation sets.

    The split is stratified by order only — a random shuffle followed by a
    fixed cutpoint. For a dataset of this size (56K records), simple random
    splitting gives sufficient class balance without stratification.

    Args:
        records: Full list of product dicts.
        val_split_ratio: Fraction of records to hold out for validation.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_records, val_records).
    """
    shuffled = records.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    val_size = int(len(shuffled) * val_split_ratio)
    return shuffled[val_size:], shuffled[:val_size]


def build_lora_model(base_model_name: str, lora_cfg: LoRAConfig) -> torch.nn.Module:
    """
    Loads the SigLIP 2 base model and wraps it with LoRA adapters.

    Only the q_proj and v_proj layers in both the vision and text encoders are
    made trainable. All other parameters remain frozen at their pretrained values.

    Args:
        base_model_name: HuggingFace model identifier.
        lora_cfg: LoRAConfig instance specifying adapter hyperparameters.

    Returns:
        A PEFT-wrapped SigLIP 2 model with LoRA adapters applied.
    """
    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    processor: AutoProcessor,
    val_records: List[dict],
    images_base_dir: Path,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> float:
    """
    Computes average SigLIP sigmoid loss on the validation set.

    Uses caption_variant=0 (with-brand) consistently for evaluation so that
    the metric is comparable across epochs regardless of which training variant
    was used that epoch.

    Args:
        model: The PEFT-wrapped SigLIP 2 model in eval mode.
        processor: The AutoProcessor for the SigLIP 2 checkpoint.
        val_records: List of held-out product dicts.
        images_base_dir: Root path for resolving image file paths.
        train_cfg: TrainingConfig with batch_size and fp16 settings.
        device: torch.device to run evaluation on.

    Returns:
        Average loss over all validation batches.
    """
    model.eval()
    dataset = ABOProductDataset(val_records, processor, images_base_dir, caption_variant=0)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    total_loss = 0.0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=train_cfg.fp16):
            outputs = model(**batch, return_loss=True)
        total_loss += outputs.loss.item()

    model.train()
    return total_loss / len(dataloader)


def run_training(
    model: torch.nn.Module,
    processor: AutoProcessor,
    train_records: List[dict],
    val_records: List[dict],
    images_base_dir: Path,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """
    Executes the full training loop over all epochs.

    Saving and evaluation both happen at epoch boundaries (not step boundaries)
    so checkpoints always correspond to evaluated states and at least one
    checkpoint is guaranteed regardless of dataset size.

    A best-checkpoint is tracked separately from the per-epoch checkpoints: the
    epoch with the lowest validation loss is saved to output_dir/best_checkpoint.
    The teammate running inference should load that checkpoint, not necessarily
    the final one.

    Args:
        model: The PEFT-wrapped SigLIP 2 model.
        processor: The AutoProcessor for the SigLIP 2 checkpoint.
        train_records: Training split product dicts.
        val_records: Validation split product dicts.
        images_base_dir: Root path for resolving image file paths.
        train_cfg: TrainingConfig with all hyperparameters.
        device: torch.device to run training on.
    """
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_records) / train_cfg.batch_size)
    # Effective optimiser steps account for gradient accumulation
    effective_steps_per_epoch = math.ceil(
        steps_per_epoch / train_cfg.gradient_accumulation_steps
    )
    total_steps = effective_steps_per_epoch * train_cfg.num_epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.fp16)

    global_step = 0
    best_val_loss = float("inf")
    model.train()

    for epoch in range(train_cfg.num_epochs):
        caption_variant = epoch % 2
        variant_label = "with-brand" if caption_variant == 0 else "without-brand"
        dataset = ABOProductDataset(
            train_records, processor, images_base_dir, caption_variant
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        epoch_loss = 0.0
        logger.info(
            "Epoch %d/%d — caption variant: %s",
            epoch + 1,
            train_cfg.num_epochs,
            variant_label,
        )

        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=train_cfg.fp16):
                outputs = model(**batch, return_loss=True)
                # Scale loss by accumulation steps so gradients average correctly
                loss = outputs.loss / train_cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += outputs.loss.item()

            # Only step the optimiser after accumulating enough gradients
            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_cfg.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % train_cfg.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        "Step %d | epoch %d | train loss %.4f | lr %.2e",
                        global_step,
                        epoch + 1,
                        avg_loss,
                        lr_now,
                    )

        avg_train_loss = epoch_loss / steps_per_epoch
        logger.info("Epoch %d train loss: %.4f", epoch + 1, avg_train_loss)

        # Evaluate at the end of every epoch
        val_loss = evaluate(model, processor, val_records, images_base_dir, train_cfg, device)
        logger.info("Epoch %d val loss:   %.4f", epoch + 1, val_loss)

        # Save a per-epoch checkpoint (adapter weights only)
        ckpt_path = output_dir / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(str(ckpt_path))
        logger.info("Checkpoint saved: %s", ckpt_path)

        # Track the epoch with the best validation loss separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_checkpoint"
            model.save_pretrained(str(best_path))
            logger.info(
                "New best val loss %.4f — adapter saved to %s", best_val_loss, best_path
            )

    # Always write the final adapter, even if it isn't the best epoch
    final_path = output_dir / "lora_adapter_final"
    model.save_pretrained(str(final_path))
    logger.info("Final LoRA adapter weights saved to %s", final_path)
    logger.info("Best validation loss: %.4f", best_val_loss)


def main() -> None:
    """
    Entry point for running the SigLIP 2 LoRA fine-tuning pipeline locally.

    On a CPU-only machine this will run extremely slowly. The warning message
    is intentional — local execution is meant for code validation and smoke
    testing only. Full training should be run via kaggle_finetune.ipynb on a
    Kaggle T4 GPU instance.
    """
    random.seed(42)
    torch.manual_seed(42)

    project_root = Path(__file__).resolve().parent.parent
    config = Config(project_root)
    train_cfg = TrainingConfig()
    lora_cfg = LoRAConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    if device.type == "cpu":
        logger.warning(
            "No GPU detected. Training on CPU is intended for code validation only, "
            "not full fine-tuning. Run kaggle_finetune.ipynb on a T4 GPU for actual training."
        )
        train_cfg.fp16 = False
        train_cfg.gradient_checkpointing = False

    logger.info("Loading products from %s", config.products_parquet)
    df = pd.read_parquet(config.products_parquet)
    records = df.to_dict(orient="records")
    logger.info("Total records loaded: %d", len(records))

    train_records, val_records = create_train_val_split(
        records, train_cfg.val_split_ratio, train_cfg.seed
    )
    logger.info("Train: %d | Val: %d", len(train_records), len(val_records))

    logger.info("Loading processor for %s", train_cfg.model_name)
    processor = AutoProcessor.from_pretrained(train_cfg.model_name)

    model = build_lora_model(train_cfg.model_name, lora_cfg)
    model.to(device)

    run_training(
        model=model,
        processor=processor,
        train_records=train_records,
        val_records=val_records,
        images_base_dir=config.processed_images_dir,
        train_cfg=train_cfg,
        device=device,
    )


if __name__ == "__main__":
    main()

"""
Training script for the multilingual transliteration model.

Architecture : mT5-small  (encoder-decoder, ~300M params)
Backbone     : google/mt5-small  (already knows Indic Unicode)
Strategy     : Fine-tune on Aksharantar (hi + bn + ta) with language prefix tokens

Run:
    python train.py
    python train.py --epochs 5 --batch_size 64 --lr 1e-3
"""

import gc
import os
import sys
import json
import shutil
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_from_disk

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LANGUAGES,
    LANG_TOKEN,
    paths,
    model_config,
    training_config,
)
from dataset import build_multilingual_dataset, tokenise_dataset
from evaluate import build_compute_metrics, evaluate_model_on_test

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def cleanup_memory():
    """Free CPU and GPU memory after a major step."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup_checkpoints(output_dir: str, best_dir: str):
    """Delete all intermediate checkpoints; keep only the best model folder."""
    deleted = 0
    for item in Path(output_dir).iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            shutil.rmtree(item)
            deleted += 1
    logger.info(f"Deleted {deleted} intermediate checkpoint(s). Best model at: {best_dir}")


def save_to_drive(best_model_dir: str, drive_dir: str):
    """Copy the best model to Google Drive for persistent storage."""
    drive_path = Path(drive_dir)
    if not drive_path.parent.exists():
        logger.warning(f"Drive path not mounted: {drive_dir}. Skipping Drive save.")
        return
    drive_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best_model_dir, str(drive_path / "best_model"), dirs_exist_ok=True)
    logger.info(f"Best model copied to Drive: {drive_path / 'best_model'}")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train multilingual transliteration model")
    p.add_argument("--model_name",  default=model_config.base_model_name)
    p.add_argument("--epochs",      type=int,   default=training_config.num_train_epochs)
    p.add_argument("--batch_size",  type=int,   default=training_config.per_device_train_batch_size)
    p.add_argument("--eval_batch_size", type=int, default=training_config.per_device_eval_batch_size)
    p.add_argument("--lr",          type=float, default=training_config.learning_rate)
    p.add_argument("--warmup",      type=int,   default=training_config.warmup_steps)
    p.add_argument("--fp16",        action="store_true", default=training_config.fp16)
    p.add_argument("--output_dir",  default=paths.checkpoint_dir)
    p.add_argument("--data_dir",    default=paths.processed_data_dir)
    p.add_argument("--rebuild_data",action="store_true",
                   help="Force re-download and re-preprocess data")
    p.add_argument("--resume_from", default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


# ── Tokeniser setup ────────────────────────────────────────────────────────────

def setup_tokeniser(model_name: str):
    """
    Load mT5 tokeniser and add language tokens so the model can condition
    on which Indic script to produce.
    """
    logger.info(f"Loading tokeniser: {model_name}")
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    new_tokens = list(LANG_TOKEN.values())
    added = tokeniser.add_special_tokens({"additional_special_tokens": new_tokens})
    logger.info(f"Added {added} special language tokens: {new_tokens}")

    return tokeniser


# ── Data ───────────────────────────────────────────────────────────────────────

def prepare_data(data_dir: str, rebuild: bool, tokeniser):
    processed_path = Path(data_dir)

    if not rebuild and (processed_path / "tokenised").exists():
        logger.info("Loading cached tokenised dataset …")
        from datasets import load_from_disk as lfd
        return lfd(str(processed_path / "tokenised"))

    # Build or load raw dataset
    if not rebuild and processed_path.exists():
        logger.info("Loading cached raw dataset …")
        raw_ds = load_from_disk(str(processed_path))
    else:
        logger.info("Downloading + preprocessing Aksharantar …")
        raw_ds = build_multilingual_dataset(save_dir=data_dir)

    # Tokenise
    logger.info("Tokenising dataset …")
    tokenised = tokenise_dataset(raw_ds, tokeniser)
    tokenised.save_to_disk(str(processed_path / "tokenised"))
    return tokenised


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    set_seed(training_config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1. Tokeniser
    tokeniser = setup_tokeniser(args.model_name)

    # 2. Dataset
    tokenised_ds = prepare_data(args.data_dir, args.rebuild_data, tokeniser)
    logger.info(f"Train: {len(tokenised_ds['train']):,}  "
                f"Val: {len(tokenised_ds['validation']):,}  "
                f"Test: {len(tokenised_ds['test']):,}")

    # 3. Model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Resize embeddings only if tokeniser is larger than the model's current vocab.
    # Never shrink — the 3 new lang tokens (250100→250103) still fit inside mT5's
    # default vocab_size=250112, so no resize is actually needed here.
    if len(tokeniser) > model.config.vocab_size:
        num_new = len(tokeniser) - model.config.vocab_size
        model.resize_token_embeddings(len(tokeniser))
        # Initialize new token rows to the mean of existing embeddings so they
        # start in a reasonable range — random init overflows in fp16.
        with torch.no_grad():
            inp = model.get_input_embeddings().weight
            inp[-num_new:] = inp[:-num_new].mean(dim=0, keepdim=True).expand(num_new, -1)
            out = model.get_output_embeddings().weight
            out[-num_new:] = out[:-num_new].mean(dim=0, keepdim=True).expand(num_new, -1)
        logger.info(f"Resized embeddings: {model.config.vocab_size} → {len(tokeniser)}, "
                    f"initialised {num_new} new rows to embedding mean")
    else:
        logger.info(f"No resize needed: tokeniser ({len(tokeniser)}) fits inside "
                    f"model vocab ({model.config.vocab_size})")

    if training_config.gradient_checkpointing:
        model.config.use_cache = False   # required: cache is incompatible with grad checkpointing
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled (use_cache=False)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # 4. Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=training_config.weight_decay,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=model_config.max_target_length,
        generation_num_beams=training_config.num_beams,
        logging_dir=None,
        logging_steps=training_config.logging_steps,
        eval_strategy="steps",
        eval_steps=training_config.eval_steps,
        save_strategy="steps",
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        report_to="none",          # no tensorboard needed in Colab
        seed=training_config.seed,
        dataloader_num_workers=0,  # avoids multiprocessing issues in Colab
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    # 5. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokeniser,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    # 6. Compute metrics
    compute_metrics = build_compute_metrics(tokeniser)

    # 7. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_ds["train"],
        eval_dataset=tokenised_ds["validation"],
        processing_class=tokeniser,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # 8. Train
    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # 9. Save best model
    best_model_path = output_dir / "best"
    trainer.save_model(str(best_model_path))
    tokeniser.save_pretrained(str(best_model_path))
    logger.info(f"Best model saved to {best_model_path}")

    # 10. Delete intermediate checkpoints to free disk space
    cleanup_checkpoints(str(output_dir), str(best_model_path))

    # 11. Copy best model to Drive for persistence (no-op if Drive not mounted)
    save_to_drive(str(best_model_path), training_config.drive_save_dir)

    # 12. Free GPU memory before evaluation
    del model
    cleanup_memory()

    # Reload best model for evaluation
    from transformers import AutoModelForSeq2SeqLM as _M
    model = _M.from_pretrained(str(best_model_path))

    # 13. Evaluate on test set
    logger.info("Evaluating on test set …")
    test_results = evaluate_model_on_test(
        model, tokeniser,
        tokenised_ds["test"],
        batch_size=training_config.per_device_eval_batch_size,
        num_beams=training_config.num_beams,
        device=device,
    )

    results_path = Path(paths.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    logger.info("Test results:")
    logger.info(json.dumps(test_results, indent=2, ensure_ascii=False))

    return model, tokeniser, test_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)

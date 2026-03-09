"""
Evaluation metrics for transliteration:
  - Character Error Rate (CER)  — primary metric
  - Word Error Rate (WER)       — whole-word accuracy proxy
  - Top-1 Accuracy              — exact match
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import evaluate                       # HuggingFace evaluate library
from jiwer import cer as jiwer_cer   # pip install jiwer

logger = logging.getLogger(__name__)

# ── CER / WER via jiwer ────────────────────────────────────────────────────────

def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Character Error Rate — lower is better."""
    preds = [p.strip() for p in predictions]
    refs  = [r.strip() for r in references]
    return jiwer_cer(refs, preds)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Word-level accuracy for single-word transliterations:
    WER = 1 - exact_match_rate in this context.
    """
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return 1.0 - correct / len(references)


def compute_exact_accuracy(predictions: List[str], references: List[str]) -> float:
    """Fraction of predictions that exactly match the reference."""
    correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return correct / len(references)


# ── Per-language breakdown ─────────────────────────────────────────────────────

def evaluate_per_language(
    predictions: List[str],
    references: List[str],
    source_texts: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Split predictions/references by language prefix and compute metrics
    per language.

    source_texts should contain the prefixed source, e.g. "__hi__ kitab".
    """
    from config import LANGUAGES, LANG_TOKEN

    buckets: Dict[str, Tuple[List[str], List[str]]] = {
        lang: ([], []) for lang in LANGUAGES
    }

    for pred, ref, src in zip(predictions, references, source_texts):
        for lang in LANGUAGES:
            if src.startswith(LANG_TOKEN[lang]):
                buckets[lang][0].append(pred)
                buckets[lang][1].append(ref)
                break

    results = {}
    for lang, (preds, refs) in buckets.items():
        if not preds:
            continue
        results[lang] = {
            "cer":      round(compute_cer(preds, refs), 4),
            "wer":      round(compute_wer(preds, refs), 4),
            "accuracy": round(compute_exact_accuracy(preds, refs), 4),
            "n_samples": len(preds),
        }

    return results


# ── HuggingFace Trainer-compatible compute_metrics ────────────────────────────

def build_compute_metrics(tokeniser):
    """
    Returns a compute_metrics function compatible with HuggingFace Seq2SeqTrainer.
    """

    def compute_metrics(eval_preds):
        pred_ids, label_ids = eval_preds

        # Decode predictions
        pred_ids = np.where(pred_ids != -100, pred_ids, tokeniser.pad_token_id)
        decoded_preds = tokeniser.batch_decode(pred_ids, skip_special_tokens=True)

        # Decode labels
        label_ids = np.where(label_ids != -100, label_ids, tokeniser.pad_token_id)
        decoded_labels = tokeniser.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        cer_score = compute_cer(decoded_preds, decoded_labels)
        acc_score = compute_exact_accuracy(decoded_preds, decoded_labels)

        return {
            "cer":      round(cer_score, 4),
            "accuracy": round(acc_score, 4),
        }

    return compute_metrics


# ── Standalone evaluation on test set ─────────────────────────────────────────

def evaluate_model_on_test(
    model,
    tokeniser,
    test_dataset,
    batch_size: int = 256,
    num_beams: int = 4,
    device: str = "cpu",
) -> Dict:
    """
    Run beam-search decoding on the test set and return full metrics.

    Works with both HuggingFace models and CTranslate2 models
    (pass ct2_model=True to adjust decoding path).
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from transformers import default_data_collator

    model.eval()
    model.to(device)

    all_preds, all_refs, all_sources = [], [], []

    loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_data_collator)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=64,
            )

            preds   = tokeniser.batch_decode(generated, skip_special_tokens=True)
            labels  = batch["labels"]
            labels  = torch.where(labels != -100, labels, torch.tensor(tokeniser.pad_token_id))
            refs    = tokeniser.batch_decode(labels, skip_special_tokens=True)
            sources = tokeniser.batch_decode(input_ids, skip_special_tokens=False)

            all_preds.extend(preds)
            all_refs.extend(refs)
            all_sources.extend(sources)

    overall = {
        "overall_cer":      round(compute_cer(all_preds, all_refs), 4),
        "overall_accuracy": round(compute_exact_accuracy(all_preds, all_refs), 4),
    }
    per_lang = evaluate_per_language(all_preds, all_refs, all_sources)

    return {"overall": overall, "per_language": per_lang}


# ── CLI usage ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check
    preds = ["kitāb",  "namaste", "ধন্যবাদ"]
    refs  = ["kitaab", "namaste", "ধন্যবাদ"]
    print("CER:",      compute_cer(preds, refs))
    print("Accuracy:", compute_exact_accuracy(preds, refs))

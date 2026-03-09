"""
Dataset loading and preprocessing for Aksharantar transliteration data.

The Aksharantar dataset (ai4bharat/Aksharantar on HuggingFace) contains
Roman → Indic script word pairs for 21 Indian languages.

Each split contains:
  - native_word  : word in native Indic script  (target)
  - english_word : romanised / Latin-script word (source)
"""

import gc
import io
import os
import logging
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizer

from config import (
    LANGUAGES,
    LANGUAGE_CONFIGS,
    LANG_TOKEN,
    paths,
    model_config,
    training_config,
)

logger = logging.getLogger(__name__)


# ── Raw data loading ───────────────────────────────────────────────────────────

def load_aksharantar_language(lang_code: str, split: str = "train") -> Dataset:
    """
    Load Aksharantar for one Indic language by downloading only the specific
    language zip file directly from HuggingFace Hub.

    Args:
        lang_code : short code  e.g. "hi", "bn", "ta"
        split     : "train" | "validation" | "test"

    Returns:
        HuggingFace Dataset with columns: english_word, native_word
    """
    hf_config = LANGUAGE_CONFIGS[lang_code]
    zip_filename = f"{hf_config}.zip"
    logger.info(f"Downloading Aksharantar [{hf_config}] from HuggingFace Hub …")

    try:
        zip_path = hf_hub_download(
            repo_id="ai4bharat/Aksharantar",
            filename=zip_filename,
            repo_type="dataset",
        )
    except Exception as e:
        logger.error(f"Failed to download {zip_filename}: {e}")
        raise

    logger.info(f"Extracting {zip_path} …")
    # The zip contains JSON files named: {hf_config}_train.json, {hf_config}_valid.json, etc.
    split_key_map = {
        "train":      "train",
        "validation": "valid",
        "test":       "test",
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.info(f"Zip contents: {names[:10]}{'...' if len(names) > 10 else ''}")

        split_key = split_key_map.get(split, split)
        # Match files containing the split keyword (json or csv)
        target_file = None
        for ext in (".json", ".csv", ".tsv"):
            matches = [n for n in names if split_key in n.lower() and n.endswith(ext)]
            if matches:
                target_file = matches[0]
                break

        if target_file is None:
            raise FileNotFoundError(
                f"Could not find {split} split in {zip_filename}. Available files: {names}"
            )

        logger.info(f"Reading split file: {target_file}")
        with zf.open(target_file) as f:
            if target_file.endswith(".json"):
                df = pd.read_json(io.TextIOWrapper(f, encoding="utf-8"), lines=True)
            else:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))

    # Normalise column names — Aksharantar uses "english word" / "native word" with spaces
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("english word", "english_word"):
            col_map[col] = "english_word"
        elif lc in ("native word", "native_word"):
            col_map[col] = "native_word"
    if col_map:
        df = df.rename(columns=col_map)

    assert "english_word" in df.columns, \
        f"'english_word' column missing. Found: {df.columns.tolist()}"
    assert "native_word" in df.columns, \
        f"'native_word' column missing. Found: {df.columns.tolist()}"

    df = df[["english_word", "native_word"]].dropna()
    ds = Dataset.from_pandas(df, preserve_index=False)
    del df          # free the pandas DataFrame immediately
    gc.collect()
    logger.info(f"[{lang_code}] loaded {len(ds):,} raw samples from {split} split")
    return ds


def load_all_languages(split: str = "train") -> Dict[str, Dataset]:
    """Load all configured languages for the given split."""
    return {lang: load_aksharantar_language(lang, split) for lang in LANGUAGES}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def clean_pair(roman: str, native: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Basic cleanup for a (roman, native) word pair.
    Returns (None, None) if the pair should be discarded.
    """
    roman = roman.strip().lower()
    native = native.strip()

    # Drop empty or whitespace-only entries
    if not roman or not native:
        return None, None

    # Drop pairs where roman side contains non-ASCII (likely mis-labelled)
    if not roman.isascii():
        return None, None

    # Drop suspiciously long entries (likely noise)
    if len(roman) > 50 or len(native) > 50:
        return None, None

    return roman, native


def preprocess_dataset(ds: Dataset, lang_code: str, max_samples: Optional[int] = None) -> Dataset:
    """
    Clean + prefix dataset for one language.

    Adds the language token prefix to the source (roman) word so the model can
    condition on the target script, e.g.:
        "__hi__ kitab"  →  "किताब"
    """
    prefix = LANG_TOKEN[lang_code]

    def _process(batch):
        src, tgt = [], []
        for roman, native in zip(batch["english_word"], batch["native_word"]):
            r, n = clean_pair(roman, native)
            if r is not None:
                src.append(f"{prefix} {r}")
                tgt.append(n)
        return {"source": src, "target": tgt}

    cleaned = ds.map(
        _process,
        batched=True,
        remove_columns=ds.column_names,
        desc=f"Cleaning {lang_code}",
    )

    if max_samples and len(cleaned) > max_samples:
        indices = random.sample(range(len(cleaned)), max_samples)
        cleaned = cleaned.select(indices)

    logger.info(f"[{lang_code}] {len(cleaned):,} samples after cleaning")
    return cleaned


# ── Combined multilingual dataset ─────────────────────────────────────────────

def build_multilingual_dataset(save_dir: Optional[str] = None) -> DatasetDict:
    """
    Build a combined train / validation / test DatasetDict across all three
    languages.  Samples are shuffled so languages are interleaved during training.

    Args:
        save_dir : if given, saves the processed dataset to disk

    Returns:
        DatasetDict with keys "train", "validation", "test"
    """
    cfg = training_config
    splits_cfg = {
        "train":      cfg.train_samples_per_lang,
        "validation": cfg.val_samples_per_lang,
        "test":       cfg.test_samples_per_lang,
    }

    combined: Dict[str, List[Dataset]] = {s: [] for s in splits_cfg}

    # Load one language at a time to keep peak RAM low.
    for lang in LANGUAGES:
        logger.info(f"Processing language: {lang}")
        for split, max_n in splits_cfg.items():
            hf_split = "validation" if split == "validation" else split
            raw_ds = load_aksharantar_language(lang, hf_split)
            processed = preprocess_dataset(raw_ds, lang, max_samples=max_n)
            combined[split].append(processed)
            del raw_ds      # release raw download immediately
            gc.collect()

    result = DatasetDict({
        split: concatenate_datasets(parts).shuffle(seed=42)
        for split, parts in combined.items()
    })
    del combined
    gc.collect()

    logger.info("Combined dataset sizes:")
    for split, ds in result.items():
        logger.info(f"  {split}: {len(ds):,} samples")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        result.save_to_disk(save_dir)
        logger.info(f"Dataset saved to {save_dir}")

    return result


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenise_dataset(
    dataset: DatasetDict,
    tokeniser: PreTrainedTokenizer,
) -> DatasetDict:
    """
    Tokenise source and target sequences using the provided tokeniser.
    Returns a DatasetDict ready for Seq2SeqTrainer.
    """
    max_in  = model_config.max_input_length
    max_out = model_config.max_target_length

    def _tokenise(batch):
        model_inputs = tokeniser(
            batch["source"],
            max_length=max_in,
            padding="max_length",
            truncation=True,
        )
        labels = tokeniser(
            text_target=batch["target"],
            max_length=max_out,
            padding="max_length",
            truncation=True,
        )
        # Replace pad token id with -100 so loss ignores padding
        model_inputs["labels"] = [
            [(tok if tok != tokeniser.pad_token_id else -100) for tok in label_ids]
            for label_ids in labels["input_ids"]
        ]
        return model_inputs

    return dataset.map(
        _tokenise,
        batched=True,
        remove_columns=["source", "target"],
        desc="Tokenising",
    )


# ── Quick inspection helpers ───────────────────────────────────────────────────

def show_samples(dataset: DatasetDict, n: int = 5):
    """Print a few raw samples for sanity-checking."""
    for split in ["train", "validation"]:
        print(f"\n── {split} samples ──────────────")
        ds = dataset[split]
        for i in range(min(n, len(ds))):
            row = ds[i]
            print(f"  {row['source']:40s}  →  {row['target']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    random.seed(42)

    ds = build_multilingual_dataset(save_dir=paths.processed_data_dir)
    show_samples(ds)

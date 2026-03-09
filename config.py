"""
Central configuration for the multilingual transliteration model.
Languages: Hindi (hi), Bengali (bn), Tamil (ta)
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── Language settings ──────────────────────────────────────────────────────────
LANGUAGES = ["hi", "bn", "ta"]

LANGUAGE_NAMES = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
}

# Aksharantar HF dataset config keys
LANGUAGE_CONFIGS = {
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
}

# Language token prefixes injected at input start (MarianMT / mT5 convention)
LANG_TOKEN = {lang: f"__{lang}__" for lang in LANGUAGES}


# ── Paths ──────────────────────────────────────────────────────────────────────
@dataclass
class Paths:
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    ct2_model_dir: str = "ct2_model"
    logs_dir: str = "logs"
    results_dir: str = "results"


# ── Model architecture ─────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # We fine-tune t5-small as our seq2seq backbone.
    # Its character-level tokeniser is augmented with Indic Unicode ranges.
    base_model_name: str = "google/mt5-small"

    # Special tokens
    additional_special_tokens: List[str] = field(
        default_factory=lambda: [f"__{lang}__" for lang in LANGUAGES]
    )

    max_input_length: int = 64   # Roman transliteration inputs are short
    max_target_length: int = 64


# ── Training hyperparameters ───────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Data (per language)
    train_samples_per_lang: int = 50_000
    val_samples_per_lang: int = 5_000
    test_samples_per_lang: int = 2_000

    # Training
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4   # effective batch = 128
    fp16: bool = True                      # enabled for GPU (Colab T4)
    gradient_checkpointing: bool = True    # reduces VRAM at cost of ~20% speed

    # Logging / saving
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_cer"
    greater_is_better: bool = False

    # Decoding
    num_beams: int = 4
    early_stopping: bool = True

    seed: int = 42

    # ── Colab / Drive ─────────────────────────────────────────────────────────
    drive_save_dir: str = "/content/drive/MyDrive/transliteration_model"


# ── CTranslate2 optimization ───────────────────────────────────────────────────
@dataclass
class CT2Config:
    quantization: str = "int8"          # options: int8, int8_float16, float16, float32
    inter_threads: int = 4              # parallel request threads
    intra_threads: int = 2              # threads per request
    compute_type: str = "int8"
    benchmark_iterations: int = 500
    benchmark_batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])


# ── Singleton instances ────────────────────────────────────────────────────────
paths = Paths()
model_config = ModelConfig()
training_config = TrainingConfig()
ct2_config = CT2Config()

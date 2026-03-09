"""
Model optimization: convert fine-tuned mT5 → CTranslate2 format
and benchmark inference speed vs quality.

Usage:
    python convert_to_ct2.py
    python convert_to_ct2.py --quantization int8
    python convert_to_ct2.py --quantization float16   # GPU only
    python convert_to_ct2.py --skip_convert           # only benchmark

Pipeline:
    1. Load fine-tuned HuggingFace model from checkpoints/best/
    2. Convert to CTranslate2 format (INT8 quantization)
    3. Benchmark latency and throughput (batch 1 / 8 / 32)
    4. Verify quality (CER must not regress > 1 pp vs original)
    5. Save benchmark report to results/benchmark.json
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import paths, ct2_config, model_config, LANGUAGES, LANG_TOKEN
from evaluate import compute_cer, compute_exact_accuracy

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_model_to_ct2(
    hf_model_dir: str,
    output_dir: str,
    quantization: str = "int8",
) -> None:
    """
    Convert HuggingFace mT5 checkpoint to CTranslate2 format.
    Equivalent to running:
        ct2-opus-mt-converter --model <dir> --output <dir> --quantization int8
    but done programmatically via ctranslate2.converters.
    """
    try:
        import ctranslate2
    except ImportError:
        raise ImportError("Install ctranslate2:  pip install ctranslate2")

    logger.info(f"Converting {hf_model_dir} → CTranslate2 (quantization={quantization})")

    converter = ctranslate2.converters.OpusMTConverter(hf_model_dir)
    converter.convert(
        output_dir,
        quantization=quantization,
        force=True,
    )
    logger.info(f"Conversion complete → {output_dir}")


def convert_mt5_to_ct2(
    hf_model_dir: str,
    output_dir: str,
    quantization: str = "int8",
) -> None:
    """
    mT5 uses the T5 architecture.  CTranslate2 provides a dedicated converter.
    """
    try:
        import ctranslate2
    except ImportError:
        raise ImportError("Install ctranslate2:  pip install ctranslate2")

    logger.info(f"Converting mT5 {hf_model_dir} → CTranslate2 (quantization={quantization})")

    # For mT5-based models, use the generic HuggingFace converter
    # ct2-transformers-converter --model <dir> --output_dir <dir> --quantization int8 --force
    import subprocess, shutil
    cmd = [
        "ct2-transformers-converter",
        "--model",        hf_model_dir,
        "--output_dir",   output_dir,
        "--quantization", quantization,
        "--force",
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Conversion failed:\n{result.stderr}")
    logger.info(f"Conversion complete → {output_dir}")
    logger.info(result.stdout)


# ── Inference wrappers ────────────────────────────────────────────────────────

class HFInferenceEngine:
    """Wrapper around HuggingFace model for benchmarking."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        logger.info(f"Loading HF model from {model_dir}")
        self.tokeniser = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.device = device

    def transliterate(self, texts: List[str], num_beams: int = 4) -> List[str]:
        inputs = self.tokeniser(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_config.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=model_config.max_target_length,
            )
        return self.tokeniser.batch_decode(outputs, skip_special_tokens=True)


class CT2InferenceEngine:
    """Wrapper around CTranslate2 model for benchmarking."""

    def __init__(self, ct2_model_dir: str, tokeniser_dir: str, device: str = "cpu"):
        import ctranslate2
        logger.info(f"Loading CT2 model from {ct2_model_dir}")
        self.translator = ctranslate2.Translator(
            ct2_model_dir,
            device=device,
            inter_threads=ct2_config.inter_threads,
            intra_threads=ct2_config.intra_threads,
            compute_type=ct2_config.compute_type,
        )
        self.tokeniser = AutoTokenizer.from_pretrained(tokeniser_dir)

    def transliterate(self, texts: List[str], num_beams: int = 4) -> List[str]:
        # Tokenise to token strings (CT2 expects List[List[str]])
        tokenised = self.tokeniser(
            texts, truncation=True, max_length=model_config.max_input_length
        )
        # Convert IDs → tokens
        token_seqs = [
            self.tokeniser.convert_ids_to_tokens(ids)
            for ids in tokenised["input_ids"]
        ]

        results = self.translator.translate_batch(
            token_seqs,
            beam_size=num_beams,
            max_decoding_length=model_config.max_target_length,
        )

        outputs = []
        for r in results:
            tokens = r.hypotheses[0]
            ids    = self.tokeniser.convert_tokens_to_ids(tokens)
            text   = self.tokeniser.decode(ids, skip_special_tokens=True)
            outputs.append(text)
        return outputs


# ── Benchmarking ──────────────────────────────────────────────────────────────

def benchmark_engine(
    engine,
    test_samples: List[Tuple[str, str]],   # (source, reference)
    batch_sizes: List[int],
    n_iters: int = 100,
    num_beams: int = 4,
) -> Dict:
    """
    Measure latency and throughput for a range of batch sizes.
    Also computes CER on the first 1000 samples.
    """
    results = {}

    # Quality check on first min(1000, all) samples
    q_sources = [s for s, _ in test_samples[:1000]]
    q_refs    = [r for _, r in test_samples[:1000]]
    q_preds   = engine.transliterate(q_sources)
    cer = compute_cer(q_preds, q_refs)
    acc = compute_exact_accuracy(q_preds, q_refs)
    results["quality"] = {"cer": round(cer, 4), "accuracy": round(acc, 4)}
    logger.info(f"Quality → CER: {cer:.4f}  Accuracy: {acc:.4f}")

    # Speed benchmarks
    results["speed"] = {}
    for bs in batch_sizes:
        batch = [s for s, _ in test_samples[:bs]]
        # Warm-up
        for _ in range(3):
            engine.transliterate(batch, num_beams=num_beams)

        latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            engine.transliterate(batch, num_beams=num_beams)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        avg_ms    = float(np.mean(latencies))
        p50_ms    = float(np.percentile(latencies, 50))
        p95_ms    = float(np.percentile(latencies, 95))
        throughput = bs / (avg_ms / 1000)         # samples/second

        results["speed"][f"batch_{bs}"] = {
            "avg_latency_ms":  round(avg_ms, 2),
            "p50_latency_ms":  round(p50_ms, 2),
            "p95_latency_ms":  round(p95_ms, 2),
            "throughput_sps":  round(throughput, 1),
        }
        logger.info(
            f"  batch={bs:3d} | avg={avg_ms:.1f}ms "
            f"| p95={p95_ms:.1f}ms | {throughput:.0f} samples/s"
        )

    return results


def compute_model_size(model_dir: str) -> Dict[str, float]:
    """Walk the directory and sum file sizes (MB)."""
    total = sum(p.stat().st_size for p in Path(model_dir).rglob("*") if p.is_file())
    return {"total_mb": round(total / 1_048_576, 1)}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_dir",  default=f"{paths.checkpoint_dir}/best")
    p.add_argument("--ct2_model_dir", default=paths.ct2_model_dir)
    p.add_argument("--quantization",  default=ct2_config.quantization)
    p.add_argument("--skip_convert",  action="store_true")
    p.add_argument("--data_dir",      default=f"{paths.processed_data_dir}/tokenised")
    p.add_argument("--num_beams",     type=int, default=4)
    p.add_argument("--n_iters",       type=int, default=ct2_config.benchmark_iterations)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Convert
    if not args.skip_convert:
        convert_mt5_to_ct2(args.hf_model_dir, args.ct2_model_dir, args.quantization)

    # 2. Load test samples (raw text, pre-tokenisation)
    logger.info("Loading test data for benchmarking …")
    raw_ds_path = Path(paths.processed_data_dir)
    from datasets import load_from_disk as lfd
    raw_ds = lfd(str(raw_ds_path))  # has "source" and "target" columns
    test_samples = [
        (row["source"], row["target"])
        for row in raw_ds["test"].select(range(min(2000, len(raw_ds["test"]))))
    ]

    # 3. Benchmark HF model
    logger.info("=" * 60)
    logger.info("Benchmarking HuggingFace model …")
    hf_engine = HFInferenceEngine(args.hf_model_dir, device=device)
    hf_results = benchmark_engine(
        hf_engine, test_samples,
        batch_sizes=ct2_config.benchmark_batch_sizes,
        n_iters=args.n_iters,
        num_beams=args.num_beams,
    )
    hf_size = compute_model_size(args.hf_model_dir)

    # 4. Benchmark CT2 model
    logger.info("=" * 60)
    logger.info("Benchmarking CTranslate2 model …")
    ct2_engine = CT2InferenceEngine(args.ct2_model_dir, args.hf_model_dir, device=device)
    ct2_results = benchmark_engine(
        ct2_engine, test_samples,
        batch_sizes=ct2_config.benchmark_batch_sizes,
        n_iters=args.n_iters,
        num_beams=args.num_beams,
    )
    ct2_size = compute_model_size(args.ct2_model_dir)

    # 5. Compute speedup ratios
    speedups = {}
    for bs_key in hf_results["speed"]:
        hf_tput = hf_results["speed"][bs_key]["throughput_sps"]
        ct2_tput = ct2_results["speed"].get(bs_key, {}).get("throughput_sps", 0)
        if hf_tput and ct2_tput:
            speedups[bs_key] = round(ct2_tput / hf_tput, 2)

    cer_delta = ct2_results["quality"]["cer"] - hf_results["quality"]["cer"]

    report = {
        "model_sizes": {
            "hf_mb":        hf_size["total_mb"],
            "ct2_mb":       ct2_size["total_mb"],
            "size_reduction_pct": round(
                (1 - ct2_size["total_mb"] / hf_size["total_mb"]) * 100, 1
            ),
        },
        "quality": {
            "hf_cer":           hf_results["quality"]["cer"],
            "ct2_cer":          ct2_results["quality"]["cer"],
            "cer_delta":        round(cer_delta, 4),
            "hf_accuracy":      hf_results["quality"]["accuracy"],
            "ct2_accuracy":     ct2_results["quality"]["accuracy"],
        },
        "speedup_ratios": speedups,
        "hf_speed":  hf_results["speed"],
        "ct2_speed": ct2_results["speed"],
        "quantization": args.quantization,
    }

    # 6. Save report
    results_dir = Path(paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "benchmark.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"  Model size   HF: {hf_size['total_mb']} MB  →  CT2: {ct2_size['total_mb']} MB "
                f"({report['model_sizes']['size_reduction_pct']}% reduction)")
    logger.info(f"  CER delta    {cer_delta:+.4f}  (< 0.01 = acceptable quality)")
    logger.info(f"  Speedup (batch=1): {speedups.get('batch_1', 'N/A')}×")
    logger.info(f"  Report saved → {report_path}")


if __name__ == "__main__":
    main()

"""
Gradio demo for the multilingual transliteration model.
Deployed to HuggingFace Spaces.

Supports:
  - Single word transliteration for all 3 languages simultaneously
  - Batch text transliteration
  - Live CER display against a user-supplied reference (optional)
  - Language-specific example sets

Set HF_MODEL_ID env var to your HuggingFace model repo, e.g.:
    HF_MODEL_ID=your-username/multilingual-transliteration
"""

import os
import re
import logging
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Try CTranslate2 first for faster inference on Spaces CPU tier
try:
    import ctranslate2
    CT2_AVAILABLE = True
except ImportError:
    CT2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

HF_MODEL_ID   = os.getenv("HF_MODEL_ID", "avinyaa/multilingual-transliteration-mt51")
HF_CT2_ID     = os.getenv("HF_CT2_ID",   "avinyaa/multilingual-transliteration-ct2")
CT2_MODEL_DIR = os.getenv("CT2_MODEL_DIR", "./ct2_model")   # local path inside Space


def _ensure_ct2_model():
    """Download CT2 model from Hub if not already present locally."""
    local = Path(CT2_MODEL_DIR)
    if local.exists() and any(local.iterdir()):
        return
    if not CT2_AVAILABLE:
        return
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading CT2 model from {HF_CT2_ID} …")
        snapshot_download(repo_id=HF_CT2_ID, local_dir=CT2_MODEL_DIR)
        logger.info(f"CT2 model downloaded to {CT2_MODEL_DIR}")
    except Exception as e:
        logger.warning(f"CT2 model download failed ({e}). Will use HF model.")

LANG_CONFIG = {
    "Hindi (hi)":   {"token": "__hi__", "script": "Devanagari"},
    "Bengali (bn)": {"token": "__bn__", "script": "Bengali"},
    "Tamil (ta)":   {"token": "__ta__", "script": "Tamil"},
}

EXAMPLES = {
    "Hindi (hi)":   ["namaste", "kitab", "dilli", "pyar", "raat", "ghar", "paani", "khana"],
    "Bengali (bn)": ["dhanyabad", "kolkata", "amra", "shundor", "bhalobasa", "raat", "bhai"],
    "Tamil (ta)":   ["vanakkam", "nandri", "illam", "mazhai", "vidu", "kathal", "sollu"],
}

MAX_BEAMS  = 4
MAX_LENGTH = 64


# ── Model loading ─────────────────────────────────────────────────────────────

class TransliterationModel:
    def __init__(self):
        self.tokeniser   = None
        self.hf_model    = None
        self.ct2_model   = None
        self.use_ct2     = False
        self._load()

    def _load(self):
        _ensure_ct2_model()
        logger.info(f"Loading tokeniser from {HF_MODEL_ID} …")
        self.tokeniser = AutoTokenizer.from_pretrained(HF_MODEL_ID)

        # Prefer CT2 for faster CPU inference on HF Spaces free tier
        if CT2_AVAILABLE and os.path.isdir(CT2_MODEL_DIR):
            try:
                logger.info(f"Loading CTranslate2 model from {CT2_MODEL_DIR}")
                self.ct2_model = ctranslate2.Translator(
                    CT2_MODEL_DIR,
                    device="cpu",
                    inter_threads=2,
                    intra_threads=4,
                    compute_type="int8",
                )
                self.use_ct2 = True
                logger.info("✓ Using CTranslate2 for inference")
            except Exception as e:
                logger.warning(f"CT2 load failed ({e}), falling back to HF model")

        if not self.use_ct2:
            logger.info(f"Loading HuggingFace model from {HF_MODEL_ID}")
            self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID)
            self.hf_model.eval()
            logger.info("✓ Using HuggingFace model for inference")

    def transliterate(self, texts: List[str]) -> List[str]:
        if self.use_ct2:
            return self._ct2_infer(texts)
        return self._hf_infer(texts)

    def _hf_infer(self, texts: List[str]) -> List[str]:
        inputs = self.tokeniser(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                num_beams=MAX_BEAMS,
                max_length=MAX_LENGTH,
            )
        return self.tokeniser.batch_decode(outputs, skip_special_tokens=True)

    def _ct2_infer(self, texts: List[str]) -> List[str]:
        tokenised = self.tokeniser(
            texts, truncation=True, max_length=MAX_LENGTH
        )
        token_seqs = [
            self.tokeniser.convert_ids_to_tokens(ids)
            for ids in tokenised["input_ids"]
        ]
        results = self.ct2_model.translate_batch(
            token_seqs,
            beam_size=MAX_BEAMS,
            max_decoding_length=MAX_LENGTH,
        )
        outputs = []
        for r in results:
            tokens = r.hypotheses[0]
            ids    = self.tokeniser.convert_tokens_to_ids(tokens)
            text   = self.tokeniser.decode(ids, skip_special_tokens=True)
            outputs.append(text)
        return outputs


# Lazy global model instance
_model = None

def get_model() -> TransliterationModel:
    global _model
    if _model is None:
        _model = TransliterationModel()
    return _model


# ── Core transliteration logic ────────────────────────────────────────────────

def transliterate_single(roman_word: str) -> Tuple[str, str, str]:
    """
    Transliterate one Roman word into all three languages simultaneously.
    Returns (hindi_output, bengali_output, tamil_output).
    """
    roman_word = roman_word.strip().lower()
    if not roman_word:
        return "", "", ""

    model = get_model()
    inputs = [
        f"{LANG_CONFIG['Hindi (hi)']['token']} {roman_word}",
        f"{LANG_CONFIG['Bengali (bn)']['token']} {roman_word}",
        f"{LANG_CONFIG['Tamil (ta)']['token']} {roman_word}",
    ]
    outputs = model.transliterate(inputs)
    return outputs[0], outputs[1], outputs[2]


def transliterate_batch(
    text: str,
    target_language: str,
) -> str:
    """
    Transliterate a multi-word / multi-line input for a single selected language.
    Words are transliterated individually and reconstructed.
    """
    if not text.strip():
        return ""

    lang_token = LANG_CONFIG[target_language]["token"]
    model      = get_model()

    # Split into tokens preserving whitespace structure
    words  = re.findall(r'\S+', text)
    inputs = [f"{lang_token} {w.lower()}" for w in words]
    preds  = model.transliterate(inputs)

    # Rebuild with original spacing
    result = text
    for orig, pred in zip(words, preds):
        result = result.replace(orig, pred, 1)
    return result


def compute_cer_display(prediction: str, reference: str) -> str:
    """Compute CER between user-supplied reference and prediction."""
    if not prediction or not reference:
        return "—"
    try:
        from jiwer import cer
        score = cer(reference.strip(), prediction.strip())
        return f"{score:.4f}"
    except Exception:
        return "—"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    model_type = "CTranslate2 (INT8 quantized)" if CT2_AVAILABLE else "HuggingFace mT5-small"

    with gr.Blocks(
        title="Multilingual Transliteration",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 20px 0; }
        .badge  { display:inline-block; background:#6366f1; color:#fff;
                  padding:2px 10px; border-radius:9999px; font-size:0.8rem; }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="header">
          <h1>🌐 Multilingual Transliteration</h1>
          <p>Roman → Indic script for <strong>Hindi</strong>, <strong>Bengali</strong>, and <strong>Tamil</strong></p>
          <span class="badge">Model: mT5-small fine-tuned on Aksharantar</span>&nbsp;
          <span class="badge">Engine: {model_type}</span>
        </div>
        """)

        # ── Tab 1: Single-word (all 3 languages at once) ─────────────────────
        with gr.Tab("🔤 Single Word → All Languages"):
            gr.Markdown(
                "Type a **Romanised word** and instantly see its transliteration "
                "in all three Indic scripts side-by-side."
            )

            with gr.Row():
                single_input = gr.Textbox(
                    label="Roman Input",
                    placeholder="e.g.  namaste  /  dhanyabad  /  vanakkam",
                    lines=1,
                    scale=3,
                )
                single_btn = gr.Button("Transliterate ➜", variant="primary", scale=1)

            with gr.Row():
                hi_out = gr.Textbox(label="🇮🇳 Hindi (Devanagari)",  interactive=False)
                bn_out = gr.Textbox(label="🇧🇩 Bengali",             interactive=False)
                ta_out = gr.Textbox(label="🇮🇳 Tamil",               interactive=False)

            # Example buttons
            gr.Markdown("**Quick examples:**")
            with gr.Row():
                for word in ["namaste", "pyar", "khana", "raat", "dhanyabad", "vanakkam"]:
                    gr.Button(word, size="sm").click(
                        lambda w=word: w,
                        outputs=single_input,
                    )

            single_btn.click(
                transliterate_single,
                inputs=single_input,
                outputs=[hi_out, bn_out, ta_out],
            )
            single_input.submit(
                transliterate_single,
                inputs=single_input,
                outputs=[hi_out, bn_out, ta_out],
            )

        # ── Tab 2: Batch / sentence mode ─────────────────────────────────────
        with gr.Tab("📄 Batch / Sentence Mode"):
            gr.Markdown(
                "Enter multiple words or a sentence and choose the **target language**."
            )

            batch_lang = gr.Dropdown(
                label="Target Language",
                choices=list(LANG_CONFIG.keys()),
                value="Hindi (hi)",
            )

            with gr.Row():
                batch_input  = gr.Textbox(label="Roman Input",  lines=5,
                                          placeholder="namaste dilli\npyar ghar paani")
                batch_output = gr.Textbox(label="Transliterated Output", lines=5,
                                          interactive=False)

            with gr.Row():
                batch_btn = gr.Button("Transliterate ➜", variant="primary")

            # Optional CER evaluation
            with gr.Accordion("📊 Evaluate against reference (optional)", open=False):
                ref_input  = gr.Textbox(label="Reference (ground truth)",
                                        placeholder="नमस्ते दिल्ली\nप्यार घर पानी",
                                        lines=3)
                cer_output = gr.Textbox(label="CER Score", interactive=False)
                eval_btn   = gr.Button("Compute CER")
                eval_btn.click(
                    compute_cer_display,
                    inputs=[batch_output, ref_input],
                    outputs=cer_output,
                )

            batch_btn.click(
                transliterate_batch,
                inputs=[batch_input, batch_lang],
                outputs=batch_output,
            )

        # ── Tab 3: Language examples ──────────────────────────────────────────
        with gr.Tab("💡 Examples per Language"):
            for lang_name, words in EXAMPLES.items():
                with gr.Accordion(lang_name, open=lang_name == "Hindi (hi)"):
                    token = LANG_CONFIG[lang_name]["token"]
                    model_ref = get_model()
                    inputs = [f"{token} {w}" for w in words]
                    preds  = model_ref.transliterate(inputs)

                    table_html = "<table style='width:100%;border-collapse:collapse'>"
                    table_html += "<tr><th>Roman</th><th>Transliterated</th></tr>"
                    for word, pred in zip(words, preds):
                        table_html += f"<tr><td>{word}</td><td style='font-size:1.3em'>{pred}</td></tr>"
                    table_html += "</table>"
                    gr.HTML(table_html)

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown("""
---
**Model:** mT5-small fine-tuned on [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)
&nbsp;|&nbsp; **Optimization:** CTranslate2 INT8 quantization
&nbsp;|&nbsp; **Languages:** Hindi · Bengali · Tamil
        """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )

---
title: Multilingual Transliteration
emoji: 🌐
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
license: mit
---

# Multilingual Transliteration

Roman → Indic script transliteration for **Hindi**, **Bengali**, and **Tamil**.

- **Model:** mT5-small fine-tuned on [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)
- **Optimization:** CTranslate2 INT8 quantization (~4× faster, 74% smaller)
- **Input:** Romanised text (e.g. `namaste`)
- **Output:** Native Indic script (e.g. `नमस्ते`, `নমস্তে`, `நமஸ்தே`)

---
title: SlideBank
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
python_version: "3.10"
app_file: space_app.py
pinned: false
preload_from_hub:
  - Qwen/Qwen2.5-VL-3B-Instruct
---

# SlideBank

Explore the prebuilt whole-slide evidence bank and ask questions using
Qwen2.5-VL-3B-Instruct on ZeroGPU. The question policy, agent router, and compact
evidence retrieval use the shared SlideBank workbench implementation.

The hosted demonstration model differs from the paper's pathology backbones.
Evidence browsing also works on CPU; live QA requires ZeroGPU allocation and
is subject to Hugging Face's queue and usage quota. No visitor API key is needed.

[Paper](https://arxiv.org/abs/2609.00342) ·
[Code](https://github.com/gexinh/SlideBank) ·
[Project page](https://gexinh.github.io/SlideBank/)

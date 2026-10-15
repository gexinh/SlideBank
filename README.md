<h1 align="center"><img src="assets/slidebank-icon.svg" width="42" height="42" align="absmiddle" alt=""> SlideBank</h1>

<p align="center"><strong>A Persistent Hierarchical Evidence Bank for Consistent Whole-Slide Reasoning</strong></p>

<p align="center">
  Beidi Zhao<sup>*</sup> · Gexin Huang<sup>*</sup> · Ciro Zhang<sup>*</sup> · Anqi Li · Yusheng Tan · Chen Zhou · Gang Wang · Zu-hua Gao · Xiaoxiao Li
</p>

<p align="center"><sup>*</sup> Equal contribution.</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.00342"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&amp;logoColor=white" alt="Read the paper on arXiv"></a>
  <a href="https://gexinh.github.io/SlideBank/"><img src="https://img.shields.io/badge/Project%20Page-Visit-256D5B?logo=googlechrome&amp;logoColor=white" alt="Visit the project page"></a>
  <a href="#interactive-workbench"><img src="https://img.shields.io/badge/Workbench-Gradio-F97316?logo=gradio&amp;logoColor=white" alt="Open the workbench guide"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-4051B5?logo=googlescholar&amp;logoColor=white" alt="View the BibTeX citation"></a>
  <a href="https://github.com/gexinh/SlideBank/actions/workflows/ci.yml"><img src="https://github.com/gexinh/SlideBank/actions/workflows/ci.yml/badge.svg" alt="Continuous integration status"></a>
</p>

SlideBank separates whole-slide exploration from question answering. It builds a persistent evidence bank with global summaries, regional observations, and patch views, linked through pathology concepts and slide coordinates. The bank can then be reused to retrieve relevant evidence for successive questions.

<p align="center">
  <a href="assets/main-figure.pdf"><img src="assets/main-figure.png" width="100%" alt="SlideBank paper main figure: hierarchical evidence-bank construction, evidence-augmented reasoning, and persistent reuse across queries"></a>
</p>

<!-- main-figure-caption:start -->
*Main figure from the SlideBank paper. [Original PDF](assets/main-figure.pdf).*
<!-- main-figure-caption:end -->

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ui,wsi]"
```

The `wsi` extra includes OpenSlide's Python bindings and bundled library; if your platform needs a system installation, follow the [OpenSlide installation guide](https://openslide.org/download/). Install `.[local]` for local model inference or `.[gemini]` for the Gemini API. Quilt-LLaVA additionally needs its external runtime repository and model weights; use `configs/build_quilt.yaml` and `configs/infer_quilt.yaml` after setting `SLIDEBANK_QUILT_LLAVA_REPO`.

Model weights, datasets, and precomputed banks are obtained separately.

## Build, retrieve, and answer

Edit [configs/build.yaml](configs/build.yaml) and [configs/infer.yaml](configs/infer.yaml) with your slide, bank, artifact, endpoint, and model settings. Their API model name is a placeholder to replace before running. CSV inputs use `Slide`, `Question`, `A`, `B`, `C`, and `D`; optional `Answer` or `GT` columns provide evaluation labels, and `Image` can identify a thumbnail. The templates use the recursive slide search pattern `**/*.svs`; the CLI fallback is `*/*.svs`. See [the data guide](docs/DATA.md) for the matching and bank schemas.

```bash
# Build a bank for one slide before processing the remaining collection.
python scripts/build_bank.py --config configs/build.yaml --max-slides 1

# Run one MCQ through the command-line inference pipeline.
python scripts/infer.py --config configs/infer.yaml --max-samples 1

# Export a bank for the evidence browser.
python scripts/export_ui_tree.py --bank-dir /path/to/bank \
  --out-dir outputs/ui_tree
```

Bank construction writes persistent records and visual artifacts. Inference writes JSONL predictions and a summary to the paths in the inference config. API backends accept credentials through environment variables; set the corresponding `api_key_env` field in the config.

Use `--help` on each script to inspect runtime overrides. Supported inference backends include Quilt-LLaVA, local Hugging Face models, and Gemini/OpenAI/OpenAI-compatible APIs; model compatibility depends on the selected runtime.

## Interactive workbench

```bash
python app.py
```

Open **http://127.0.0.1:7864**, configure a model, and load an exported evidence bank to browse global, regional, and patch views or ask a question. For a remote server, forward port 7864 through SSH or your editor. See the [workbench guide](docs/WORKBENCH.md) for runtime and data configuration.

Choose an entry point for interactive questions or batch evaluation:

| Entry point | Retrieval and answer flow |
| --- | --- |
| `app.py` | Question policy, agent routing with rule fallback, compact evidence retrieval, and a final VLM answer. Supports free-form and multiple-choice questions. |
| `scripts/infer.py` | Batch MCQ inference using the tool-context retrieval pipeline; `single` and `multi` modes control whether previous answers enter the prompt. |

## Project page

The [project page](https://gexinh.github.io/SlideBank/) includes an evidence viewer for exported JSON and local images. To run the site locally:

```bash
python -m http.server --directory project 8080
```

Open **http://127.0.0.1:8080**. Model inference runs through the Python workbench. See the [hosting guide](docs/PROJECT_PAGE.md) to connect a hosted workbench to the page.

## Repository layout

```text
app.py          Interactive workbench
slidebank/      Bank construction, retrieval, VLM adapters, and UI support
scripts/        Build, inference, UI-tree export, and figure preparation
configs/        Portable configuration templates
project/        Static project page
assets/         Original paper main figure and web preview
docs/           Data, workbench, hosting, and source guides
```

## Citation

If you use SlideBank in your research, please cite the paper:

```bibtex
@article{zhao2026slidebank,
  title   = {SlideBank: A Persistent Hierarchical Evidence Bank for Consistent Whole-Slide Reasoning},
  author  = {Zhao, Beidi and Huang, Gexin and Zhang, Ciro and Li, Anqi and Tan, Yusheng and Zhou, Chen and Wang, Gang and Gao, Zu-hua and Li, Xiaoxiao},
  journal = {arXiv preprint arXiv:2609.00342},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.00342},
  url     = {https://arxiv.org/abs/2609.00342}
}
```

## Acknowledgments and licensing

SlideBank builds on WSIChat's WSI processing and evidence-bank components. Original SlideBank contributions are available under the [MIT License](LICENSE); see [source attribution and terms](docs/THIRD_PARTY.md) for inherited components and external assets.

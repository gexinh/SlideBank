# Source provenance and third-party materials

## Source provenance

SlideBank includes WSI processing and evidence-bank components from WSIChat's `agent_quilt_llava/wsi_agent` code. Related inference components originate from `infer_vlm_see_patch`. The interactive workbench adapts WSIChat's question policy, routing, and compact-context retrieval.

Inherited components:

- `slidebank/wsi_agent/__init__.py`
- `slidebank/wsi_agent/loader.py`
- `slidebank/wsi_agent/memory_bank.py`
- `slidebank/wsi_agent/signal_ontology.json`
- `slidebank/wsi_agent/tumor_size_tool.py`
- `slidebank/wsi_agent/vector_store.py`

No source-code license was included with the WSIChat components listed above. Their reuse terms are unspecified.

Original SlideBank contributions use the [MIT License](../LICENSE). That license does not relicense inherited WSIChat components or cover external runtimes, models, datasets, or paper assets. The WSIChat repository is not required to run the packaged backend.

## Runtime dependencies and models

Dependencies are installed separately and remain subject to their own licenses. Relevant projects include [OpenSlide](https://openslide.org/), [Gradio](https://github.com/gradio-app/gradio), [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [Pillow](https://github.com/python-pillow/Pillow), and [OpenCV](https://github.com/opencv/opencv). Consult the exact installed distributions for their notices and terms.

Local model runtimes and weights, including Quilt-LLaVA, are obtained separately under their respective terms. Hosted inference follows the selected provider's service terms.

## Paper, figures, and data

The [arXiv paper](https://arxiv.org/abs/2609.00342) is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its main figure is available as [`assets/main-figure.pdf`](../assets/main-figure.pdf), with a PNG rendering for the README and project page. Paper and figure terms are separate from the source-code license.

The [included demonstration bank](../demos/README.md) contains derived images and precomputed observations for one open-access TCGA-BRCA slide. Credit the TCGA Research Network and follow the linked GDC data terms. Additional WSIs, annotations, and banks are obtained separately under their source terms.

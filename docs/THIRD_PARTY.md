# Source provenance and third-party materials

## WSIChat development code

SlideBank's clean backend was assembled from the authors' WSIChat development code. Bank construction and low-level WSI processing originate from `agent_quilt_llava/wsi_agent`; related inference development used `infer_vlm_see_patch`. The V5 interactive flow adapts the question policy, routing, and compact-context retrieval documented for the earlier `agent_quilt_llava` inference runbook.

The inherited source paths in this release are:

- `slidebank/wsi_agent/__init__.py`
- `slidebank/wsi_agent/loader.py`
- `slidebank/wsi_agent/memory_bank.py`
- `slidebank/wsi_agent/signal_ontology.json`
- `slidebank/wsi_agent/tumor_size_tool.py`
- `slidebank/wsi_agent/vector_store.py`

The inspected WSIChat checkout and the original clean SlideBank directory do not contain a top-level source-code license. No separate license or copyright notice was found in the relevant WSIChat source subdirectories. These observations do not establish a reusable upstream license. The inherited components' licensing remains to be confirmed with their original authors; this preparation does not relicense them.

Original SlideBank contributions are released under the [MIT License](../LICENSE), as selected by the repository author. That grant does not override the terms or unresolved license status of inherited components, external runtimes, models, datasets, or paper assets.

The release omits historical experiment branches, old UI entry points, downloaded runtimes, model checkpoints, generated banks, and benchmark images. The upstream WSIChat repository is not required for ordinary use of the packaged backend.

## Runtime dependencies and models

Dependencies are installed separately and remain subject to their own licenses. Relevant projects include [OpenSlide](https://openslide.org/), [Gradio](https://github.com/gradio-app/gradio), [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [Pillow](https://github.com/python-pillow/Pillow), and [OpenCV](https://github.com/opencv/opencv). Consult the exact installed distributions for their notices and terms.

Quilt-LLaVA, other local model runtimes, and model weights are external assets. Their availability does not grant a license to redistribute their weights or training data. Configure a separately obtained runtime/checkpoint when using a local backend. Hosted inference follows the selected provider's service terms.

## Paper, figures, and data

The [arXiv paper](https://arxiv.org/abs/2609.00342) is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as linked by its arXiv record. That license applies to the paper and does not establish a license for this source code. The author-supplied main figure is preserved at `assets/main-figure.pdf`; its PNG rendering is used in the README and project page. These figure assets retain the paper's applicable terms.

Benchmark WSIs, annotations, model-generated records, and private local banks are not bundled. Users obtain data from the original sources and follow the applicable dataset terms. Before adding public demonstration images or bank exports, record their source and redistribution terms alongside those assets.

## Release item still to resolve

Confirm the license for the inherited WSIChat components with their original authors before public distribution. Preserve any applicable upstream notices when adding external source or assets.

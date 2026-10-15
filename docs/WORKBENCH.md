# SlideBank Workbench

The workbench combines the latest conversation interface with a hierarchical
evidence browser: global slide overview, selected grid cell, and anchor patch
views. Browsing an existing evidence tree requires no model call. Question
answering and rebuilding a bank use the model you configure.

## Start locally

Install the package and UI dependencies from the repository root:

```bash
python -m pip install -e '.[ui]'
python app.py
```

Open <http://127.0.0.1:7864>. The supported UI dependency is Gradio 5.x; install
through the package extra to get the declared version range. A fresh checkout
opens an empty workbench and explains how to load your own evidence bank.

## Load your evidence

First build a bank using your WSI files and CSV input as described in the root
README. Then export the visualization tree:

```bash
python scripts/export_ui_tree.py --bank-dir outputs/memory_bank
SLIDEBANK_UI_TREE_DIR=outputs/memory_bank/ui_tree python app.py
```

Select a **Configured Slide** and press **Load Evidence**. Click the overview
image or select a grid cell to inspect its anchors and patches. Exported tree
files use the `slidebank-ui-tree-v1` schema. Image paths may be absolute paths
accessible on the server or paths relative to the tree JSON file.

For a single tree, select **Upload UI-tree JSON**, choose the exported JSON, and
press **Load Evidence**. Uploading JSON transfers the tree only. Its images must
already be accessible on the server; use absolute image paths for uploaded trees
because Gradio stores the JSON in a temporary upload directory. Use configured
trees for bundles with relative image paths.

Raw WSI inputs go through `scripts/build_bank.py`. The workbench upload control
accepts exported JSON, and **Rebuild Configured Slide** rebuilds the selected
slide from the local CSV and WSI locations in `configs/build.yaml`. Rebuilding
requires the selected slide to match a CSV row and an accessible WSI file.

## Ask a question

QA requires the original memory bank as well as the visualization tree. Trees
contain visualization metadata; they do not replace the bank's retrieval data.
The workbench resolves bank locations from explicit environment overrides,
embedded `_agent` paths, the thumbnail's `artifacts/` directory, or
`configs/infer.yaml`.

```bash
SLIDEBANK_UI_TREE_DIR=outputs/memory_bank/ui_tree \
SLIDEBANK_BANK_DIR=outputs/memory_bank \
SLIDEBANK_ARTIFACT_DIR=outputs/memory_bank/artifacts \
python app.py
```

Choose **Local Model** for Quilt-LLaVA or Qwen2.5-VL. Install the corresponding
model dependencies and download the weights into the local Hugging Face cache
or configure a local snapshot path before sending a question. The Hugging Face
loader uses cached/local weights (`local_files_only=True`).
Quilt-LLaVA also requires its external code repository; set
`SLIDEBANK_QUILT_LLAVA_REPO` or the corresponding configuration field.

For **API Model**, choose Google, OpenAI, or an OpenAI-compatible custom endpoint,
then enter its model ID and API key. A custom endpoint also requires a base URL.
The UI validates missing credentials and model IDs before inference. API keys
remain in the running server's session/model-client memory; use a local or
trusted server. Provider requests incur the provider's usage charges.

Questions can be free form or contain four options on separate lines prefixed
with `A.`, `B.`, `C.`, and `D.`. The answer includes a collapsible route/evidence
trace. A reported parse score measures option matching, rather than model
confidence or clinical accuracy.

## Configure the server

| Variable | Default | Purpose |
| --- | --- | --- |
| `SLIDEBANK_HOST` | `127.0.0.1` | Listening address |
| `SLIDEBANK_PORT` | `7864` | Listening port; falls back to `GRADIO_SERVER_PORT` when set |
| `SLIDEBANK_UI_TREE_DIR` | `outputs/ui_tree` | Directory containing `*.ui_tree.json` |
| `SLIDEBANK_BUILD_CONFIG` | `configs/build.yaml` | WSI bank-build configuration |
| `SLIDEBANK_INFER_CONFIG` | `configs/infer.yaml` | QA model and bank configuration |
| `SLIDEBANK_BANK_DIR` | Inferred or configured | Override the retrieval bank |
| `SLIDEBANK_ARTIFACT_DIR` | Inferred or configured | Override the image artifact directory |
| `SLIDEBANK_RUN_DIR` | `outputs/workbench_runs` | Destination for rebuilt banks |
| `GRADIO_TEMP_DIR` | `outputs/gradio_tmp` | Gradio cache/upload directory |

Relative workbench environment paths and data paths in its YAML configurations
resolve against the repository root. CLI scripts use the working directory, so
run the documented CLI commands from the repository root. Changing the tree
directory requires restarting the server to refresh the configured slide list.

For remote use, keep the default localhost binding and forward the port:

```bash
ssh -L 7864:127.0.0.1:7864 username@server
```

The static project page can be hosted on GitHub Pages. This workbench needs a
Python server and its configured data/model access; deploy it separately on a
machine or service that supports those resources.

## Validation without model calls

```bash
python -m unittest discover -s tests -p 'test_ui_release.py' -v
```

These checks build the empty Gradio interface, resolve tree paths, reject invalid
schema and raw slide uploads, and verify that missing banks or credentials stop
before inference. They use temporary structural fixtures and never produce
predictions or package patient images.

# Hosted evidence workbench

The hosted workbench opens the prebuilt evidence bank in `demos/`. Visitors can
browse the slide, regions, anchors, and patch captions without model inference.
On ZeroGPU, questions run through the shared question policy, agent router with
rule fallback, compact evidence retrieval, and final answer generation.

The hosted model is [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).
It is a demonstration backbone, separate from the paper's pathology models and
from the models used to construct the stored observations. Hosted answers are
generated when a visitor submits a question; the viewer does not replay saved
answers.

## CPU preview

```bash
python -m pip install -e '.[ui]'
python space_app.py
```

Open <http://127.0.0.1:7860>. The launcher binds to `0.0.0.0` for hosted use.
On ordinary CPU hardware it starts the evidence browser and disables live QA.
It does not load a language model or send a request to an inference provider.

The hosted interface uses the bundled slides. Upload, model/API configuration,
and rebuilding controls remain available in the local `python app.py` workbench.

## Hugging Face Space

Prepare the deployment directory:

```bash
python scripts/stage_space.py --out-dir build/hf-space
```

The staging command packages the code and demo assets, using
`hf_space/README.md` as the Space card and `hf_space/requirements.txt` as its
root requirements file. The repository's main README and requirements stay
unchanged. The Space requirements list dependencies directly because the
Gradio builder installs them before copying application source; the app imports
the bundled Python package from its working directory. Upload the staged
directory to the intended Space:

```bash
hf upload gexinh/SlideBank build/hf-space --repo-type space
```

For a fork, replace `gexinh/SlideBank` with its Space repository ID. Select
**ZeroGPU** in the Space hardware settings for live QA. The launcher checks the
platform's `SPACES_ZERO_GPU` flag and the Spaces SDK; it keeps QA disabled on
`cpu-basic` even though the GPU decorator alone would be a no-op there.

The Space uses Python 3.10, Gradio 5.34.2, PyTorch 2.8.0, and Transformers
4.51.3. The standalone Space requirements and the `hosted` package extra declare
the same model/runtime dependencies.
Quilt-LLaVA's external legacy runtime is not needed by this hosted model.

## Model preparation and GPU execution

The Space card's `preload_from_hub` entry caches the Qwen checkpoint during the
Space build. Startup then explicitly prepares the model using the Spaces SDK's
CUDA emulation, before starting the web interface. Importing the launcher does
not prepare a model; CPU-only startup skips this step entirely.

The complete question-answering call runs inside `spaces.GPU(duration=120)`.
This includes routing and final answer generation, so both calls use the same
allocated GPU and cached client. The workbench queues one request at a time,
with up to eight queued requests. GPU availability and daily quota are managed
by Hugging Face. An allocation, quota, or inference error leaves evidence
browsing available and asks the visitor to retry later.

No visitor API key or paid inference endpoint is used. There is no automatic
fallback to an API or to multi-billion-parameter CPU inference.

The runtime follows the [ZeroGPU SDK documentation](https://huggingface.co/docs/hub/spaces-zerogpu).
The Space's hardware selection and build status determine whether live QA is
available; a successful CPU preview checks browsing, not GPU inference.

## Configuration

| Setting | Hosted default |
| --- | --- |
| Entry point | `space_app.py` |
| Inference configuration | `configs/hosted.yaml` |
| Evidence trees | `demos/ui_tree` |
| Retrieval bank | `demos/bank` |
| Image artifacts | `demos/artifacts` |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Device and precision | `cuda:0`, `bf16` |
| Output limit | 256 generated tokens per model call |
| Image budget | 401,408 pixels, about 512 visual tokens for Qwen |
| Question limit | 2,000 characters |
| Host and port | `0.0.0.0:7860` |

The existing `SLIDEBANK_UI_TREE_DIR`, `SLIDEBANK_BANK_DIR`,
`SLIDEBANK_ARTIFACT_DIR`, `SLIDEBANK_INFER_CONFIG`, `SLIDEBANK_HOST`, and
`SLIDEBANK_PORT` variables can override the corresponding launcher paths or
network settings. `SLIDEBANK_GPU_DURATION` sets a GPU duration between 15 and
120 seconds. Changing model, path, or hardware settings requires a Space restart.

`allow_remote_download: true` in `configs/hosted.yaml` explicitly permits model
downloads on the hosted path. Local Hugging Face inference remains cache-only
by default; enabling downloads for the Space does not alter that default.

## Checks without model downloads

```bash
python -m unittest discover -s tests -p 'test_hosted_runtime.py' -v
```

These checks exercise CPU-only startup, hosted control restrictions, model
configuration, the GPU boundary, and decoder output parsing with temporary
fixtures. Actual ZeroGPU generation is checked on the deployed Space.

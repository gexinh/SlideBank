# Data and bank contract

No WSIs, benchmark annotations, model weights, private bank exports, or existing
run outputs are bundled. Supply these separately and edit the YAML templates.
Run the documented CLI commands from the repository root.

## CSV input

Both CLI entry points use `slidebank/data.py::load_mcq_rows`. The CSV is read as
UTF-8 with an optional BOM.

| Column | Meaning |
| --- | --- |
| `Slide` | Slide identifier or prefix used to match a WSI; `slide` is accepted. |
| `Question` | Question text; `question` is accepted. Rows missing a slide or question are skipped. |
| `A`, `B`, `C`, `D` | Four option texts, in label order; lowercase option headers are not aliases. |
| `Answer` | Optional reference answer text or label; `answer` is accepted. |
| `GT` | Optional explicit reference label A/B/C/D, taking priority over answer-text matching; `gt` is accepted. |
| `Image` | Optional thumbnail path or filename; `image` is accepted. |
| `RowId` | Optional record ID; fallbacks are `row_id`, `Unnamed: 0`, `ID`, then the zero-based CSV row number. |

Bank construction uses this loader to obtain the slide list. The loader still
requires a nonempty question field, although bank construction explores each
slide independently of the question text. Inference expects meaningful four-way
options. Labels are optional for unlabeled prediction; accuracy is interpretable
only when valid reference labels are supplied.

## Slide matching

The sample configurations use `data/slides` and `**/*.svs`, which searches
recursively. The CLI fallback pattern is `*/*.svs`. Paths are sorted, then
indexed by the filename prefix before `-01Z-`; for other naming conventions the
filename is retained. Matching first checks an exact key and then a prefix in
either direction. If multiple files share a key, the first sorted path is kept.
Use unambiguous slide identifiers and verify matching before a large run.

## Outputs and retrieval

With the provided configuration, bank construction writes under:

```text
outputs/memory_bank/
  build_summary.json
  wsi_vqa_eval_memory_fallback.json         # persistent collection manifest
  wsi_vqa_eval_memory_fallback.json.shards/ # per-slide JSONL records
  readable/                               # human-readable per-slide JSONL
  artifacts/                              # thumbnails, regions, patches
```

The local collection also supports older JSON and JSONL persistence formats.
The manifest and shards hold the retrieval records; `readable/` is the source
for visualization export. `build_summary.json` records requested, ingested,
reused, missing, and failed slides; inspect these fields before treating a run
as complete. Rebuild/force-ingest options may replace existing records and
artifacts, so use a separate output directory for an independent experiment.

`scripts/infer.py` writes JSONL predictions and a JSON summary to the configured
`output.jsonl` and `output.summary` paths. Use a separate output prefix when
running independent comparisons.

## Visualization export

```bash
python scripts/export_ui_tree.py --bank-dir outputs/memory_bank \
  --out-dir outputs/ui_tree
```

The exporter writes one `*.ui_tree.json` per slide plus `export_summary.json`.
The schema identifier is `slidebank-ui-tree-v1`:

| Field | Contents |
| --- | --- |
| `slide_id` | Slide identifier. |
| `global` | Summary, selected metrics, signal overview, and thumbnail metadata. |
| `global.thumbnail` | Thumbnail path, optional overlay path, level-0 dimensions, grid shape. |
| `tiles[]` | Region ID, grid position, level-0 bounding box, region image path, description, and anchors. |
| `tiles[].anchors_summary[]` | Anchor descriptions, selection confidence, and candidate signal summaries. |
| `tiles[].anchors[]` | Anchor ID, evidence ID, bounding box, multi-scale `views`, and optional signal-screening results. |
| `tiles[].anchors[].views[]` | View ID, magnification, image path, caption, and level-0 bounding box. |

Bounding boxes named `xywh` are `[x, y, width, height]`; `level0` coordinates use
the full-resolution slide. Selection confidence is a construction-time value,
not an inference probability or a benchmark accuracy.

The export references image files and does not embed them. The Python workbench
loads server-accessible paths; configured trees can resolve relative paths
against their JSON directory. The static project viewer reads selected images
from the visitor's local folder and matches paths by their longest unique suffix.
Neither a tree JSON nor its images replace the persistent bank needed for QA.
See [WORKBENCH.md](WORKBENCH.md) for runtime configuration.

New builds normalize their input and output directories to absolute paths so
moving the exported JSON into a separate visualization directory does not change
its image base. For older banks, the exporter resolves relative paths only when
the referenced file actually exists under its working directory. It preserves
other relative paths for portable JSON/image bundles. Unresolved paths are not
guessed: restore the original images, correct the tree paths, or rebuild the bank.

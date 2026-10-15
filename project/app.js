"use strict";

const SlideBankViewerData = (() => {
  function validateTree(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)
        || value.schema_version !== "slidebank-ui-tree-v1"
        || !value.global || typeof value.global !== "object" || Array.isArray(value.global)
        || !Array.isArray(value.tiles)) {
      throw new Error("Choose a slidebank-ui-tree-v1 JSON exported by SlideBank.");
    }
    for (const tile of value.tiles) {
      if (!tile || typeof tile !== "object" || !Array.isArray(tile.anchors)) {
        throw new Error("Invalid region record: expected an anchors array.");
      }
      for (const anchor of tile.anchors) {
        if (!anchor || typeof anchor !== "object" || !Array.isArray(anchor.views)
            || anchor.views.some((view) => !view || typeof view !== "object")) {
          throw new Error("Invalid anchor record: expected patch view objects.");
        }
      }
    }
    return value;
  }

  function demoUrl(path, base, root) {
    const raw = String(path || "");
    if (!raw || raw !== raw.trim() || /^[a-z][a-z\d+.-]*:/i.test(raw)
        || raw.startsWith("/") || /[\\?#]/.test(raw) || /%2f|%5c/i.test(raw)) {
      throw new Error("Example assets must use relative paths inside demos/.");
    }
    const url = new URL(raw, base);
    const boundary = new URL(root);
    if (url.protocol !== boundary.protocol || url.origin !== boundary.origin
        || !url.pathname.startsWith(boundary.pathname) || url.search || url.hash) {
      throw new Error("Example asset is outside the published demos directory.");
    }
    return url.href;
  }

  function validateManifest(value, manifestUrl) {
    if (!value || value.schema_version !== "slidebank-demo-manifest-v1"
        || !Array.isArray(value.samples) || !value.samples.length) {
      throw new Error("The example slide list is unavailable. Please retry.");
    }
    const root = new URL("./", manifestUrl).href;
    const seen = new Set();
    const samples = value.samples.map((sample) => {
      if (!sample || typeof sample.id !== "string" || !sample.id || seen.has(sample.id)
          || typeof sample.label !== "string" || typeof sample.tree !== "string"
          || !Array.isArray(sample.images) || sample.images.some((path) => typeof path !== "string")) {
        throw new Error("The example slide list has an invalid sample record.");
      }
      seen.add(sample.id);
      return {
        id: sample.id,
        label: sample.label,
        treeUrl: demoUrl(sample.tree, manifestUrl, root),
        imageUrls: new Set(sample.images.map((path) => demoUrl(path, manifestUrl, root))),
        root,
      };
    });
    const defaultId = value.default_demo || samples[0].id;
    if (!seen.has(defaultId)) throw new Error("The default example is missing from the slide list.");
    return { samples, defaultId };
  }

  function presetImage(path, sample) {
    if (!sample || !path) return null;
    try {
      const url = demoUrl(path, sample.treeUrl, sample.root);
      return sample.imageUrls.has(url) ? url : null;
    } catch { return null; }
  }

  const basename = (value) => String(value || "").replaceAll("\\", "/").split("/").pop();
  function localImage(path, images) {
    const wanted = String(path || "").replaceAll("\\", "/");
    if (!wanted) return null;
    const candidates = images.filter((item) => item.url.startsWith("blob:") && basename(item.path) === basename(wanted));
    if (candidates.length === 1) return candidates[0].url;
    let best = null;
    let score = 0;
    let tied = false;
    for (const item of candidates) {
      const a = wanted.split("/");
      const b = item.path.split("/");
      let count = 0;
      while (a.length && b.length && a.pop() === b.pop()) count++;
      if (count > score) { best = item; score = count; tied = false; }
      else if (count === score) tied = true;
    }
    return best && !tied ? best.url : null;
  }

  function imageUrl(mode, path, sample, images) {
    return mode === "preset" ? presetImage(path, sample) : localImage(path, images);
  }
  return { validateTree, validateManifest, demoUrl, imageUrl, basename };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SlideBankViewerData;

if (typeof document !== "undefined") (() => {
  const $ = (id) => document.getElementById(id);
  const { validateTree, validateManifest, imageUrl, basename } = SlideBankViewerData;
  const manifestUrl = new URL("demos/manifest.json", document.baseURI).href;
  let manifest = null;
  let sample = null;
  let mode = "preset";
  let tree = null;
  let tileIndex = 0;
  let anchorIndex = 0;
  let localImages = [];
  let loadSequence = 0;
  let controller = null;
  const list = (value) => Array.isArray(value) ? value : [];
  const status = (message) => { $("viewer-status").textContent = message; };
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };

  function releaseLocalImages() {
    for (const item of localImages) URL.revokeObjectURL(item.url);
    localImages = [];
    $("image-folder").value = "";
  }

  function cancelLoad() {
    loadSequence++;
    if (controller) controller.abort();
    controller = null;
  }

  function clearSelection() {
    tree = null; tileIndex = 0; anchorIndex = 0; sample = null;
  }

  function firstAnchor(tile) {
    const index = list(tile && tile.anchors).findIndex((anchor) => list(anchor.views).length);
    return Math.max(0, index);
  }

  function selectInitialEvidence() {
    const index = tree.tiles.findIndex((tile) => tile.anchors.some((anchor) => anchor.views.length));
    tileIndex = Math.max(0, index);
    anchorIndex = firstAnchor(tree.tiles[tileIndex]);
  }

  function showImage(container, path, label) {
    container.replaceChildren();
    const url = imageUrl(mode, path, sample, localImages);
    if (url) {
      const img = element("img");
      img.src = url;
      img.alt = label;
      img.addEventListener("error", () => {
        if (!container.contains(img)) return;
        container.replaceChildren(element("span", "This image could not be displayed."));
        if (mode === "preset") {
          status("An example image could not load. Reset the example to retry.");
          $("retry-demo").hidden = false;
        }
      }, { once: true });
      container.append(img);
    } else {
      const message = !path ? "No image referenced in this record." : mode === "preset"
        ? "This image is not included in the selected example."
        : `${basename(path)}\nAdd your image folder to display this view.`;
      container.append(element("span", `${label}\n${message}`));
    }
  }

  function paragraph(container, label, value) {
    if (value === null || value === undefined || value === "") return;
    const p = element("p");
    p.append(element("strong", `${label} `), document.createTextNode(Array.isArray(value) ? value.join(", ") : String(value)));
    container.append(p);
  }

  function choose(container, records, idKey, selectedIndex, onSelect) {
    container.replaceChildren();
    records.forEach((record, index) => {
      const button = element("button", record[idKey] || `${idKey} ${index + 1}`, "choice");
      button.type = "button";
      button.setAttribute("aria-pressed", String(index === selectedIndex));
      button.addEventListener("click", () => onSelect(index));
      container.append(button);
    });
  }

  function render() {
    $("viewer-mode").textContent = mode === "preset" ? "Example slide" : "Your evidence";
    for (const id of ["slide-info", "region-info", "patch-content", "tile-list", "anchor-list"]) $(id).replaceChildren();
    if (!tree) {
      $("global-image").replaceChildren(element("span", mode === "preset" ? "Loading example slide…" : "Open your evidence JSON"));
      $("region-image").replaceChildren(element("span", "Select a region"));
      $("patch-content").append(element("p", "Anchor patches appear here after a slide is loaded.", "panel-copy"));
      return;
    }
    const global = tree.global;
    const thumbnail = global.thumbnail || {};
    showImage($("global-image"), thumbnail.thumbnail_path, "Global slide overview");
    paragraph($("slide-info"), "Slide", tree.slide_id || "Unnamed");
    paragraph($("slide-info"), "Summary", global.summary);
    paragraph($("slide-info"), "Dimensions (level 0)", thumbnail.slide_dimensions_level0);
    paragraph($("slide-info"), "Regions", tree.tiles.length);
    choose($("tile-list"), tree.tiles, "tile_id", tileIndex, (index) => {
      tileIndex = index; anchorIndex = firstAnchor(tree.tiles[index]); render();
    });
    const tile = tree.tiles[tileIndex];
    if (!tile) {
      $("region-image").replaceChildren(element("span", "This bank contains no regions."));
      return;
    }
    showImage($("region-image"), tile.tile_image_path, String(tile.tile_id || "Region"));
    paragraph($("region-info"), "Region", tile.tile_id);
    paragraph($("region-info"), "Observation", tile.tile_summary);
    paragraph($("region-info"), "Coarse label", tile.coarse_label);
    paragraph($("region-info"), "Bounding box (x, y, w, h)", tile.bbox_xywh_level0);
    choose($("anchor-list"), tile.anchors, "anchor_id", anchorIndex, (index) => { anchorIndex = index; render(); });
    const anchor = tile.anchors[anchorIndex];
    if (!anchor) {
      $("patch-content").append(element("p", "This region has no anchor views.", "panel-copy"));
      return;
    }
    paragraph($("patch-content"), "Anchor", anchor.anchor_id);
    const summary = list(tile.anchors_summary).find((record) => record && record.anchor_id === anchor.anchor_id);
    if (summary) paragraph($("patch-content"), "Observation", summary.summary);
    if (!anchor.views.length) $("patch-content").append(element("p", "No patch views recorded.", "panel-copy"));
    for (const view of anchor.views) {
      const card = element("div", undefined, "patch-card");
      const magnification = String(view.mag || "").replace(/[x×]$/i, "");
      card.append(element("h4", `${view.view_id || "Patch"}${magnification ? ` · ${magnification}×` : ""}`));
      const slot = element("div", undefined, "image-slot");
      showImage(slot, view.patch_path, String(view.view_id || "Patch"));
      card.append(slot);
      if (view.caption) card.append(element("p", view.caption));
      paragraph(card, "Bounding box", view.bbox_xywh_level0);
      $("patch-content").append(card);
    }
    const signals = list(anchor.signal_screening).filter((signal) => signal && typeof signal === "object");
    if (signals.length) {
      const block = element("div", undefined, "signals");
      block.append(element("strong", "Screened signals"));
      for (const signal of signals) {
        paragraph(block, signal.signal_name || signal.signal_id || "Signal", signal.status || "Unspecified");
        if (signal.rationale) block.append(element("p", signal.rationale));
      }
      $("patch-content").append(block);
    }
  }

  async function fetchJson(url, signal) {
    const response = await fetch(url, { signal, credentials: "omit", cache: "no-cache", redirect: "error" });
    if (!response.ok) throw new Error(`Example data could not load (HTTP ${response.status}).`);
    return response.json();
  }

  async function loadExample(id) {
    if (!manifest) return loadManifest();
    const selected = manifest.samples.find((entry) => entry.id === id) || manifest.samples.find((entry) => entry.id === manifest.defaultId);
    cancelLoad();
    const sequence = loadSequence;
    controller = new AbortController();
    mode = "preset"; clearSelection(); releaseLocalImages(); $("tree-file").value = "";
    $("demo-select").value = selected.id;
    $("retry-demo").hidden = true;
    render(); status(`Loading ${selected.label}…`);
    try {
      const loaded = validateTree(await fetchJson(selected.treeUrl, controller.signal));
      if (sequence !== loadSequence) return;
      sample = selected; tree = loaded; selectInitialEvidence(); render();
      status(`${selected.label} · ${tree.tiles.length} regions. Select a region and anchor to explore.`);
    } catch (error) {
      if (sequence !== loadSequence || error.name === "AbortError") return;
      $("global-image").replaceChildren(element("span", "Example unavailable. Use Retry example."));
      status(error.message || "The example could not load. Please retry.");
      $("retry-demo").hidden = false;
    }
  }

  async function loadManifest() {
    cancelLoad();
    const sequence = loadSequence;
    controller = new AbortController();
    mode = "preset"; clearSelection(); releaseLocalImages(); render();
    $("demo-select").disabled = true; $("retry-demo").hidden = true;
    status("Loading the example slide…");
    try {
      const loaded = validateManifest(await fetchJson(manifestUrl, controller.signal), manifestUrl);
      if (sequence !== loadSequence) return;
      manifest = loaded;
      const placeholder = element("option", "Choose an example");
      placeholder.value = ""; placeholder.disabled = true;
      $("demo-select").replaceChildren(placeholder, ...manifest.samples.map((entry) => {
        const option = element("option", entry.label); option.value = entry.id; return option;
      }));
      $("demo-select").disabled = false;
      await loadExample(manifest.defaultId);
    } catch (error) {
      if (sequence !== loadSequence || error.name === "AbortError") return;
      $("global-image").replaceChildren(element("span", "Example unavailable. Use Retry example."));
      status(error.message || "The example list could not load. Please retry.");
      $("retry-demo").hidden = false;
    }
  }

  function ownEvidence() {
    cancelLoad();
    if (mode !== "local") { clearSelection(); releaseLocalImages(); }
    mode = "local"; sample = null; $("demo-select").value = ""; $("retry-demo").hidden = true;
  }

  $("demo-select").addEventListener("change", (event) => loadExample(event.target.value));
  $("reset-demo").addEventListener("click", () => loadExample(manifest ? manifest.defaultId : ""));
  $("retry-demo").addEventListener("click", () => loadExample($("demo-select").value));
  $("tree-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    ownEvidence(); clearSelection(); releaseLocalImages(); render();
    const sequence = loadSequence;
    try {
      if (file.size > 50 * 1024 * 1024) throw new Error("Choose an evidence tree smaller than 50 MB.");
      const loaded = validateTree(JSON.parse(await file.text()));
      if (sequence !== loadSequence) return;
      tree = loaded; selectInitialEvidence(); render();
      status(`Your evidence: ${tree.slide_id || file.name} · ${tree.tiles.length} regions. Add its image folder to display views.`);
    } catch (error) { if (sequence === loadSequence) status(error.message || "Could not read this evidence tree."); }
  });
  $("image-folder").addEventListener("change", (event) => {
    const files = Array.from(event.target.files);
    ownEvidence(); releaseLocalImages();
    localImages = files.filter((file) => /\.(png|jpe?g|webp|gif|bmp)$/i.test(file.name))
      .map((file) => ({ path: file.webkitRelativePath || file.name, url: URL.createObjectURL(file) }));
    render();
    status(`${localImages.length} local images available${tree ? ` for ${tree.slide_id || "your slide"}` : "; open your evidence JSON next"}.`);
  });
  $("clear-data").addEventListener("click", () => {
    ownEvidence(); clearSelection(); releaseLocalImages(); $("tree-file").value = "";
    render(); status("Your evidence cleared. Open a JSON, or use Reset example to return to the sample slide.");
  });
  window.addEventListener("beforeunload", () => { cancelLoad(); releaseLocalImages(); });

  $("copy-citation").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("bibtex").textContent);
      $("copy-status").textContent = "BibTeX copied.";
    } catch {
      const range = document.createRange(); range.selectNodeContents($("bibtex"));
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      $("copy-status").textContent = "Citation selected. Use your browser's Copy command.";
    }
  });

  const config = window.SLIDEBANK_CONFIG || {};
  if (config.demoUrl) {
    try {
      const url = new URL(config.demoUrl);
      const pageUrl = new URL(config.demoPageUrl || url.href);
      if (url.protocol !== "https:" || pageUrl.protocol !== "https:") throw new Error("The hosted workbench URL must use HTTPS.");
      $("demo-link").href = pageUrl.href; $("demo-frame").src = url.href;
      $("hosted-workbench").hidden = false;
      $("local-workbench").hidden = true;
      $("open-workbench").hidden = false;
      $("workbench-description").textContent = config.qaEnabled === true
        ? "Ask questions about the preloaded slide below. The hosted demo uses Qwen2.5-VL-3B on Hugging Face ZeroGPU; no API key is needed."
        : "Explore the preloaded slide in the full Gradio workbench below. Its interface shows the available question-answering options.";
    } catch (error) { $("workbench-description").textContent = "Browse the example above, or run the Gradio workbench locally using the guide below."; }
  }

  if (config.mainFigure === "main-figure.png") {
    $("main-figure").src = "assets/main-figure.png";
    $("main-figure").alt = "SlideBank paper main figure.";
  }
  loadManifest();
})();

"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  let tree = null;
  let tileIndex = 0;
  let anchorIndex = 0;
  let localImages = [];
  const list = (value) => Array.isArray(value) ? value : [];
  const basename = (value) => String(value || "").replaceAll("\\", "/").split("/").pop();
  const status = (message) => { $("viewer-status").textContent = message; };
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };

  function validateTree(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)
        || value.schema_version !== "slidebank-ui-tree-v1"
        || !value.global || typeof value.global !== "object"
        || !Array.isArray(value.tiles)) {
      throw new Error("Choose a slidebank-ui-tree-v1 JSON file exported by scripts/export_ui_tree.py.");
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

  function resolveImage(path) {
    const wanted = String(path || "").replaceAll("\\", "/");
    if (!wanted) return null;
    const candidates = localImages.filter((item) => basename(item.path) === basename(wanted));
    if (candidates.length === 1) return candidates[0].url;
    let best = null;
    let bestScore = 0;
    let tied = false;
    for (const item of candidates) {
      const a = wanted.split("/");
      const b = item.path.split("/");
      let score = 0;
      while (a.length && b.length && a.pop() === b.pop()) score++;
      if (score > bestScore) { best = item; bestScore = score; tied = false; }
      else if (score === bestScore) tied = true;
    }
    return best && !tied ? best.url : null;
  }

  function showImage(container, path, label) {
    container.replaceChildren();
    const url = resolveImage(path);
    if (url) {
      const img = element("img");
      img.src = url;
      img.alt = label;
      img.addEventListener("error", () => container.replaceChildren(element("span", "Image could not be displayed.")), { once: true });
      container.append(img);
    } else {
      container.append(element("span", path ? `${label}\n${basename(path)}\nAdd the image folder to display this view.` : `${label}\nNo image referenced in this record.`));
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
    $("slide-info").replaceChildren();
    $("region-info").replaceChildren();
    $("patch-content").replaceChildren();
    $("tile-list").replaceChildren();
    $("anchor-list").replaceChildren();
    if (!tree) {
      $("global-image").replaceChildren(element("span", "No slide loaded"));
      $("region-image").replaceChildren(element("span", "Select a region"));
      $("patch-content").append(element("p", "Open an evidence JSON to inspect its patch views.", "panel-copy"));
      return;
    }
    const global = tree.global;
    const thumbnail = global.thumbnail || {};
    showImage($("global-image"), thumbnail.thumbnail_path, "Global slide overview");
    paragraph($("slide-info"), "Slide", tree.slide_id || "Unnamed");
    paragraph($("slide-info"), "Summary", global.summary);
    paragraph($("slide-info"), "Dimensions (level 0)", thumbnail.slide_dimensions_level0);
    paragraph($("slide-info"), "Regions", tree.tiles.length);
    choose($("tile-list"), tree.tiles, "tile_id", tileIndex, (index) => { tileIndex = index; anchorIndex = 0; render(); });
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

  $("tree-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 50 * 1024 * 1024) throw new Error("Choose an evidence tree smaller than 50 MB.");
      const loaded = validateTree(JSON.parse(await file.text()));
      tree = loaded; tileIndex = 0; anchorIndex = 0;
      render();
      status(`Loaded ${tree.slide_id || file.name} · ${tree.tiles.length} regions · ${localImages.length} local images.`);
    } catch (error) { status(error.message || "Could not read this evidence tree."); }
  });

  $("image-folder").addEventListener("change", (event) => {
    for (const item of localImages) URL.revokeObjectURL(item.url);
    // SVG and remote URLs are not loaded from imported evidence.
    localImages = Array.from(event.target.files)
      .filter((file) => /\.(png|jpe?g|webp|gif|bmp)$/i.test(file.name))
      .map((file) => ({ path: file.webkitRelativePath || file.name, url: URL.createObjectURL(file) }));
    render();
    status(`${localImages.length} local images available${tree ? ` for ${tree.slide_id || "this slide"}` : "; open an evidence JSON next"}.`);
  });

  $("clear-data").addEventListener("click", () => {
    for (const item of localImages) URL.revokeObjectURL(item.url);
    localImages = []; tree = null; tileIndex = 0; anchorIndex = 0;
    $("tree-file").value = ""; $("image-folder").value = "";
    render(); status("Load a .ui_tree.json exported by SlideBank.");
  });

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

  const demoUrl = (window.SLIDEBANK_CONFIG || {}).demoUrl;
  if (demoUrl) {
    try {
      const url = new URL(demoUrl);
      if (url.protocol !== "https:") throw new Error("The hosted workbench URL must use HTTPS.");
      $("demo-link").href = url.href; $("demo-frame").src = url.href;
      $("hosted-workbench").hidden = false;
    } catch (error) { console.warn(error.message); }
  }

  if ((window.SLIDEBANK_CONFIG || {}).mainFigure === "main-figure.png") {
    $("main-figure").src = "assets/main-figure.png";
    $("main-figure").alt = "SlideBank paper main figure.";
    const link = element("a", "Paper main figure · Download PDF");
    link.href = "assets/main-figure.pdf";
    $("figure-caption").replaceChildren(link);
  }
})();

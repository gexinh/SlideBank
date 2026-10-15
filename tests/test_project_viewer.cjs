"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { validateManifest, validateTree, demoUrl, imageUrl } = require("../project/app.js");
const manifestUrl = "https://example.org/SlideBank/demos/manifest.json";
const rawManifest = () => ({
  schema_version: "slidebank-demo-manifest-v1", default_demo: "one",
  samples: [{ id: "one", label: "One", tree: "ui_tree/one.json", images: ["artifacts/one/patch.png"] },
    { id: "two", label: "Two", tree: "ui_tree/two.json", images: ["artifacts/two/patch.png"] }],
});

test("preset tree paths resolve only to the selected sample's exact image whitelist", () => {
  const manifest = validateManifest(rawManifest(), manifestUrl);
  const one = manifest.samples[0];
  assert.equal(imageUrl("preset", "../artifacts/one/patch.png", one, []), "https://example.org/SlideBank/demos/artifacts/one/patch.png");
  assert.equal(imageUrl("preset", "../artifacts/two/patch.png", one, []), null);
  assert.equal(imageUrl("preset", "../artifacts/one/missing.png", one, []), null);
});

test("preset asset rules reject external, server, and escaped paths", () => {
  const root = "https://example.org/SlideBank/demos/";
  for (const path of ["https://evil.invalid/a.png", "//evil.invalid/a.png", "/data/private.png", "../../private.png", "../%2fprivate.png", "file:///tmp/a.png", "a.png?secret=1", "a.png#fragment", "..\\private.png"]) {
    assert.throws(() => demoUrl(path, manifestUrl, root));
  }
  const value = rawManifest(); value.samples[0].images.push("https://evil.invalid/a.png");
  assert.throws(() => validateManifest(value, manifestUrl));
});

test("local JSON never inherits preset image URLs or fetches a path from the JSON", () => {
  const one = validateManifest(rawManifest(), manifestUrl).samples[0];
  for (const path of ["../artifacts/one/patch.png", "https://evil.invalid/patch.png", "/data/private/patch.png"]) {
    assert.equal(imageUrl("local", path, one, []), null);
    assert.equal(imageUrl("local", path, one, [{ path: "patch.png", url: "https://evil.invalid/patch.png" }]), null);
  }
  assert.equal(imageUrl("local", "/data/private/patch.png", one, [{ path: "my-images/patch.png", url: "blob:local-file" }]), "blob:local-file");
});

test("local image basename collisions require an unambiguous folder suffix", () => {
  const files = [{ path: "set/a/patch.png", url: "blob:a" }, { path: "set/b/patch.png", url: "blob:b" }];
  assert.equal(imageUrl("local", "/server/a/patch.png", null, files), "blob:a");
  assert.equal(imageUrl("local", "/server/patch.png", null, files), null);
});

test("invalid manifest and tree schemas fail before rendering", () => {
  assert.throws(() => validateManifest({ samples: [] }, manifestUrl));
  const duplicate = rawManifest(); duplicate.samples[1].id = "one";
  assert.throws(() => validateManifest(duplicate, manifestUrl));
  assert.throws(() => validateTree({ schema_version: "slidebank-ui-tree-v1", global: {}, tiles: [{ anchors: [{}] }] }));
  assert.throws(() => validateTree({ schema_version: "slidebank-ui-tree-v1", global: [], tiles: [] }));
});

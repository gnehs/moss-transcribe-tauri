import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { findMetallibs } from "./prepare-mlx-metallib.mjs";

test("finds metallibs from mlx-sys Cargo build directories", (t) => {
  const targetRoot = fs.mkdtempSync(path.join(os.tmpdir(), "moss-metallib-"));
  t.after(() => fs.rmSync(targetRoot, { recursive: true, force: true }));

  const currentPath = createMetallib(targetRoot, "release", "mlx-sys-current");
  createMetallib(targetRoot, "release", "unrelated-crate-build");

  assert.deepEqual(findMetallibs([targetRoot], ["release"]).sort(), [
    currentPath,
  ]);
});

test("only searches the requested Cargo profile", (t) => {
  const targetRoot = fs.mkdtempSync(path.join(os.tmpdir(), "moss-metallib-"));
  t.after(() => fs.rmSync(targetRoot, { recursive: true, force: true }));

  const debugPath = createMetallib(targetRoot, "debug", "mlx-sys-debug");
  createMetallib(targetRoot, "release", "mlx-sys-release");

  assert.deepEqual(findMetallibs([targetRoot], ["debug"]), [debugPath]);
});

function createMetallib(targetRoot, profile, buildDirectory) {
  const metallibPath = path.join(
    targetRoot,
    profile,
    "build",
    buildDirectory,
    "out",
    "build",
    "lib",
    "mlx.metallib"
  );
  fs.mkdirSync(path.dirname(metallibPath), { recursive: true });
  fs.writeFileSync(metallibPath, buildDirectory);
  return metallibPath;
}

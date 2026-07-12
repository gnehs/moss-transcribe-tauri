import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "..");
const files = {
  packageJson: path.join(repoRoot, "package.json"),
  cargoManifest: path.join(repoRoot, "src-tauri", "Cargo.toml"),
  tauriConfig: path.join(repoRoot, "src-tauri", "tauri.conf.json"),
  cargoLock: path.join(repoRoot, "src-tauri", "Cargo.lock"),
};
const cargoPackageName = "moss-transcribe-tauri";

const semverPattern =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main();
}

function main() {
  const [version] = process.argv.slice(2);

  if (process.argv.length !== 3 || !semverPattern.test(version ?? "")) {
    console.error("Usage: pnpm version:set <x.y.z>");
    process.exitCode = 1;
    return;
  }

  const originalContents = new Map(
    Object.values(files).map((filePath) => [filePath, read(filePath)])
  );

  try {
    write(
      files.packageJson,
      updateJsonVersion(
        files.packageJson,
        originalContents.get(files.packageJson),
        version
      )
    );
    write(
      files.cargoManifest,
      updateCargoVersion(
        files.cargoManifest,
        originalContents.get(files.cargoManifest),
        version
      )
    );
    write(
      files.tauriConfig,
      updateJsonVersion(
        files.tauriConfig,
        originalContents.get(files.tauriConfig),
        version
      )
    );

    write(
      files.cargoLock,
      updateCargoLock(originalContents.get(files.cargoLock), version)
    );
    assertCargoLockVersion(read(files.cargoLock), version);
    console.log(`Updated application version to ${version}.`);
  } catch (error) {
    for (const [filePath, contents] of originalContents) {
      write(filePath, contents);
    }

    console.error(`Failed to update application version: ${error.message}`);
    process.exitCode = 1;
  }
}

function updateJsonVersion(filePath, contents, version) {
  const config = JSON.parse(contents);
  if (typeof config.version !== "string") {
    throw new Error(
      `${path.relative(repoRoot, filePath)} has no string version field.`
    );
  }

  const versionFieldPattern = /(^\s*"version"\s*:\s*)"[^"]+"/m;
  if (!versionFieldPattern.test(contents)) {
    throw new Error(`Could not locate the version field in ${filePath}.`);
  }

  return contents.replace(versionFieldPattern, `$1"${version}"`);
}

function updateCargoVersion(filePath, contents, version) {
  const versionFieldPattern = /(^version\s*=\s*)"[^"]+"/m;
  if (!versionFieldPattern.test(contents)) {
    throw new Error(`Could not locate the version field in ${filePath}.`);
  }

  return contents.replace(versionFieldPattern, `$1"${version}"`);
}

function updateCargoLock(contents, version) {
  const packagePattern = new RegExp(
    `(\\[\\[package\\]\\]\\s+name = "${cargoPackageName}"\\s+version = )"[^"]+"`
  );
  if (!packagePattern.test(contents)) {
    throw new Error(
      `Could not locate ${cargoPackageName} in src-tauri/Cargo.lock.`
    );
  }

  return contents.replace(packagePattern, `$1"${version}"`);
}

function assertCargoLockVersion(contents, version) {
  const packagePattern = new RegExp(
    `\\[\\[package\\]\\]\\s+name = "${cargoPackageName}"\\s+version = "([^"]+)"`
  );
  const match = contents.match(packagePattern);

  if (!match) {
    throw new Error(
      `Could not locate ${cargoPackageName} in src-tauri/Cargo.lock.`
    );
  }
  if (match[1] !== version) {
    throw new Error(
      `src-tauri/Cargo.lock still contains version ${match[1]}, expected ${version}.`
    );
  }
}

function read(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function write(filePath, contents) {
  fs.writeFileSync(filePath, contents);
}

export {
  assertCargoLockVersion,
  semverPattern,
  updateCargoLock,
  updateCargoVersion,
  updateJsonVersion,
};

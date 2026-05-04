// Dual ESM + CJS + types build using only `tsc`.
//
// Strategy:
//   1. Run tsc with `tsconfig.esm.json` to emit ES2022 modules under
//      dist/esm/. These are spec-conformant ESM with explicit `.js`
//      specifiers; Node loads them when consumers import this package
//      via `import` because `package.json` sets `"type": "module"`.
//   2. Run tsc with `tsconfig.cjs.json` to emit CommonJS under
//      dist/cjs-build/, then post-process: rename .js → .cjs, rewrite
//      relative `require("./foo.js")` calls to `require("./foo.cjs")`
//      so the CJS files load cleanly when extracted under a folder
//      that sibling-overrides the package's `"type": "module"`.
//   3. Drop a tiny `package.json` with `"type": "commonjs"` next to
//      the CJS output so Node treats `.cjs` files unambiguously even
//      if a downstream packer rewrites extensions.
//   4. Run tsc with `tsconfig.types.json` to emit declarations under
//      dist/types/.
//
// Keeping this in plain Node — no rollup, no tsup — minimises the
// devDependency surface for a binding that only exists to read
// Python-written traces.

import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");
const dist = join(root, "dist");

function runTsc(project) {
  // Resolve tsc: prefer the locally-installed binary, fall back to npx.
  const localTsc = join(root, "node_modules", ".bin", "tsc");
  const cmd = existsSync(localTsc) ? localTsc : "npx";
  const args = existsSync(localTsc) ? ["-p", project] : ["tsc", "-p", project];
  const res = spawnSync(cmd, args, { cwd: root, stdio: "inherit" });
  if (res.status !== 0) {
    throw new Error(`tsc -p ${project} failed with status ${res.status}`);
  }
}

function rewriteCjs(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      rewriteCjs(full);
      continue;
    }
    if (!entry.name.endsWith(".js")) continue;
    const cjsPath = full.replace(/\.js$/, ".cjs");
    let src = readFileSync(full, "utf8");
    // Rewrite relative require("./x.js") and require("../x.js")
    // emitted from `import "./x.js"` after tsc compiles to CJS.
    src = src.replace(
      /require\((['"])(\.\.?\/[^'"]+?)\.js\1\)/g,
      "require($1$2.cjs$1)"
    );
    writeFileSync(cjsPath, src);
    rmSync(full);
  }
}

function copyTree(srcDir, dstDir) {
  mkdirSync(dstDir, { recursive: true });
  for (const entry of readdirSync(srcDir, { withFileTypes: true })) {
    const s = join(srcDir, entry.name);
    const d = join(dstDir, entry.name);
    if (entry.isDirectory()) {
      copyTree(s, d);
    } else {
      cpSync(s, d);
    }
  }
}

function main() {
  if (existsSync(dist)) rmSync(dist, { recursive: true, force: true });
  mkdirSync(dist, { recursive: true });

  // 1. ESM
  runTsc("tsconfig.esm.json");

  // 2. CJS
  runTsc("tsconfig.cjs.json");
  const cjsBuild = join(dist, "cjs-build");
  const cjsOut = join(dist, "cjs");
  if (!existsSync(cjsBuild)) {
    throw new Error("cjs build did not emit output");
  }
  copyTree(cjsBuild, cjsOut);
  rmSync(cjsBuild, { recursive: true, force: true });
  rewriteCjs(cjsOut);
  // Pin the CJS subdirectory's module type so .cjs files always
  // load as CommonJS regardless of where the package is unpacked.
  writeFileSync(
    join(cjsOut, "package.json"),
    JSON.stringify({ type: "commonjs" }, null, 2) + "\n"
  );

  // 3. Types
  runTsc("tsconfig.types.json");

  // Sanity check: required entry points exist.
  const required = [
    join(dist, "esm", "index.js"),
    join(cjsOut, "index.cjs"),
    join(dist, "types", "index.d.ts"),
  ];
  for (const f of required) {
    if (!existsSync(f) || statSync(f).size === 0) {
      throw new Error(`expected build output missing or empty: ${f}`);
    }
  }
  console.log("build ok:", required.map((f) => f.replace(root + "/", "")).join(", "));
}

main();

/**
 * Assemble the Python side of the desktop app into desktop/resources/.
 *
 *   resources/python/   a standalone CPython, copied whole
 *   resources/app/      the server package and the built web UI
 *
 * Not PyInstaller. torch, transformers and docling all resolve imports and data
 * files at runtime, which PyInstaller has to be told about one hidden import at
 * a time; copying a real interpreter sidesteps the entire problem.
 *
 * The interpreter comes from uv, which ships python-build-standalone builds.
 * Those are relocatable only if PYTHONHOME points at the copy - their build
 * prefix is compiled in - and the shell sets that when it spawns the backend.
 */

import { execFileSync } from "node:child_process";
import {
  cpSync, existsSync, mkdirSync, readdirSync, readlinkSync, realpathSync,
  rmSync, statSync, symlinkSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DESKTOP = path.resolve(HERE, "..");
const REPO = path.resolve(DESKTOP, "..");
const RESOURCES = path.join(DESKTOP, "resources");
const PY_DIR = path.join(RESOURCES, "python");
const APP_DIR = path.join(RESOURCES, "app");

const WIN = process.platform === "win32";
const run = (cmd, args, opts = {}) =>
  execFileSync(cmd, args, { stdio: "inherit", ...opts });

function uv() {
  for (const candidate of ["uv", path.join(process.env.HOME || "", ".local/bin/uv")]) {
    try {
      execFileSync(candidate, ["--version"], { stdio: "ignore" });
      return candidate;
    } catch {
      /* try the next */
    }
  }
  throw new Error("uv is required to bundle the runtime: https://docs.astral.sh/uv/");
}

function standaloneRuntime(uvBin) {
  const found = execFileSync(uvBin, ["python", "find", "3.12"], { encoding: "utf8" }).trim();
  // `uv python find` may report a venv shim, so follow it to the real binary.
  let dir = path.dirname(realpathSync(found));

  // Walk up to the runtime root rather than assuming a fixed depth: POSIX puts
  // the interpreter in <root>/bin/python3 but Windows puts it at
  // <root>\python.exe, so a fixed number of dirname() calls is right on one
  // platform and overshoots on the other.
  for (let up = 0; up < 3; up++) {
    const hasStdlib = existsSync(path.join(dir, "lib")) || existsSync(path.join(dir, "Lib"));
    const hasExe =
      existsSync(path.join(dir, "python.exe")) || existsSync(path.join(dir, "bin"));
    if (hasStdlib && hasExe) return dir;
    dir = path.dirname(dir);
  }
  throw new Error(
    `could not locate a standalone runtime root above ${realpathSync(found)}`,
  );
}

/**
 * Rewrite symlinks that escape the bundle so they point within it.
 *
 * The runtime stores these relative (`python -> python3.12`), but Node's cpSync
 * resolves them against the source and writes absolute targets. Those paths
 * exist only on the machine that built the bundle; everywhere else `bin/python3`
 * dangles and the app cannot start at all.
 */
function relinkInside(root, sourceRoot) {
  let fixed = 0;
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        const target = readlinkSync(full);
        if (!path.isAbsolute(target)) continue;
        // Only rewrite links aimed at the runtime we copied; anything else is
        // a genuine system reference and must be left alone.
        const inside = path.relative(sourceRoot, target);
        if (inside.startsWith("..") || path.isAbsolute(inside)) continue;
        const rebased = path.relative(path.dirname(full), path.join(root, inside));
        rmSync(full, { force: true });
        symlinkSync(rebased, full);
        fixed += 1;
      } else if (entry.isDirectory()) {
        walk(full);
      }
    }
  };
  walk(root);
  return fixed;
}

/** Drop caches, tests and headers that only bloat the installer. */
function prune(root) {
  let freed = 0;
  const junk = new Set(["__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo"]);
  const walk = (dir) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (junk.has(entry.name)) {
          try {
            freed += dirSize(full);
            rmSync(full, { recursive: true, force: true });
          } catch {
            /* best effort */
          }
          continue;
        }
        walk(full);
      } else if (entry.name.endsWith(".pyc") || entry.name.endsWith(".pyi")) {
        try {
          freed += statSync(full).size;
          rmSync(full, { force: true });
        } catch {
          /* best effort */
        }
      }
    }
  };
  const dirSize = (dir) => {
    let total = 0;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const f = path.join(dir, e.name);
      try {
        total += e.isDirectory() ? dirSize(f) : statSync(f).size;
      } catch {
        /* skip */
      }
    }
    return total;
  };
  walk(root);
  return freed;
}

function main() {
  const uvBin = uv();

  if (!existsSync(path.join(REPO, "web", "dist", "index.html"))) {
    throw new Error("build the web UI first:  cd web && pnpm install && pnpm build");
  }

  console.log("• copying the standalone interpreter");
  rmSync(RESOURCES, { recursive: true, force: true });
  mkdirSync(RESOURCES, { recursive: true });
  const source = standaloneRuntime(uvBin);
  cpSync(source, PY_DIR, { recursive: true, dereference: false });
  const relinked = relinkInside(PY_DIR, source);
  console.log(`  rebased ${relinked} symlink(s) to stay inside the bundle`);

  // PEP 668: uv marks its runtimes externally managed, which blocks installing
  // into them. Inside a bundle we are the environment, so the marker goes.
  for (const marker of ["lib/python3.12/EXTERNALLY-MANAGED", "Lib/EXTERNALLY-MANAGED"]) {
    rmSync(path.join(PY_DIR, marker), { force: true });
  }

  console.log("• installing the app and its dependencies into the bundle");
  const python = path.join(PY_DIR, WIN ? "python.exe" : "bin/python3");
  const sitePackages = WIN
    ? path.join(PY_DIR, "Lib", "site-packages")
    : path.join(PY_DIR, "lib", "python3.12", "site-packages");
  // --target, not --python. A copied interpreter still reports its build-time
  // prefix, so --python resolves back to the source runtime and uv would
  // install into the machine that built the bundle instead of the bundle.
  run(uvBin, ["pip", "install", "--target", sitePackages, "--python", python, REPO]);

  console.log("• copying the web UI");
  mkdirSync(APP_DIR, { recursive: true });
  cpSync(path.join(REPO, "web", "dist"), path.join(APP_DIR, "web", "dist"), { recursive: true });

  console.log("• pruning");
  const freed = prune(PY_DIR);
  console.log(`  removed ${(freed / 1e6).toFixed(0)} MB of caches and unused stdlib`);

  console.log("• verifying nothing escapes the bundle");
  const escapes = [];
  const scan = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        const resolved = path.resolve(path.dirname(full), readlinkSync(full));
        if (!resolved.startsWith(RESOURCES)) escapes.push(`${full} -> ${resolved}`);
      } else if (entry.isDirectory()) {
        scan(full);
      }
    }
  };
  scan(PY_DIR);
  if (escapes.length) {
    throw new Error(
      `${escapes.length} symlink(s) point outside the bundle and would dangle ` +
        `on any other machine:\n  ${escapes.slice(0, 5).join("\n  ")}`,
    );
  }
  console.log("  ok — every symlink resolves within the bundle");

  console.log("• verifying the bundle runs standalone");
  const probe = [
    "import sys, torch, docling, lancedb, rapidocr, server",
    "outside = [p for p in sys.path if p and not p.startswith(sys.prefix)]",
    "print(sys.prefix)",
    "print('OUTSIDE:' + repr(outside))",
    "print('SERVER:' + server.__file__)",
  ].join("; ");
  const out = execFileSync(python, ["-c", probe], {
    env: { ...process.env, PYTHONHOME: PY_DIR },
    encoding: "utf8",
  }).trim().split("\n");

  const [prefix, outside, serverPath] = out;
  if (!prefix.startsWith(PY_DIR)) {
    throw new Error(`bundle resolved its prefix outside itself: ${prefix}`);
  }
  if (outside !== "OUTSIDE:[]") {
    throw new Error(`bundle imports from outside itself: ${outside}`);
  }
  if (!serverPath.includes(PY_DIR)) {
    throw new Error(`server package came from outside the bundle: ${serverPath}`);
  }
  console.log("  ok — prefix, sys.path and every heavy import stay inside the bundle");
}

main();

"use strict";

/**
 * Owns the Python backend process: where to find it, how to start it, how to
 * know it is ready, and — the part that decides whether this app feels solid —
 * how to guarantee it is gone afterwards.
 */

const { spawn } = require("node:child_process");
const { existsSync } = require("node:fs");
const path = require("node:path");
const http = require("node:http");

// The backend prints this once, with the port it actually bound.
const READY_LINE = /^LOCALRAG_READY (\{.*\})$/m;
const START_TIMEOUT_MS = 120_000;

/**
 * Locate the Python interpreter and the directory `server` lives in.
 *
 * Packaged, both sit under resources/. In development we reuse the repo's own
 * .venv so the app can be run without building anything.
 */
function resolvePython() {
  const win = process.platform === "win32";
  const exe = win ? "python.exe" : "python3";
  const home = path.join(process.resourcesPath || "", "python");
  const bundled = path.join(home, win ? "" : "bin", exe);
  if (existsSync(bundled)) {
    // PYTHONHOME is not optional. The standalone runtime has its build-time
    // prefix compiled in, so a copied interpreter resolves its stdlib back to
    // wherever it was built - which exists on the build machine and nowhere
    // else. Without this the app works for whoever packaged it and fails for
    // every single user.
    const app = path.join(process.resourcesPath, "app");
    return { python: bundled, cwd: app, home, webDist: path.join(app, "web", "dist") };
  }
  const repo = path.resolve(__dirname, "..", "..");
  const dev = path.join(repo, ".venv", win ? "Scripts" : "bin", exe);
  // A dev venv resolves its own prefix correctly; forcing PYTHONHOME breaks it.
  if (existsSync(dev)) return { python: dev, cwd: repo, home: null, webDist: null };
  return null;
}

function get(url, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => {
        try {
          resolve({ status: res.statusCode, json: JSON.parse(body) });
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on("timeout", () => req.destroy(new Error("timeout")));
    req.on("error", reject);
  });
}

class Backend {
  constructor(options = {}) {
    this.proc = null;
    this.port = null;
    // Where documents, the index and the database live. Must be outside the
    // app bundle: on macOS that is read-only and on Windows it may sit under
    // Program Files, so writing there fails for every user who is not admin.
    this.dataDir = options.dataDir || null;
    this.apiKey = options.apiKey || "";
    this.onLog = options.onLog || (() => {});
    this.log = [];
    this.exited = null;
    this.onUnexpectedExit = null;
  }

  get origin() {
    return this.port ? `http://127.0.0.1:${this.port}` : null;
  }

  record(line) {
    this.log.push(line);
    // Bounded: a long-running ingest is chatty and this is only ever read to
    // show the user why a start-up failed.
    if (this.log.length > 400) this.log.splice(0, this.log.length - 400);
    this.onLog(line);
  }

  /** Start the backend and resolve once it reports the port it bound. */
  start() {
    const found = resolvePython();
    if (!found) {
      return Promise.reject(
        new Error(
          "Could not find the bundled Python runtime. The installation looks incomplete — reinstalling should fix it.",
        ),
      );
    }

    // stdin is a pipe we never write to. The backend watches it and exits the
    // moment it closes, which is what stops a force-quit of this app leaving a
    // multi-gigabyte Python process running with no window attached to it.
    this.proc = spawn(
      found.python,
      ["-m", "server.cli", "serve", "--port", "0", "--exit-with-parent"],
      {
        cwd: found.cwd,
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          PYTHONIOENCODING: "utf-8",
          ...(found.home ? { PYTHONHOME: found.home } : {}),
          ...(found.webDist ? { LOCALRAG_WEB_DIST: found.webDist } : {}),
          ...(this.dataDir ? { LOCALRAG_DATA_DIR: this.dataDir } : {}),
          ...(this.apiKey ? { GROQ_API_KEY: this.apiKey } : {}),
        },
        windowsHide: true,
      },
    );

    return new Promise((resolve, reject) => {
      let settled = false;
      let stdout = "";

      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        reject(new Error("The engine did not start within two minutes."));
      }, START_TIMEOUT_MS);

      const finish = (err, port) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        err ? reject(err) : resolve(port);
      };

      this.proc.stdout.on("data", (chunk) => {
        const text = chunk.toString();
        stdout += text;
        text.split(/\r?\n/).filter(Boolean).forEach((l) => this.record(l));
        const match = READY_LINE.exec(stdout);
        if (match) {
          try {
            this.port = JSON.parse(match[1]).port;
            finish(null, this.port);
          } catch (err) {
            finish(err);
          }
        }
      });

      this.proc.stderr.on("data", (chunk) =>
        chunk.toString().split(/\r?\n/).filter(Boolean).forEach((l) => this.record(l)),
      );

      this.proc.on("error", (err) =>
        finish(new Error(`Could not launch the engine: ${err.message}`)),
      );

      this.proc.on("exit", (code, signal) => {
        this.exited = { code, signal };
        finish(new Error(`The engine stopped unexpectedly (exit ${code ?? signal}).`));
        if (this.onUnexpectedExit) this.onUnexpectedExit(code, signal);
      });
    });
  }

  /** Poll until the API answers, so the window only opens on something live. */
  async waitForHealth({ attempts = 120, delayMs = 500 } = {}) {
    for (let i = 0; i < attempts; i++) {
      if (this.exited) throw new Error("The engine stopped while starting up.");
      try {
        const res = await get(`${this.origin}/api/health`);
        if (res.status === 200) return res.json;
      } catch {
        /* not listening yet */
      }
      await new Promise((r) => setTimeout(r, delayMs));
    }
    throw new Error("The engine started but never became reachable.");
  }

  health() {
    return get(`${this.origin}/api/health`, 5000).then((r) => r.json);
  }

  /**
   * Stop the backend.
   *
   * Closing stdin is the primary signal and the one the backend acts on within
   * a fraction of a second. SIGKILL / taskkill is the backstop for a process
   * already wedged in native OCR code, where nothing at the Python level runs.
   */
  stop() {
    const proc = this.proc;
    if (!proc || proc.exitCode !== null) return;
    this.proc = null;
    try {
      proc.stdin.end();
    } catch {
      /* already gone */
    }
    setTimeout(() => {
      if (proc.exitCode !== null) return;
      try {
        if (process.platform === "win32") {
          spawn("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { windowsHide: true });
        } else {
          proc.kill("SIGKILL");
        }
      } catch {
        /* nothing left to kill */
      }
    }, 2000).unref?.();
  }
}

module.exports = { Backend, resolvePython };

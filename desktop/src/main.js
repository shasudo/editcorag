"use strict";

const { app, BrowserWindow, dialog, ipcMain, shell, Menu } = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const http = require("node:http");
const { Backend } = require("./backend");

/**
 * The API key lives beside the user's data, not in the app bundle, and is the
 * one secret this app holds. Storing it per-install rather than compiling it in
 * means a shared build cannot leak one person's Groq quota to everyone else.
 */
function keyFile() {
  return path.join(app.getPath("userData"), "credentials.json");
}

function loadKey() {
  try {
    return JSON.parse(fs.readFileSync(keyFile(), "utf8")).groqApiKey || "";
  } catch {
    return "";
  }
}

function saveKeyToDisk(key) {
  try {
    fs.mkdirSync(path.dirname(keyFile()), { recursive: true });
    fs.writeFileSync(keyFile(), JSON.stringify({ groqApiKey: key }), { mode: 0o600 });
  } catch (err) {
    console.error("could not persist the API key:", err.message);
  }
}

let backend;
let setupWindow = null;
let mainWindow = null;
let quitting = false;

const isMac = process.platform === "darwin";

/** Push a status update to whichever window is currently showing setup. */
function report(state) {
  if (setupWindow && !setupWindow.isDestroyed()) {
    setupWindow.webContents.send("status", state);
  }
}

function createSetupWindow() {
  setupWindow = new BrowserWindow({
    width: 620,
    height: 460,
    resizable: false,
    show: false,
    title: "EditcoRAG",
    backgroundColor: "#0b0e14",
    webPreferences: { preload: path.join(__dirname, "preload.js"), sandbox: true },
  });
  setupWindow.removeMenu?.();
  setupWindow.loadFile(path.join(__dirname, "setup.html"));
  setupWindow.once("ready-to-show", () => setupWindow.show());
  setupWindow.on("closed", () => {
    setupWindow = null;
    // Closing the setup window before the app opened means the user gave up.
    if (!mainWindow && !quitting) app.quit();
  });
  return setupWindow;
}

function createMainWindow(url) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    show: false,
    title: "EditcoRAG",
    backgroundColor: "#0b0e14",
    webPreferences: { sandbox: true },
  });
  mainWindow.loadURL(url);
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    if (setupWindow && !setupWindow.isDestroyed()) setupWindow.close();
  });
  // Anything not our own origin belongs in the user's browser, not in here.
  mainWindow.webContents.setWindowOpenHandler(({ url: target }) => {
    shell.openExternal(target);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, target) => {
    if (!target.startsWith(backend.origin)) {
      event.preventDefault();
      shell.openExternal(target);
    }
  });
  mainWindow.on("closed", () => (mainWindow = null));
  return mainWindow;
}

/**
 * Consume one of the provisioning SSE streams, forwarding progress.
 *
 * `label` is what the user sees; the two streams differ only in which model
 * they are fetching.
 */
function streamSetup(pathname, label) {
  return new Promise((resolve, reject) => {
    const req = http.get(`${backend.origin}${pathname}`, (res) => {
      let buffer = "";
      res.on("data", (chunk) => {
        buffer += chunk.toString();
        let split;
        while ((split = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, split);
          buffer = buffer.slice(split + 2);
          const data = frame
            .split(/\r?\n/)
            .filter((l) => l.startsWith("data:"))
            .map((l) => l.slice(5).trim())
            .join("");
          if (!data) continue;
          let event;
          try {
            event = JSON.parse(data);
          } catch {
            continue;
          }
          if (event.stage === "error") return reject(new Error(event.detail));
          if (event.stage === "download") {
            const gb = event.total_bytes
              ? `${(event.done_bytes / 1e9).toFixed(2)} of ${(event.total_bytes / 1e9).toFixed(2)} GB`
              : event.detail || label;
            report({ phase: "downloading", percent: event.percent, detail: gb });
          } else if (event.stage === "start") {
            report({ phase: "downloading", percent: 0, detail: label });
          }
        }
      });
      res.on("end", resolve);
      res.on("error", reject);
    });
    req.on("error", reject);
  });
}

async function boot() {
  createSetupWindow();
  report({ phase: "starting", detail: "Starting the engine" });

  await backend.start();
  backend.onUnexpectedExit = (code) => {
    if (quitting) return;
    dialog.showErrorBox(
      "EditcoRAG stopped",
      `The engine exited unexpectedly (code ${code}).\n\n` +
        `Last messages:\n${backend.log.slice(-12).join("\n")}`,
    );
    app.quit();
  };

  report({ phase: "starting", detail: "Waiting for the engine" });
  let health = await backend.waitForHealth();

  if (!health.models?.ready) {
    report({
      phase: "downloading",
      percent: 0,
      detail: "First run: downloading the language models (about 7 GB)",
    });
    await streamSetup("/api/setup/stream", "Downloading the search models");
    health = await backend.health();
  }

  if (health.llm?.status !== "ready") {
    report({ phase: "llm", llm: health.llm });
    return; // the window waits for the user to fix it or continue anyway
  }

  createMainWindow(backend.origin);
}

async function retryLlm() {
  const health = await backend.health();
  if (health.llm?.status === "ready") createMainWindow(backend.origin);
  else report({ phase: "llm", llm: health.llm });
  return health.llm;
}

/** Hand a Groq key to the running backend, keep it if it works, then continue. */
async function submitKey(key) {
  const status = await new Promise((resolve, reject) => {
    const body = JSON.stringify({ key });
    const req = http.request(
      `${backend.origin}/api/setup/llm-key`,
      { method: "POST", headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) } },
      (res) => {
        let out = "";
        res.on("data", (c) => (out += c));
        res.on("end", () => {
          try {
            resolve(JSON.parse(out));
          } catch (err) {
            reject(err);
          }
        });
      },
    );
    req.on("error", reject);
    req.end(body);
  });

  if (status.status === "ready") {
    saveKeyToDisk(key);
    createMainWindow(backend.origin);
  } else {
    report({ phase: "llm", llm: status });
  }
  return status;
}

// A second launch should surface the window that already exists, not start a
// second engine fighting over the same database.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    const win = mainWindow || setupWindow;
    if (win) {
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  app.whenReady().then(() => {
    backend = new Backend({
      dataDir: app.getPath("userData"),
      apiKey: loadKey(),
      // Mirrored to the terminal so `npm start` shows engine output, and so a
      // packaged run leaves something in the system log to diagnose from.
      onLog: (line) => console.log("[engine]", line),
    });
    if (isMac) Menu.setApplicationMenu(Menu.buildFromTemplate([{ role: "appMenu" }, { role: "editMenu" }, { role: "windowMenu" }]));
    else Menu.setApplicationMenu(null);

    ipcMain.handle("retry-llm", retryLlm);
    ipcMain.handle("submit-key", (_e, key) => submitKey(key));
    ipcMain.handle("continue-anyway", () => createMainWindow(backend.origin));
    ipcMain.handle("open-external", (_e, url) => shell.openExternal(url));
    ipcMain.handle("get-logs", () => backend.log.slice(-200).join("\n"));

    boot().catch((err) => {
      report({ phase: "error", detail: err.message, log: backend.log.slice(-12).join("\n") });
    });

    app.on("activate", () => {
      if (!mainWindow && !setupWindow && backend.origin) createMainWindow(backend.origin);
    });
  });

  app.on("window-all-closed", () => {
    if (!isMac) app.quit();
  });

  // Every path out of the app funnels through here, so the engine is stopped
  // exactly once whether the user quits, closes the last window, or the OS
  // asks us to shut down.
  app.on("before-quit", () => (quitting = true));
  app.on("will-quit", () => backend.stop());
  process.on("exit", () => backend.stop());
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => {
      backend.stop();
      app.quit();
    });
  }
}

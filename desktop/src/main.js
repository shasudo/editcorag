"use strict";

const { app, BrowserWindow, dialog, ipcMain, shell, Menu } = require("electron");
const path = require("node:path");
const http = require("node:http");
const { Backend } = require("./backend");

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
  await backend.waitForHealth();

  // Straight to the home screen. Model downloads and the API key are asked for
  // in place, inside the app, so the only thing this splash covers is the few
  // seconds before the server can serve a page at all.
  createMainWindow(backend.origin);
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
      // Mirrored to the terminal so `npm start` shows engine output, and so a
      // packaged run leaves something in the system log to diagnose from.
      onLog: (line) => console.log("[engine]", line),
    });
    if (isMac) Menu.setApplicationMenu(Menu.buildFromTemplate([{ role: "appMenu" }, { role: "editMenu" }, { role: "windowMenu" }]));
    else Menu.setApplicationMenu(null);

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

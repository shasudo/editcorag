"use strict";

const { contextBridge, ipcRenderer } = require("electron");

// The setup window runs sandboxed with no node access; this is the entire
// surface it is allowed to touch.
contextBridge.exposeInMainWorld("shellApi", {
  onStatus: (handler) => ipcRenderer.on("status", (_event, state) => handler(state)),
  retryLlm: () => ipcRenderer.invoke("retry-llm"),
  submitKey: (key) => ipcRenderer.invoke("submit-key", key),
  continueAnyway: () => ipcRenderer.invoke("continue-anyway"),
  openExternal: (url) => ipcRenderer.invoke("open-external", url),
  getLogs: () => ipcRenderer.invoke("get-logs"),
});

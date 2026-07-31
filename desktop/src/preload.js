"use strict";

const { contextBridge, ipcRenderer } = require("electron");

// The splash runs sandboxed with no node access. It only ever reports progress
// and, if start-up fails, shows the engine log - the API key and model download
// are handled inside the app itself.
contextBridge.exposeInMainWorld("shellApi", {
  onStatus: (handler) => ipcRenderer.on("status", (_event, state) => handler(state)),
  openExternal: (url) => ipcRenderer.invoke("open-external", url),
  getLogs: () => ipcRenderer.invoke("get-logs"),
});

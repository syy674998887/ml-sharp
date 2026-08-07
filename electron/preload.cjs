const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("sharpDesktop", {
  isDesktop: true,
  platform: process.platform,
  windowControls: {
    minimize: () => ipcRenderer.invoke("window:minimize"),
    toggleMaximize: () => ipcRenderer.invoke("window:toggle-maximize"),
    close: () => ipcRenderer.invoke("window:close"),
    isMaximized: () => ipcRenderer.invoke("window:is-maximized"),
    onMaximizedChange: (callback) => {
      if (typeof callback !== "function") {
        return () => {};
      }

      const listener = (_event, isMaximized) => callback(Boolean(isMaximized));
      ipcRenderer.on("window:maximized", listener);
      return () => ipcRenderer.removeListener("window:maximized", listener);
    },
  },
});

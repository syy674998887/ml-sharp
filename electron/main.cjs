const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const fs = require("fs");
const http = require("http");
const net = require("net");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

const ROOT_DIR = path.resolve(__dirname, "..");
const BACKEND_HOST = "127.0.0.1";
const BACKEND_READY_TIMEOUT_MS = 15 * 60 * 1000;

let backendProcess = null;
let backendLogStream = null;
let backendStartError = null;
let splashWindow = null;
let mainWindow = null;
let isQuitting = false;

function commandExists(command) {
  const result = spawnSync(command, ["--version"], {
    cwd: ROOT_DIR,
    encoding: "utf8",
    shell: false,
    timeout: 5000,
    windowsHide: true,
  });
  return result.status === 0;
}

function getVenvPython() {
  const relativePath =
    process.platform === "win32"
      ? [".venv", "Scripts", "python.exe"]
      : [".venv", "bin", "python"];
  const pythonPath = path.join(ROOT_DIR, ...relativePath);
  if (!fs.existsSync(pythonPath)) {
    return null;
  }

  const result = spawnSync(pythonPath, ["--version"], {
    cwd: ROOT_DIR,
    encoding: "utf8",
    shell: false,
    timeout: 5000,
    windowsHide: true,
  });
  return result.status === 0 ? pythonPath : null;
}

function resolveBackendLaunch(port) {
  const webuiPath = path.join(ROOT_DIR, "webui.py");
  const backendArgs = [webuiPath, "--host", BACKEND_HOST, "--port", String(port)];
  const env = {
    ...process.env,
    PYTHONNOUSERSITE: "1",
    PYTHONUNBUFFERED: "1",
    SHARP_DESKTOP: "1",
  };

  if (process.env.SHARP_PYTHON) {
    return {
      command: process.env.SHARP_PYTHON,
      args: backendArgs,
      env,
      label: process.env.SHARP_PYTHON,
    };
  }

  if (commandExists("uv")) {
    return {
      command: "uv",
      args: ["run", "--python", "3.10", "python", ...backendArgs],
      env,
      label: "uv run --python 3.10 python webui.py",
    };
  }

  const venvPython = getVenvPython();
  if (venvPython) {
    return {
      command: venvPython,
      args: backendArgs,
      env,
      label: venvPython,
    };
  }

  return {
    command: process.platform === "win32" ? "python" : "python3",
    args: backendArgs,
    env,
    label: process.platform === "win32" ? "python webui.py" : "python3 webui.py",
  };
}

function allocatePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, BACKEND_HOST, () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : null;
      server.close(() => {
        if (port) {
          resolve(port);
        } else {
          reject(new Error("Unable to allocate a backend port."));
        }
      });
    });
  });
}

function createLogStream() {
  const logsDir = path.join(app.getPath("userData"), "logs");
  fs.mkdirSync(logsDir, { recursive: true });
  const stream = fs.createWriteStream(path.join(logsDir, "backend.log"), { flags: "a" });
  stream.on("error", (error) => {
    console.error("Unable to write backend log:", error);
  });
  return stream;
}

function appendBackendLog(chunk) {
  if (!backendLogStream || backendLogStream.destroyed || backendLogStream.writableEnded) {
    return;
  }
  backendLogStream.write(chunk);
}

function openExternalUrl(url) {
  try {
    const parsedUrl = new URL(url);
    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      return;
    }
    shell.openExternal(parsedUrl.toString()).catch((error) => {
      appendBackendLog(`[desktop] unable to open external URL: ${error.message}\n`);
    });
  } catch {
    // Ignore malformed or unsupported external links.
  }
}

function startBackend(port) {
  const launch = resolveBackendLaunch(port);
  backendStartError = null;
  backendLogStream = createLogStream();
  appendBackendLog(`\n\n[desktop] starting backend with: ${launch.label}\n`);
  appendBackendLog(`[desktop] cwd: ${ROOT_DIR}\n`);
  appendBackendLog(`[desktop] url: http://${BACKEND_HOST}:${port}\n`);

  backendProcess = spawn(launch.command, launch.args, {
    cwd: ROOT_DIR,
    env: launch.env,
    shell: false,
    windowsHide: true,
  });

  backendProcess.stdout.on("data", appendBackendLog);
  backendProcess.stderr.on("data", appendBackendLog);

  backendProcess.once("error", (error) => {
    backendStartError = error;
    appendBackendLog(`[desktop] backend spawn error: ${error.stack || error.message}\n`);
  });

  backendProcess.once("exit", (code, signal) => {
    appendBackendLog(`[desktop] backend exited: code=${code} signal=${signal}\n`);
    if (!isQuitting && mainWindow && !mainWindow.isDestroyed()) {
      dialog.showErrorBox(
        "ML-SHARP backend stopped",
        `The Python backend stopped unexpectedly.\n\nCode: ${code}\nSignal: ${
          signal || "none"
        }\n\nSee backend.log in the app data logs folder.`
      );
    }
  });
}

function requestStatus(port) {
  return new Promise((resolve) => {
    const req = http.get(
      {
        hostname: BACKEND_HOST,
        port,
        path: "/status",
        timeout: 2000,
      },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(port) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < BACKEND_READY_TIMEOUT_MS) {
    if (backendStartError) {
      throw new Error(`Unable to launch the Python backend: ${backendStartError.message}`);
    }
    if (backendProcess && backendProcess.exitCode !== null) {
      throw new Error(`Backend exited before it became ready. Exit code: ${backendProcess.exitCode}`);
    }
    if (await requestStatus(port)) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("Timed out while waiting for the Python backend to start.");
}

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 560,
    height: 360,
    backgroundColor: "#00000000",
    transparent: true,
    hasShadow: false,
    resizable: false,
    minimizable: false,
    maximizable: false,
    frame: false,
    show: false,
    title: "Starting ML-SHARP",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  splashWindow.removeMenu();
  splashWindow.loadFile(path.join(__dirname, "splash.html"));
  splashWindow.once("ready-to-show", () => splashWindow.show());
}

function createMainWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    frame: false,
    backgroundColor: "#101116",
    show: false,
    title: "ML-SHARP Desktop",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, "preload.cjs"),
    },
  });
  mainWindow.removeMenu();

  const sendWindowState = () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("window:maximized", mainWindow.isMaximized());
    }
  };

  mainWindow.on("maximize", sendWindowState);
  mainWindow.on("unmaximize", sendWindowState);

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    openExternalUrl(url);
    return { action: "deny" };
  });

  const backendOrigin = `http://${BACKEND_HOST}:${port}`;
  mainWindow.webContents.on("will-navigate", (event, url) => {
    let isBackendNavigation = false;
    try {
      isBackendNavigation = new URL(url).origin === backendOrigin;
    } catch {
      isBackendNavigation = false;
    }

    if (isBackendNavigation) return;

    event.preventDefault();
    openExternalUrl(url);
  });

  mainWindow.once("ready-to-show", () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
    }
    mainWindow.show();
  });

  mainWindow.loadURL(`http://${BACKEND_HOST}:${port}/`).catch((error) => {
    if (isQuitting || !mainWindow || mainWindow.isDestroyed()) {
      return;
    }
    appendBackendLog(`[desktop] main window load failed: ${error.stack || error.message}\n`);
    dialog.showErrorBox(
      "Unable to load ML-SHARP",
      `${error.message}\n\nSee backend.log in the app data logs folder.`
    );
    app.quit();
  });
}

function getTrustedIpcWindow(event) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return null;
  }
  if (
    event.sender !== mainWindow.webContents ||
    event.senderFrame !== mainWindow.webContents.mainFrame
  ) {
    return null;
  }
  return mainWindow;
}

ipcMain.handle("window:minimize", (event) => {
  const window = getTrustedIpcWindow(event);
  if (window) {
    window.minimize();
  }
});

ipcMain.handle("window:toggle-maximize", (event) => {
  const window = getTrustedIpcWindow(event);
  if (!window) {
    return false;
  }
  if (window.isMaximized()) {
    window.unmaximize();
    return false;
  }
  window.maximize();
  return true;
});

ipcMain.handle("window:close", (event) => {
  const window = getTrustedIpcWindow(event);
  if (window) {
    window.close();
  }
});

ipcMain.handle("window:is-maximized", (event) => {
  const window = getTrustedIpcWindow(event);
  return window ? window.isMaximized() : false;
});

function stopBackend() {
  if (!backendProcess || backendProcess.exitCode !== null) {
    return;
  }

  appendBackendLog("[desktop] stopping backend\n");
  if (process.platform === "win32") {
    spawnSync("taskkill", ["/pid", String(backendProcess.pid), "/T", "/F"], {
      windowsHide: true,
    });
  } else {
    backendProcess.kill("SIGTERM");
    setTimeout(() => {
      if (backendProcess && backendProcess.exitCode === null) {
        backendProcess.kill("SIGKILL");
      }
    }, 3000).unref();
  }
}

async function bootstrap() {
  createSplashWindow();
  const port = await allocatePort();
  startBackend(port);
  await waitForBackend(port);
  createMainWindow(port);
}

app.whenReady().then(() => {
  bootstrap().catch((error) => {
    appendBackendLog(`[desktop] startup failed: ${error.stack || error.message}\n`);
    dialog.showErrorBox(
      "Unable to start ML-SHARP",
      `${error.message}\n\nMake sure uv or Python 3.10 is installed, then try again.`
    );
    app.quit();
  });
});

app.on("before-quit", () => {
  isQuitting = true;
  stopBackend();
  if (backendLogStream) {
    backendLogStream.end();
  }
});

app.on("window-all-closed", () => {
  app.quit();
});

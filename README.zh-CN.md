# ML-SHARP Desktop

[English](README.md) | **简体中文**

[![上游项目](https://img.shields.io/badge/upstream-apple%2Fml--sharp-555)](https://github.com/apple/ml-sharp)
[![项目主页](https://img.shields.io/badge/SHARP-Project_Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

本仓库是 Apple [ml-sharp](https://github.com/apple/ml-sharp) 的社区 fork。项目保留了
原始 SHARP 模型和命令行工具，并在此基础上增加 Electron 桌面应用、交互式 Gaussian
Splat 查看器、图像与视频工作流，以及本地 Python 后端。

这是一个独立的社区项目，不属于 Apple，也未获得 Apple 的关联或背书。

> [!IMPORTANT]
> SHARP 模型仅授权用于**非商业研究用途**。下载或使用模型权重前，请阅读
> [LICENSE_MODEL](LICENSE_MODEL)。源代码许可证与模型许可证相互独立。

![ML-SHARP Desktop 创建与资源库界面](webui_static/Screenshot%201.png)

![ML-SHARP Desktop Gaussian Splat 查看器](webui_static/Screenshot%202.png)

## 本 fork 的新增内容

- Electron 桌面界面，可自动启动和停止本地 Python 后端。
- 支持单图、多图、摄像头、OBS Virtual Camera 和视频输入。
- 基于 Three.js 与 Spark 的交互式 `.ply` Gaussian Splat 查看器。
- 视频转 PLY 序列及序列播放功能。
- 可选的 CUDA/gsplat 并排立体（Side-by-Side，SBS）3D 视频渲染。
- 输出资源库，可重新打开、下载和删除生成文件。
- 可选的本地 Codex/imagegen 图像扩展流程，可在 SHARP 重建前扩展图像边缘。

## 快速开始

推荐环境为 Windows 10/11、Node.js 22.12 或更高版本、uv，以及 NVIDIA GPU。
Linux 支持仍处于实验阶段；本 fork 尚未验证 macOS 和纯 CPU 安装流程。SBS 渲染需要
CUDA。

### 1. 安装前置工具

安装 [Node.js](https://nodejs.org/) 22.12 或更高版本。Node.js 同时提供 `npm`，用于
安装依赖和启动 Electron。本项目使用
[uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 3.10 和锁定的
Python 依赖。

安装 Node.js 后，打开 PowerShell，运行 uv 官方 Windows 安装命令：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

该命令会临时允许安装脚本运行，从 `astral.sh` 下载脚本，并为当前用户安装 uv。
安装完成后关闭并重新打开 PowerShell，然后确认以下三个命令均可使用：

```powershell
node --version
npm --version
uv --version
```

### 2. 克隆并启动桌面应用

```powershell
git clone https://github.com/syy674998887/ml-sharp.git
cd ml-sharp
npm ci
npm start
```

`npm run dev` 是目前等效的开发启动命令。

本仓库当前提供源代码，尚未提供预构建安装程序。Electron 会自动启动后端。首次启动时，
uv 会创建 `.venv` 并安装锁定的 Python 依赖；第一次运行 SHARP 生成任务时会自动下载
模型检查点。生成文件保存在 `output/` 中。

### 高级用法：仅运行本地后端

如需开发或排查后端问题，而不启动 Electron，可运行：

```powershell
uv run --python 3.10 python webui.py
```

后端默认只监听本机地址，并会在启动时输出访问地址。它不提供多用户身份验证，因此不要将
其暴露到不受信任的网络。

### 在本 fork 环境中使用原始 CLI

```powershell
uv run sharp --help
uv run sharp predict -i path\to\input -o path\to\output
```

渲染和 SBS 功能还需要可选的 `gsplat` 依赖，以及兼容的 CUDA/PyTorch 环境：

```powershell
uv sync --python 3.10 --extra render
```

`gsplat` 可能需要兼容的预编译 wheel，或在本机编译 CUDA/C++ 扩展。基础的图像转 PLY
流程不依赖此可选组件。

## 可选的图像扩展控制

只有在本机已安装并登录支持图像生成的 Codex CLI 时，图像扩展面板才可使用。启动图像
扩展任务前，应用会检查 Codex 的安装和登录状态；如果不可用，会显示相应的处理提示。

- **Guard Band (%)**：要求在选中边缘扩展大约指定百分比的新图像区域。
- **Protect Original Image**：要求生成式编辑器尽量避免改动原图已有区域。这是一条模型
  指令，并不保证像素级锁定。
- **Image Count**：设置需要生成的候选图像数量。
- **Parallel Agent**：设置可并发运行的本地 Codex worker 数量。
- **Optional User Prompt**：添加针对当前场景的补充要求。

图像生成的输入、输出和诊断信息会保存在 `output/imagegen/jobs/` 中，直到用户主动删除。

## 致谢

SHARP、模型和原始 CLI 由 Apple 上游 README 及论文中列出的研究团队开发。本仓库的
WebUI 与视频/SBS 工作流基础来自
[iVideoGameBoss](https://github.com/iVideoGameBoss/ml-sharp) 的社区工作；其贡献和原始
提交作者信息均保留在本仓库的 Git 历史中。

## Fork 状态

本项目保留了上游原始 SHARP 模型和 CLI。桌面应用、本地后端、图像扩展和长视频工作流
仍在持续开发中。

本仓库适合公开获取源代码和测试，但 fork 新增的应用层目前仍标记为 **Beta**，而不是
Stable/1.0。原因包括：尚未通过自动化流程验证从全新 clone 到安装运行的完整路径；CI
尚未覆盖所支持的 GPU 与平台组合；可选 CUDA/gsplat 的安装依赖具体环境；长视频任务
可能占用大量 RAM、VRAM 和存储空间。

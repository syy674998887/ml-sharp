# ML-SHARP Desktop

**English** | [简体中文](README.zh-CN.md)

[![Upstream](https://img.shields.io/badge/upstream-apple%2Fml--sharp-555)](https://github.com/apple/ml-sharp)
[![Project Page](https://img.shields.io/badge/SHARP-Project_Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This repository is a community fork of Apple's
[ml-sharp](https://github.com/apple/ml-sharp). It retains the original SHARP model and
CLI while adding an Electron desktop app, an interactive Gaussian Splat viewer, image
and video workflows, and a local Python backend.

This is an independent community project. It is not an Apple product and is not
affiliated with or endorsed by Apple.

> [!IMPORTANT]
> The SHARP model is licensed for **non-commercial research purposes only**. Read
> [LICENSE_MODEL](LICENSE_MODEL) before downloading or using the model weights. The
> source-code license and model license are separate.

![ML-SHARP Desktop create and library view](webui_static/Screenshot%201.png)

![ML-SHARP Desktop Gaussian Splat viewer](webui_static/Screenshot%202.png)

## Additions in this fork

- Electron desktop interface that starts and stops its local Python backend automatically.
- Image, multi-image, webcam, OBS Virtual Camera, and video input workflows.
- Interactive `.ply` Gaussian Splat viewer powered by Three.js and Spark.
- Video-to-PLY-sequence processing and playback.
- Optional CUDA/gsplat Side-by-Side (SBS) 3D movie rendering.
- Output library for reopening, downloading, and deleting generated files.
- Optional local Codex/imagegen workflow for extending image borders before SHARP
  reconstruction.

## Quick start

The recommended setup is Windows 10/11 with Node.js 22.12 or newer, uv, and an NVIDIA GPU.
Linux support is experimental. macOS and CPU-only installation paths have not yet been
validated for this fork. SBS rendering requires CUDA.

### 1. Install the prerequisites

Install [Node.js](https://nodejs.org/) 22.12 or newer; it also provides the `npm`
command used to install and launch Electron. This project uses
[uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python 3.10
and the locked Python dependencies.

After installing Node.js, open PowerShell and run the official uv Windows installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

This command temporarily permits the installer script to run, downloads it from
`astral.sh`, and installs uv for the current user. Close and reopen PowerShell when
it finishes, then verify that all three commands are available:

```powershell
node --version
npm --version
uv --version
```

### 2. Clone and launch the desktop app

```powershell
git clone https://github.com/syy674998887/ml-sharp.git
cd ml-sharp
npm ci
npm start
```

`npm run dev` is an equivalent development launch command.

This repository currently provides source code rather than a prebuilt installer.
Electron starts the backend automatically. On first launch, uv creates `.venv` and
installs the locked Python dependencies; the first SHARP generation downloads the model
checkpoint. Generated files are stored in `output/`.

### Advanced: run only the local backend

For backend development or troubleshooting without Electron:

```powershell
uv run --python 3.10 python webui.py
```

The backend listens only on the local machine by default and prints its address when it
starts. It has no multi-user authentication, so do not expose it to an untrusted network.

### Original CLI through this fork's environment

```powershell
uv run sharp --help
uv run sharp predict -i path\to\input -o path\to\output
```

Rendering and SBS features additionally require the optional `gsplat` dependency and a
compatible CUDA/PyTorch build:

```powershell
uv sync --python 3.10 --extra render
```

`gsplat` may require a compatible prebuilt wheel or a local CUDA/C++ build. The base
image-to-PLY workflow can be used without this optional extra.

## Optional image extension controls

The image-extension panel is available only when a local Codex CLI with image generation
is installed and authenticated. Before an image-extension job starts, the app checks the
local installation and sign-in state and shows corrective guidance when either is unavailable.

- **Guard Band (%)** requests approximately that much new image area on the selected
  edges.
- **Protect Original Image** asks the generative editor to avoid unnecessary changes to
  the existing image area. It is a model instruction, not a guaranteed pixel lock.
- **Image Count** selects the number of requested candidates.
- **Parallel Agent** selects how many local Codex workers can run concurrently.
- **Optional User Prompt** adds scene-specific instructions.

Image-generation inputs, outputs, and diagnostics are stored under
`output/imagegen/jobs/` until deleted.

## Credits

SHARP, the model, and the original CLI were created by the Apple research team named
in the original README and paper below. The WebUI and video/SBS workflow foundation in
this repository builds on prior community work by
[iVideoGameBoss](https://github.com/iVideoGameBoss/ml-sharp). Their contribution and
original commit authorship are retained in this repository's Git history.

## Fork status

The original SHARP model and CLI are retained from upstream. The desktop, local backend,
image-extension, and long-video workflows are under active development.

This repository is suitable for public source use and testing, but the fork-specific
application layer is currently marked **Beta** rather than Stable/1.0 because clean-clone
installation is not yet tested automatically, supported GPU/platform combinations are
not yet covered by CI, optional CUDA/gsplat installation is environment-dependent, and
long video jobs can consume substantial RAM, VRAM, and storage.

> [!NOTE]
> Use the **Quick start** above to install this fork. The section below retains Apple's
> upstream README for reference; its original installation commands do not apply to
> this fork.

---

# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

### Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

### Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

#### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

### Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

### Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

### Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

### License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.

"""WebUI for ml-sharp 3D Gaussian Splat prediction.

A simple Flask-based web interface for uploading images and generating 3DGS PLY files.
"""

from __future__ import annotations

import logging
import os
import json
import math
import re
import subprocess
import tempfile
import uuid
import shutil
import threading
import time
from pathlib import Path
import urllib.parse

import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg
import pillow_heif
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_file

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

pillow_heif.register_heif_opener(thumbnails=False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger(__name__)

logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Flask app - use absolute paths for static and template folders
_base_dir = Path(__file__).parent.absolute()
app = Flask(
    __name__,
    static_folder=str(_base_dir / "webui_static"),
    static_url_path="/static",
    template_folder=str(_base_dir / "webui_templates")
)

# Global model cache
_model_cache = {"predictor": None, "device": None}
_model_load_lock = threading.Lock()
_inference_lock = threading.Lock()
_job_prefix_lock = threading.Lock()

# Global job store for async video processing
_active_jobs = {}

# Global job store for local Codex image generation/editing jobs
_imagegen_jobs = {}
_imagegen_semaphore = threading.Semaphore(1)
TERMINAL_JOB_STATUSES = {"done", "stopped", "error", "timeout"}

# Output directory for generated files. Keep it anchored to the repository so
# desktop launches from another working directory cannot scatter output files.
OUTPUT_DIR = _base_dir / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGEGEN_DIR = OUTPUT_DIR / "imagegen" / "jobs"
IMAGEGEN_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAIL_DIR = OUTPUT_DIR / "thumbnails"
THUMBNAIL_DIR.mkdir(exist_ok=True)
THUMBNAIL_SIZE = (360, 240)
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif", ".webp"}
SUPPORTED_IMAGEGEN_ASPECTS = {"keep", "16:9", "4:3", "1:1", "9:16"}
IMAGEGEN_EDGE_ORDER = ("top", "right", "bottom", "left")
IMAGEGEN_EDGE_LABELS = {
    "top": "top edge",
    "right": "right edge",
    "bottom": "bottom edge",
    "left": "left edge",
}
IMAGEGEN_LEGACY_DIRECTIONS = {
    "all": ("top", "right", "bottom", "left"),
    "horizontal": ("right", "left"),
    "vertical": ("top", "bottom"),
    "top": ("top",),
    "bottom": ("bottom",),
    "left": ("left",),
    "right": ("right",),
}
IMAGEGEN_PADDING_RANGE = (1, 50)
IMAGEGEN_VARIANT_RANGE = (1, 8)
IMAGEGEN_PARALLEL_AGENT_RANGE = (1, IMAGEGEN_VARIANT_RANGE[1])
IMAGEGEN_TIMEOUT_SECONDS = 10 * 60
SEQUENCE_METADATA_FILENAME = "sequence.json"

# Model URL
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def sanitize_output_stem(filename: str, fallback: str = "output") -> str:
    """Return a filesystem-safe, bounded stem for a user-supplied filename."""
    leaf_name = re.split(r"[/\\]", filename or "")[-1]
    stem = Path(leaf_name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip(" ._")
    stem = stem[:120].rstrip(" .") or fallback
    if stem.split(".", 1)[0].upper() in WINDOWS_RESERVED_STEMS:
        stem = f"_{stem}"
    return stem


def ensure_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize an image/video frame to a contiguous three-channel RGB array."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] in {1, 2}:
        frame = np.repeat(frame[:, :, :1], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    return np.ascontiguousarray(frame)


def create_gsplat_renderer(color_space: str):
    """Create the optional gsplat renderer for SBS/video rendering."""
    try:
        from sharp.utils import gsplat
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SBS Movie and SBS Preview require the optional gsplat dependency. "
            "Install it with `uv sync --python 3.10 --extra render`."
        ) from exc

    return gsplat.GSplatRenderer(color_space=color_space)


def clear_finished_memory_jobs() -> dict:
    """Clear completed job records from memory without touching disk outputs."""
    count_before = len(_active_jobs) + len(_imagegen_jobs)

    for job_id, job in list(_active_jobs.items()):
        if job.get("status") in TERMINAL_JOB_STATUSES:
            _active_jobs.pop(job_id, None)

    for job_id, job in list(_imagegen_jobs.items()):
        if job.get("status") in TERMINAL_JOB_STATUSES:
            _imagegen_jobs.pop(job_id, None)

    active_total = len(_active_jobs) + len(_imagegen_jobs)

    return {
        "cleared_total": count_before - active_total,
        "active_total": active_total,
    }


def get_device() -> torch.device:
    """Get the best available device."""
    if _model_cache["device"] is not None:
        return _model_cache["device"]

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _load_predictor() -> tuple[RGBGaussianPredictor, torch.device]:
    """Load the Gaussian predictor model if it is not already cached."""
    if _model_cache["predictor"] is None:
        target_device = torch.device("cpu")
        
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
            try:
                gpu_name = torch.cuda.get_device_name(0)
                LOGGER.info(f"CUDA GPU detected: {gpu_name}")
            except Exception:
                LOGGER.info("CUDA GPU detected (name unknown)")
        elif torch.mps.is_available():
            target_device = torch.device("mps")
            LOGGER.info("Apple MPS acceleration detected.")
        else:
            LOGGER.info("No active GPU detected. Using CPU.")

        LOGGER.info(f"Targeting device for inference: {target_device}")

        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL,
            progress=True,
            map_location="cpu",
        )

        LOGGER.info("Initializing predictor...")
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        
        final_device = torch.device("cpu")
        if target_device.type != "cpu":
            try:
                LOGGER.info(f"Moving model to {target_device}...")
                predictor.to(target_device)
                dummy = torch.zeros(1).to(target_device)
                del dummy
                final_device = target_device
            except RuntimeError as e:
                LOGGER.warning(f"Failed to initialize on {target_device}: {e}.")
                LOGGER.warning("Falling back to CPU mode.")
                predictor.to("cpu")
                final_device = torch.device("cpu")
        else:
            predictor.to("cpu")

        _model_cache["predictor"] = predictor
        _model_cache["device"] = final_device
        LOGGER.info(f"Model successfully loaded and running on: {final_device}")

    return _model_cache["predictor"], _model_cache["device"]


def get_predictor() -> tuple[RGBGaussianPredictor, torch.device]:
    """Get the cached predictor, serializing first-time model initialization."""
    if _model_cache["predictor"] is not None:
        return _model_cache["predictor"], _model_cache["device"]

    with _model_load_lock:
        return _load_predictor()


def get_next_job_prefix() -> str:
    """Scan output dir and return the next available numeric prefix."""
    try:
        max_idx = 0
        for item in OUTPUT_DIR.iterdir():
            prefix, separator, _ = item.name.partition("_")
            if separator and prefix.isdigit():
                max_idx = max(max_idx, int(prefix))
        return f"{max_idx + 1:03d}"
    except OSError as exc:
        raise RuntimeError("Unable to allocate an output job number.") from exc


def reserve_numbered_output_dir(original_stem: str, suffix: str) -> tuple[str, Path]:
    """Atomically reserve a numbered output directory across threads and processes."""
    with _job_prefix_lock:
        for _ in range(1000):
            job_prefix = get_next_job_prefix()
            reservation_dir = OUTPUT_DIR / f"{job_prefix}_.reserve"
            work_dir = OUTPUT_DIR / f"{job_prefix}_{original_stem}_{suffix}"

            try:
                reservation_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue

            try:
                reservation_dir.rename(work_dir)
            except OSError:
                try:
                    reservation_dir.rmdir()
                except OSError:
                    pass
                raise

            return job_prefix, work_dir

    raise RuntimeError("Unable to reserve a unique output directory.")


def natural_path_sort_key(path: Path) -> tuple:
    """Return a case-insensitive key that sorts embedded numbers numerically."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
        if part
    )


def get_sequence_fps(folder_path: Path, default: float = 30.0) -> float:
    """Read a generated sequence's persisted FPS, falling back for legacy folders."""
    metadata_path = folder_path / SEQUENCE_METADATA_FILENAME
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = float(data.get("fps", default))
        if math.isfinite(fps) and fps > 0:
            return fps
    except (OSError, TypeError, ValueError):
        pass
    return default


def parse_finite_form_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Parse and range-check a finite floating-point form value."""
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name.replace('_', ' ')} value.") from exc

    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"{name.replace('_', ' ').capitalize()} must be between {minimum} and {maximum}."
        )
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is inside parent, compatible with Python 3.10."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_file(filename: str) -> Path | None:
    """Resolve a user-facing output path without allowing traversal outside output."""
    file_path = (OUTPUT_DIR / filename).resolve()
    output_root = OUTPUT_DIR.resolve()
    if not _is_relative_to(file_path, output_root):
        return None
    return file_path


def get_thumbnail_path_for_ply(ply_path: Path) -> Path:
    """Return the thumbnail path that corresponds to a PLY inside OUTPUT_DIR."""
    relative_path = ply_path.resolve().relative_to(OUTPUT_DIR.resolve())
    return THUMBNAIL_DIR / relative_path.with_suffix(".jpg")


def save_thumbnail(image: np.ndarray, ply_path: Path) -> Path | None:
    """Save a small JPEG thumbnail next to the generated PLY metadata."""
    try:
        thumb_path = get_thumbnail_path_for_ply(ply_path)
        thumb_path.parent.mkdir(parents=True, exist_ok=True)

        image_np = image
        if image_np.ndim == 2:
            image_np = np.dstack((image_np, image_np, image_np))
        if image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        image_pil = Image.fromarray(image_np)
        image_pil.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        image_pil.save(thumb_path, "JPEG", quality=84, optimize=True)
        return thumb_path
    except Exception:
        LOGGER.exception("Failed to save thumbnail for %s", ply_path)
        return None


def get_thumbnail_url_for_ply(ply_path: Path) -> str | None:
    """Return the thumbnail URL for a generated PLY if one exists."""
    thumbnail_path = get_thumbnail_path_for_ply(ply_path)
    if not thumbnail_path.exists():
        return None

    thumbnail_relative = thumbnail_path.relative_to(THUMBNAIL_DIR).as_posix()
    return f"/thumbnail/{urllib.parse.quote(thumbnail_relative)}"


def get_single_output_item(ply_path: Path) -> dict:
    """Return a Library item for one generated PLY file."""
    stat = ply_path.stat()
    relative_path = ply_path.relative_to(OUTPUT_DIR)
    relative_url_path = relative_path.as_posix()

    return {
        "type": "ply",
        "filename": relative_url_path,
        "name": ply_path.name,
        "folder": "" if relative_path.parent == Path(".") else relative_path.parent.as_posix(),
        "size_bytes": stat.st_size,
        "modified": stat.st_mtime,
        "view_url": f"/ply/{urllib.parse.quote(relative_url_path)}",
        "download_url": f"/download/{urllib.parse.quote(relative_url_path)}",
        "thumbnail_url": get_thumbnail_url_for_ply(ply_path),
    }


def get_video_output_item(folder_path: Path, ply_files: list[Path]) -> dict:
    """Return a Library item for a generated PLY Sequence folder."""
    relative_folder = folder_path.relative_to(OUTPUT_DIR)
    folder_parent = "" if relative_folder.parent == Path(".") else relative_folder.parent.as_posix()
    size_bytes = sum(path.stat().st_size for path in ply_files)
    modified = max(path.stat().st_mtime for path in ply_files)
    thumbnail_url = None
    for ply_path in ply_files:
        thumbnail_url = get_thumbnail_url_for_ply(ply_path)
        if thumbnail_url:
            break

    return {
        "type": "video",
        "filename": relative_folder.as_posix(),
        "name": folder_path.name,
        "folder": folder_parent,
        "size_bytes": size_bytes,
        "modified": modified,
        "frame_count": len(ply_files),
        "fps": get_sequence_fps(folder_path),
        "base_url": "",
        "ply_files": [
            f"/ply/{urllib.parse.quote(path.relative_to(OUTPUT_DIR).as_posix())}"
            for path in ply_files
        ],
        "thumbnail_url": thumbnail_url,
    }


def get_movie_output_item(movie_path: Path) -> dict:
    """Return a Library item for a generated SBS Movie."""
    stat = movie_path.stat()
    relative_path = movie_path.relative_to(OUTPUT_DIR)
    relative_url_path = relative_path.as_posix()
    frame_folder = None
    frame_folder_path = get_sbs_frame_folder_for_movie(movie_path)
    size_bytes = stat.st_size

    if frame_folder_path and frame_folder_path.exists():
        try:
            frame_folder = frame_folder_path.relative_to(OUTPUT_DIR).as_posix()
            size_bytes += sum(path.stat().st_size for path in frame_folder_path.rglob("*") if path.is_file())
        except Exception:
            LOGGER.exception("Failed to inspect SBS frame folder %s", frame_folder_path)

    return {
        "type": "movie",
        "filename": relative_url_path,
        "name": movie_path.name,
        "folder": "" if relative_path.parent == Path(".") else relative_path.parent.as_posix(),
        "size_bytes": size_bytes,
        "modified": stat.st_mtime,
        "download_url": f"/download/{urllib.parse.quote(relative_url_path)}",
        "frame_folder": frame_folder,
    }


def get_active_output_paths() -> set[str]:
    """Return output paths currently owned by non-terminal background jobs."""
    return {
        str(job.get("output_path", "")).replace("\\", "/").strip("/")
        for job in list(_active_jobs.values())
        if job.get("status") not in TERMINAL_JOB_STATUSES and job.get("output_path")
    }


def get_output_items(limit: int = 60) -> list[dict]:
    """Return recent generated files and generated PLY frame sequences."""
    items = []
    grouped_dirs = set()
    output_root = OUTPUT_DIR.resolve()
    thumbnail_root = THUMBNAIL_DIR.resolve()
    active_output_paths = get_active_output_paths()

    for folder_path in OUTPUT_DIR.rglob("*"):
        try:
            if not folder_path.is_dir():
                continue

            resolved_folder = folder_path.resolve()
            if _is_relative_to(resolved_folder, thumbnail_root):
                continue
            if not _is_relative_to(resolved_folder, output_root):
                continue
            relative_folder = folder_path.relative_to(OUTPUT_DIR).as_posix()
            if relative_folder in active_output_paths:
                continue

            ply_files = sorted(folder_path.glob("*.ply"), key=natural_path_sort_key)
            if not ply_files:
                continue

            is_sequence_folder = folder_path.name.endswith("_plys")
            if not is_sequence_folder:
                continue

            items.append(get_video_output_item(folder_path, ply_files))
            grouped_dirs.add(resolved_folder)
        except Exception:
            LOGGER.exception("Unable to inspect Library output directory %s", folder_path)

    for ply_path in OUTPUT_DIR.rglob("*.ply"):
        try:
            resolved_ply = ply_path.resolve()
            if _is_relative_to(resolved_ply, thumbnail_root):
                continue
            if not _is_relative_to(resolved_ply, output_root):
                continue
            relative_ply = ply_path.relative_to(OUTPUT_DIR).as_posix()
            if any(
                relative_ply == active_path or relative_ply.startswith(f"{active_path}/")
                for active_path in active_output_paths
            ):
                continue
            if ply_path.parent.resolve() in grouped_dirs:
                continue

            items.append(get_single_output_item(ply_path))
        except Exception:
            LOGGER.exception("Failed to inspect output file %s", ply_path)

    for movie_path in OUTPUT_DIR.glob("*.mp4"):
        try:
            resolved_movie = movie_path.resolve()
            if not _is_relative_to(resolved_movie, output_root):
                continue
            if movie_path.relative_to(OUTPUT_DIR).as_posix() in active_output_paths:
                continue
            items.append(get_movie_output_item(movie_path))
        except Exception:
            LOGGER.exception("Failed to inspect output movie %s", movie_path)

    items.sort(key=lambda item: item["modified"], reverse=True)
    return items[:limit]


def prune_empty_parents(path: Path, stop_at: Path) -> None:
    """Remove empty parent folders up to but not including stop_at."""
    stop_at = stop_at.resolve()
    current = path.resolve()
    while _is_relative_to(current, stop_at) and current != stop_at:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def safe_unlink(path: Path, parent: Path) -> bool:
    """Delete a file only when it resolves inside parent."""
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if not _is_relative_to(resolved_path, resolved_parent) or not resolved_path.is_file():
        return False
    resolved_path.unlink()
    prune_empty_parents(resolved_path.parent, resolved_parent)
    return True


def safe_rmtree(path: Path, parent: Path) -> bool:
    """Delete a directory tree only when it resolves inside parent."""
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if (
        resolved_path == resolved_parent
        or not _is_relative_to(resolved_path, resolved_parent)
        or not resolved_path.is_dir()
    ):
        return False
    shutil.rmtree(resolved_path)
    prune_empty_parents(resolved_path.parent, resolved_parent)
    return True


def get_imagegen_job_dir(job_id: str) -> Path | None:
    """Resolve an imagegen job folder without allowing traversal."""
    if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in job_id):
        return None

    job_dir = (IMAGEGEN_DIR / job_id).resolve()
    imagegen_root = IMAGEGEN_DIR.resolve()
    if not _is_relative_to(job_dir, imagegen_root):
        return None
    return job_dir


def stop_process_tree(process: subprocess.Popen | None) -> None:
    """Terminate a process tree for long-running local jobs."""
    if process is None or process.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/pid", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("Process %s did not exit after being killed", process.pid)
        else:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("Process %s did not exit after being killed", process.pid)
    except Exception:
        LOGGER.exception("Failed to stop process tree for pid %s", getattr(process, "pid", None))


def get_codex_subprocess_env() -> dict:
    """Return an environment that can resolve Windows app execution aliases."""
    env = os.environ.copy()
    if os.name == "nt":
        local_app_data = env.get("LOCALAPPDATA")
        if local_app_data:
            windows_apps = str(Path(local_app_data) / "Microsoft" / "WindowsApps")
            path_parts = env.get("PATH", "").split(os.pathsep)
            if windows_apps not in path_parts:
                env["PATH"] = windows_apps + os.pathsep + env.get("PATH", "")
    return env


def get_codex_command_base() -> list[str]:
    """Return the Codex command base, allowing packaged apps to override it."""
    explicit_path = os.environ.get("SHARP_CODEX", "").strip().strip('"')
    if explicit_path:
        return [explicit_path]
    return ["codex"]


def run_codex_cli(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run Codex CLI with Windows app-alias compatible process creation."""
    command = get_codex_command_base() + args
    env = kwargs.pop("env", get_codex_subprocess_env())
    try:
        return subprocess.run(command, env=env, **kwargs)
    except OSError:
        if os.name != "nt":
            raise
        return subprocess.run(
            subprocess.list2cmdline(command),
            shell=True,
            env=env,
            **kwargs,
        )


def popen_codex_cli(args: list[str], **kwargs) -> subprocess.Popen:
    """Start Codex CLI with Windows app-alias compatible process creation."""
    command = get_codex_command_base() + args
    env = kwargs.pop("env", get_codex_subprocess_env())
    try:
        return subprocess.Popen(command, env=env, **kwargs)
    except OSError:
        if os.name != "nt":
            raise
        return subprocess.Popen(
            subprocess.list2cmdline(command),
            shell=True,
            env=env,
            **kwargs,
        )


def get_codex_status() -> dict:
    """Return a sanitized local Codex CLI status summary."""
    env = get_codex_subprocess_env()
    explicit_path = os.environ.get("SHARP_CODEX", "").strip().strip('"')
    codex_path = explicit_path or shutil.which("codex", path=env.get("PATH"))
    if not codex_path:
        return {
            "available": False,
            "ready": False,
            "version": None,
            "login_status": "unknown",
            "error": (
                "Codex CLI was not found on PATH. Install Codex, restart ML-SHARP, "
                "or set SHARP_CODEX to the Codex executable path."
            ),
        }

    try:
        version_result = run_codex_cli(
            ["--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        return {
            "available": False,
            "ready": False,
            "version": None,
            "login_status": "unknown",
            "error": f"Unable to run Codex CLI: {exc}",
        }

    version = (version_result.stdout or version_result.stderr).strip() or None
    if version_result.returncode != 0:
        return {
            "available": False,
            "ready": False,
            "version": version,
            "login_status": "unknown",
            "error": "Codex CLI returned an error while checking version.",
        }

    login_status = "unknown"
    try:
        login_result = run_codex_cli(
            ["login", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        login_text = f"{login_result.stdout}\n{login_result.stderr}".lower()
        if login_result.returncode == 0:
            login_status = "logged_in"
        elif "not" in login_text and ("login" in login_text or "signed" in login_text):
            login_status = "not_logged_in"
        else:
            login_status = "unknown"
    except Exception:
        login_status = "unknown"

    ready = login_status == "logged_in"
    login_error = None
    if login_status == "not_logged_in":
        login_error = "Codex CLI is installed but not signed in. Run `codex login`, then try again."
    elif login_status == "unknown":
        login_error = (
            "Codex CLI is installed, but ML-SHARP could not confirm the login state. "
            "Run `codex login status` in a terminal and sign in if required."
        )

    return {
        "available": True,
        "ready": ready,
        "version": version,
        "login_status": login_status,
        "error": login_error,
    }


def normalize_imagegen_directions(raw_value: str) -> tuple[str, ...]:
    """Normalize Image Extension edge selections from the UI."""
    raw_value = (raw_value or "all").strip()
    if raw_value in IMAGEGEN_LEGACY_DIRECTIONS:
        return IMAGEGEN_LEGACY_DIRECTIONS[raw_value]

    requested = {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }
    selected = tuple(edge for edge in IMAGEGEN_EDGE_ORDER if edge in requested)
    if not selected or len(selected) != len(requested):
        raise ValueError("Invalid extension direction.")
    return selected


def describe_imagegen_directions(directions: tuple[str, ...]) -> str:
    """Return a prompt-friendly description for selected extension edges."""
    selected = tuple(edge for edge in IMAGEGEN_EDGE_ORDER if edge in directions)
    selected_set = set(selected)
    if selected_set == set(IMAGEGEN_EDGE_ORDER):
        return "all four sides"
    if selected_set == {"left", "right"}:
        return "the left and right edges only"
    if selected_set == {"top", "bottom"}:
        return "the top and bottom edges only"
    labels = [IMAGEGEN_EDGE_LABELS[edge] for edge in selected]
    if len(labels) == 1:
        return f"the {labels[0]} only"
    if len(labels) == 2:
        return f"the {labels[0]} and {labels[1]} only"
    return "the " + ", ".join(labels[:-1]) + f", and {labels[-1]} only"


def build_imagegen_prompt(
    padding_percent: int,
    guard_band_enabled: bool,
    directions: tuple[str, ...],
    aspect_mode: str,
    variants: int,
    preserve_original_details: bool,
    user_prompt: str,
) -> str:
    """Build the fixed Codex imagegen prompt."""
    custom = user_prompt.strip()
    custom_section = f"\nUser direction:\n{custom}\n" if custom else ""
    aspect_text = (
        "Keep the natural expanded aspect ratio; do not force a specific target ratio."
        if aspect_mode == "keep"
        else f"After extending outward, keep the final composition close to {aspect_mode} without cropping the original image content."
    )
    variant_text = (
        "Generate exactly one final candidate image."
        if variants == 1
        else f"Generate exactly {variants} distinct candidate images."
    )
    direction_text = describe_imagegen_directions(directions)
    extension_text = (
        f"Extend the image outward on {direction_text} by approximately {padding_percent}%."
        if guard_band_enabled
        else (
            f"Extend the image outward on {direction_text}. "
            "Follow the user direction for how much new image area to add; "
            "if no amount is specified, choose a natural expansion amount."
        )
    )
    source_detail_line = (
        "- Use the original image as the visual anchor; avoid unnecessary changes to the existing image area.\n"
        if preserve_original_details
        else ""
    )

    return f"""Use $imagegen to edit the attached image.

Task:
- {extension_text}
{source_detail_line}- Keep the result visually coherent with the scene, style, lighting, perspective, materials, and background.
- {aspect_text}
- Avoid adding text, logos, watermarks, borders, or frames unless explicitly requested.
{custom_section}
Output:
- {variant_text}
- If image generation fails, return the failure message.
- After image generation finishes, stop.
"""


def write_stream_to_file(stream, output_file) -> None:
    """Copy a subprocess stream to a file until EOF."""
    try:
        for chunk in iter(lambda: stream.readline(), ""):
            if not chunk:
                break
            output_file.write(chunk)
            output_file.flush()
    except Exception:
        LOGGER.exception("Failed while streaming Codex subprocess output")


def write_codex_stdout_to_file(stream, output_file, job: dict) -> None:
    """Copy Codex JSONL stdout and capture thread metadata for generated image lookup."""
    try:
        for chunk in iter(lambda: stream.readline(), ""):
            if not chunk:
                break
            output_file.write(chunk)
            output_file.flush()

            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "thread.started" and event.get("thread_id"):
                job["thread_id"] = event["thread_id"]
            elif event.get("type") == "turn.completed":
                job["codex_turn_completed_at"] = time.time()
            elif event.get("type") == "turn.failed":
                job["codex_turn_completed_at"] = time.time()
                job["error_msg"] = event.get("error") or "Image Extension failed."

            item = event.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                job["last_agent_message"] = item["text"]
    except Exception:
        LOGGER.exception("Failed while streaming Codex JSONL output")


def validate_image_file(path: Path) -> None:
    """Open an image to verify the generated output is readable."""
    with Image.open(path) as img:
        img.verify()


def get_codex_generated_images_root() -> Path:
    """Return the local Codex generated-images folder."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "generated_images"
    return Path.home() / ".codex" / "generated_images"


def find_generated_image_candidates(job: dict) -> list[Path]:
    """Find generated Codex image candidates for this job."""
    generated_root = get_codex_generated_images_root()
    if not generated_root.exists():
        return []

    roots = []
    thread_id = job.get("thread_id")
    if thread_id:
        thread_root = generated_root / thread_id
        if thread_root.exists():
            roots.append(thread_root)
    if not roots:
        return []

    started_at = job.get("started_at") or job.get("created_at") or 0
    candidates = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime + 2 < started_at:
                continue
            candidates.append((stat.st_mtime, stat.st_size, path))

    result = []
    copied_sources = set(job.get("copied_source_paths") or [])
    for _, _, path in sorted(candidates):
        path_str = str(path)
        if path_str in copied_sources:
            continue
        try:
            validate_image_file(path)
            result.append(path)
        except Exception:
            continue
    return result


def add_imagegen_variant(job: dict, generated_path: Path) -> bool:
    """Store one generated image as a selectable PNG variant."""
    lock = job.get("lock")
    if lock is not None:
        with lock:
            return _add_imagegen_variant_unlocked(job, generated_path)
    return _add_imagegen_variant_unlocked(job, generated_path)


def _add_imagegen_variant_unlocked(job: dict, generated_path: Path) -> bool:
    """Store one generated image while the caller owns the job lock."""
    outputs = job.setdefault("outputs", [])
    target_count = job.get("variants")
    if target_count and len(outputs) >= target_count:
        return False

    copied_sources = job.setdefault("copied_source_paths", [])
    if str(generated_path) in copied_sources:
        return False

    index = len(outputs) + 1
    filename = f"variant-{index:02d}.png"
    variant_path = job["job_dir"] / filename
    with Image.open(generated_path) as image:
        image.save(variant_path, "PNG")
    copied_sources.append(str(generated_path))

    item = {
        "index": index,
        "filename": filename,
        "source_path": str(generated_path),
    }
    outputs.append(item)

    if not job.get("selected_output"):
        job["selected_output"] = filename
        job["generated_source_path"] = str(generated_path)

    (job["job_dir"] / "generated-source.txt").write_text(
        "\n".join(item["source_path"] for item in outputs),
        encoding="utf-8",
    )
    return True


def count_imagegen_outputs(job: dict) -> int:
    """Return the number of generated imagegen variants."""
    lock = job.get("lock")
    if lock is not None:
        with lock:
            return len(job.get("outputs") or [])
    return len(job.get("outputs") or [])


def get_selected_imagegen_variant_path(job: dict) -> Path | None:
    """Resolve the selected imagegen variant without creating a separate output copy."""
    outputs = job.get("outputs") or []
    selected = job.get("selected_output")
    if not selected and outputs:
        selected = outputs[0].get("filename")
    if not selected:
        return None

    allowed = {item.get("filename") for item in outputs}
    if selected not in allowed:
        return None

    variant_path = (job["job_dir"] / selected).resolve()
    job_dir = job["job_dir"].resolve()
    if not _is_relative_to(variant_path, job_dir) or not variant_path.exists():
        return None

    return variant_path


def stop_imagegen_job_processes(job: dict) -> None:
    """Stop the parent imagegen process and any parallel worker processes."""
    stop_process_tree(job.get("process"))
    for worker in job.get("workers") or []:
        stop_process_tree(worker.get("process"))


def create_parallel_imagegen_worker(job: dict, worker_index: int, requested_variants: int) -> dict:
    """Create one worker record and directory for a parallel imagegen agent."""
    worker_dir = job["job_dir"] / f"agent-{worker_index:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_input_path = worker_dir / "input.png"
    shutil.copy2(job["input_path"], worker_input_path)

    worker = {
        "index": worker_index,
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "job_dir": worker_dir,
        "input_path": worker_input_path,
        "process": None,
        "thread_id": None,
        "codex_turn_completed_at": None,
        "last_agent_message": None,
        "requested_variants": requested_variants,
        "copied_source_paths": [],
        "output_count": 0,
        "error_msg": None,
    }

    lock = job.get("lock")
    if lock is not None:
        with lock:
            job.setdefault("workers", []).append(worker)
    else:
        job.setdefault("workers", []).append(worker)
    return worker


def format_imagegen_worker_failures(workers: list[dict]) -> str:
    """Return compact diagnostics for every failed imagegen worker."""
    details = []
    for failure in get_imagegen_worker_failures(workers):
        details.append(f"{failure['label']}: {failure['error']}")
    return " ".join(details)


def get_imagegen_worker_failures(workers: list[dict]) -> list[dict]:
    """Return structured diagnostics for every failed imagegen worker."""
    failures = []
    for worker in workers:
        if worker.get("status") not in {"error", "timeout"}:
            continue
        agent_index = worker.get("index")
        label = f"Parallel Agent {agent_index:02d}" if isinstance(agent_index, int) else "Parallel Agent"
        failures.append({
            "index": agent_index,
            "label": label,
            "status": worker.get("status"),
            "error": worker.get("error_msg") or "No failure details were reported.",
        })
    return failures


def build_parallel_imagegen_prompt(job: dict, worker: dict) -> str:
    """Build the prompt for one worker's assigned candidate count."""
    return build_imagegen_prompt(
        job["padding_percent"],
        job["guard_band_enabled"],
        job["directions"],
        job["aspect_mode"],
        worker["requested_variants"],
        job.get("preserve_original_details", True),
        job["prompt"],
    )


def run_parallel_imagegen_worker(job: dict, worker: dict) -> None:
    """Run one Codex/imagegen worker for its assigned candidate count."""
    process = None
    stdout_thread = None
    stderr_thread = None
    worker_dir = worker["job_dir"]
    prompt_path = worker_dir / "prompt.txt"
    stdout_path = worker_dir / "codex-stdout.jsonl"
    stderr_path = worker_dir / "codex-stderr.log"
    final_message_path = worker_dir / "final-message.txt"

    try:
        if job.get("stop_signal"):
            worker["status"] = "stopped"
            return

        worker["status"] = "running"
        worker["started_at"] = time.time()
        prompt = build_parallel_imagegen_prompt(job, worker)
        prompt_path.write_text(prompt, encoding="utf-8")

        command_args = [
            "exec",
            "--cd",
            str(worker_dir),
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--image",
            str(worker["input_path"]),
            "--json",
            "--output-last-message",
            str(final_message_path),
            "-",
        ]
        worker["command"] = get_codex_command_base() + command_args

        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            process = popen_codex_cli(
                command_args,
                cwd=worker_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            worker["process"] = process
            stdout_thread = threading.Thread(
                target=write_codex_stdout_to_file,
                args=(process.stdout, stdout_file, worker),
                daemon=True,
            )
            stderr_thread = threading.Thread(target=write_stream_to_file, args=(process.stderr, stderr_file), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            try:
                process.stdin.write(prompt)
                process.stdin.close()
            except Exception:
                LOGGER.exception("Failed to write prompt to parallel Codex stdin")

            while True:
                if job.get("stop_signal"):
                    stop_process_tree(process)
                    worker["status"] = "stopped"
                    return

                candidates = find_generated_image_candidates(worker)
                for generated_path in candidates:
                    try:
                        if add_imagegen_variant(job, generated_path):
                            worker.setdefault("copied_source_paths", []).append(str(generated_path))
                            worker["output_count"] = worker.get("output_count", 0) + 1
                    except Exception:
                        # The file may still be mid-write. Keep polling until it is readable.
                        pass

                if (
                    count_imagegen_outputs(job) >= job["variants"]
                    or worker.get("output_count", 0) >= worker["requested_variants"]
                ):
                    worker["status"] = "done"
                    stop_process_tree(process)
                    return

                if worker.get("codex_turn_completed_at") and time.time() - worker["codex_turn_completed_at"] > 3:
                    stop_process_tree(process)
                    break

                if process.poll() is not None:
                    break

                if time.time() - worker["started_at"] > IMAGEGEN_TIMEOUT_SECONDS:
                    stop_process_tree(process)
                    worker["status"] = "timeout"
                    worker["error_msg"] = "Image Extension timed out for this Parallel Agent."
                    return

                time.sleep(1)

        candidates = find_generated_image_candidates(worker)
        for generated_path in candidates:
            try:
                if add_imagegen_variant(job, generated_path):
                    worker.setdefault("copied_source_paths", []).append(str(generated_path))
                    worker["output_count"] = worker.get("output_count", 0) + 1
            except Exception:
                pass

        if worker.get("output_count", 0) > 0:
            worker["status"] = "done"
            return

        final_message = ""
        try:
            if final_message_path.exists():
                final_message = final_message_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            final_message = ""

        worker["status"] = "error"
        worker["error_msg"] = (
            worker.get("error_msg")
            or final_message
            or worker.get("last_agent_message")
            or "Image Extension finished, but this Parallel Agent did not produce an image."
        )

    except Exception as exc:
        LOGGER.exception("Parallel imagegen worker failed: %s", worker.get("index"))
        if process and process.poll() is None:
            stop_process_tree(process)
        worker["status"] = "error"
        worker["error_msg"] = str(exc)
    finally:
        if stdout_thread is not None:
            stdout_thread.join(timeout=1)
        if stderr_thread is not None:
            stderr_thread.join(timeout=1)
        worker["process"] = None
        worker["finished_at"] = time.time()


def process_parallel_imagegen_job(job: dict) -> None:
    """Run a parent imagegen job using a fixed number of Codex agents."""
    job["status"] = "running"
    job["started_at"] = time.time()
    target_count = job["variants"]
    agent_count = min(job.get("parallel_agents", 1), target_count)
    base_count = target_count // agent_count
    remainder = target_count % agent_count
    active_workers: list[tuple[dict, threading.Thread]] = []

    for worker_index in range(1, agent_count + 1):
        requested_variants = base_count + (1 if worker_index <= remainder else 0)
        worker = create_parallel_imagegen_worker(job, worker_index, requested_variants)
        thread = threading.Thread(target=run_parallel_imagegen_worker, args=(job, worker), daemon=True)
        thread.start()
        active_workers.append((worker, thread))

    while True:
        if job.get("stop_signal"):
            stop_imagegen_job_processes(job)
            job["status"] = "stopped"
            return

        if count_imagegen_outputs(job) >= target_count:
            job["status"] = "done"
            stop_imagegen_job_processes(job)
            for _, thread in active_workers:
                thread.join(timeout=1)
            return

        active_workers = [(worker, thread) for worker, thread in active_workers if thread.is_alive()]

        if not active_workers:
            output_count = count_imagegen_outputs(job)
            if output_count > 0:
                job["status"] = "done"
                if output_count < target_count:
                    details = format_imagegen_worker_failures(job.get("workers") or [])
                    detail = f" {details}" if details else ""
                    job["error_msg"] = (
                        f"Image Extension produced {output_count}/{target_count} images.{detail}"
                    )
                return

            details = format_imagegen_worker_failures(job.get("workers") or [])
            job["status"] = "error"
            job["error_msg"] = details or "Image Extension did not produce an image."
            return

        if time.time() - job["started_at"] > IMAGEGEN_TIMEOUT_SECONDS:
            stop_imagegen_job_processes(job)
            for _, thread in active_workers:
                thread.join(timeout=1)
            job["status"] = "timeout"
            job["error_msg"] = "Image Extension timed out and was stopped."
            return

        time.sleep(1)


def process_imagegen_job(job_id: str) -> None:
    """Run one local Codex image generation/editing job."""
    job = _imagegen_jobs[job_id]
    process = None
    stdout_thread = None
    stderr_thread = None
    acquired = False

    try:
        acquired = _imagegen_semaphore.acquire(timeout=1)
        if not acquired:
            job["status"] = "queued"
            _imagegen_semaphore.acquire()
        acquired = True
        if job.get("stop_signal"):
            job["status"] = "stopped"
            return

        if job.get("parallel_agents", 1) > 1:
            process_parallel_imagegen_job(job)
            return

        job["status"] = "running"
        job_dir = job["job_dir"]
        input_path = job["input_path"]
        desired_variants = job["variants"]
        prompt_path = job_dir / "prompt.txt"
        stdout_path = job_dir / "codex-stdout.jsonl"
        stderr_path = job_dir / "codex-stderr.log"
        final_message_path = job_dir / "final-message.txt"

        prompt = build_imagegen_prompt(
            job["padding_percent"],
            job["guard_band_enabled"],
            job["directions"],
            job["aspect_mode"],
            desired_variants,
            job.get("preserve_original_details", True),
            job["prompt"],
        )
        prompt_path.write_text(prompt, encoding="utf-8")

        command_args = [
            "exec",
            "--cd",
            str(job_dir),
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--image",
            str(input_path),
            "--json",
            "--output-last-message",
            str(final_message_path),
            "-",
        ]
        job["command"] = get_codex_command_base() + command_args
        job["started_at"] = time.time()

        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            process = popen_codex_cli(
                command_args,
                cwd=job_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            job["process"] = process
            stdout_thread = threading.Thread(target=write_codex_stdout_to_file, args=(process.stdout, stdout_file, job), daemon=True)
            stderr_thread = threading.Thread(target=write_stream_to_file, args=(process.stderr, stderr_file), daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            try:
                try:
                    process.stdin.write(prompt)
                    process.stdin.close()
                except Exception:
                    LOGGER.exception("Failed to write prompt to Codex stdin")

                while True:
                    if job.get("stop_signal"):
                        stop_process_tree(process)
                        job["status"] = "stopped"
                        return

                    candidates = find_generated_image_candidates(job)
                    for generated_path in candidates:
                        try:
                            add_imagegen_variant(job, generated_path)
                        except Exception:
                            # The file may still be mid-write. Keep polling until it is readable.
                            pass
                    if len(job.get("outputs") or []) >= desired_variants:
                        job["status"] = "done"
                        stop_process_tree(process)
                        return

                    if job.get("codex_turn_completed_at") and time.time() - job["codex_turn_completed_at"] > 3:
                        stop_process_tree(process)
                        break

                    if process.poll() is not None:
                        break

                    if time.time() - job["started_at"] > IMAGEGEN_TIMEOUT_SECONDS:
                        stop_process_tree(process)
                        job["status"] = "timeout"
                        job["error_msg"] = "Image Extension timed out and was stopped."
                        return

                    time.sleep(1)
            finally:
                if process.poll() is None:
                    stop_process_tree(process)
                stdout_thread.join(timeout=2)
                stderr_thread.join(timeout=2)

        candidates = find_generated_image_candidates(job)
        for generated_path in candidates:
            try:
                add_imagegen_variant(job, generated_path)
            except Exception:
                pass
        if job.get("outputs"):
            job["status"] = "done"
            return

        final_message = ""
        try:
            if final_message_path.exists():
                final_message = final_message_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            final_message = ""
        job["status"] = "error"
        job["error_msg"] = (
            job.get("error_msg")
            or final_message
            or job.get("last_agent_message")
            or "Image Extension finished, but no generated image was found."
        )

    except Exception as exc:
        LOGGER.exception("Imagegen job failed: %s", job_id)
        if process and process.poll() is None:
            stop_process_tree(process)
        job["status"] = "error"
        job["error_msg"] = str(exc)
    finally:
        job["process"] = None
        job["finished_at"] = time.time()
        if acquired:
            _imagegen_semaphore.release()


def get_sbs_frame_folder_for_movie(movie_path: Path) -> Path | None:
    """Return the SBS frame folder associated with a generated movie if inferable."""
    if movie_path.suffix.lower() != ".mp4":
        return None
    stem = movie_path.stem
    if not stem.endswith("_SBS"):
        return None
    return movie_path.parent / f"{stem[:-4]}_frames"


def delete_output_item(item_type: str, filename: str) -> dict:
    """Delete a generated output item and any generated sidecar files."""
    normalized_filename = filename.replace("\\", "/").strip("/")
    for active_path in get_active_output_paths():
        if (
            normalized_filename == active_path
            or normalized_filename.startswith(f"{active_path}/")
            or active_path.startswith(f"{normalized_filename}/")
        ):
            raise ValueError("Stop the active job before deleting this output.")

    file_path = resolve_output_file(filename)
    if file_path is None:
        raise ValueError("Invalid output path.")

    output_root = OUTPUT_DIR.resolve()
    deleted: list[str] = []

    if item_type == "ply":
        if file_path.suffix.lower() != ".ply" or not file_path.is_file():
            raise ValueError("The selected Library result is not a PLY file.")

        thumbnail_path = get_thumbnail_path_for_ply(file_path)
        safe_unlink(file_path, OUTPUT_DIR)
        deleted.append(filename)
        if safe_unlink(thumbnail_path, THUMBNAIL_DIR):
            deleted.append(thumbnail_path.relative_to(OUTPUT_DIR).as_posix())

    elif item_type == "video":
        if not file_path.is_dir() or not file_path.name.endswith("_plys"):
            raise ValueError("The selected Library result is not a PLY Sequence.")

        relative_folder = file_path.relative_to(output_root)
        thumbnail_folder = THUMBNAIL_DIR / relative_folder
        safe_rmtree(file_path, OUTPUT_DIR)
        deleted.append(filename)
        if safe_rmtree(thumbnail_folder, THUMBNAIL_DIR):
            deleted.append(thumbnail_folder.relative_to(OUTPUT_DIR).as_posix())

    elif item_type == "movie":
        if file_path.suffix.lower() != ".mp4" or not file_path.is_file():
            raise ValueError("The selected Library result is not an SBS Movie.")

        frame_folder = get_sbs_frame_folder_for_movie(file_path)
        safe_unlink(file_path, OUTPUT_DIR)
        deleted.append(filename)
        if frame_folder and safe_rmtree(frame_folder, OUTPUT_DIR):
            deleted.append(frame_folder.resolve().relative_to(output_root).as_posix())

    else:
        raise ValueError("Unsupported output type.")

    for job in list(_active_jobs.values()):
        job_files = job.get("files")
        if isinstance(job_files, list):
            job["files"] = [path for path in job_files if path != filename]

    return {"deleted": deleted}


def generate_ply_from_upload(
    file,
    predictor: RGBGaussianPredictor,
    device: torch.device,
    use_fp16: bool,
) -> dict:
    """Generate a PLY and thumbnail from an uploaded image file."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    unique_id = str(uuid.uuid4())[:8]
    original_stem = sanitize_output_stem(file.filename, "image")
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            file.save(tmp.name)

        image, _, f_px = io.load_rgb(tmp_path)
        height, width = image.shape[:2]
        gaussians = predict_image(predictor, image, f_px, device, use_fp16=use_fp16)

        output_filename = f"{original_stem}_{unique_id}.ply"
        output_path = OUTPUT_DIR / output_filename
        save_ply(gaussians, f_px, (height, width), output_path)

        thumbnail_path = save_thumbnail(image, output_path)
        thumbnail_url = None
        if thumbnail_path is not None:
            thumbnail_relative = thumbnail_path.relative_to(THUMBNAIL_DIR).as_posix()
            thumbnail_url = f"/thumbnail/{urllib.parse.quote(thumbnail_relative)}"

        LOGGER.info("Saved PLY to: %s", output_path)

        return {
            "filename": output_filename,
            "download_url": f"/download/{urllib.parse.quote(output_filename)}",
            "view_url": f"/ply/{urllib.parse.quote(output_filename)}",
            "thumbnail_url": thumbnail_url,
        }
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_fp16: bool = False
) -> Gaussians3D:
    """Predict Gaussians from a single image."""
    # Wrap single image in list and use batch predictor
    return predict_batch(predictor, [image], f_px, device, use_fp16)[0]


@torch.no_grad()
def predict_batch(
    predictor: RGBGaussianPredictor,
    images: list[np.ndarray],
    f_px: float,
    device: torch.device,
    use_fp16: bool = False
) -> list[Gaussians3D]:
    """Predict a batch while preventing concurrent access to the shared model."""
    with _inference_lock:
        return _predict_batch_unlocked(predictor, images, f_px, device, use_fp16)


@torch.no_grad()
def _predict_batch_unlocked(
    predictor: RGBGaussianPredictor,
    images: list[np.ndarray],
    f_px: float,
    device: torch.device,
    use_fp16: bool = False
) -> list[Gaussians3D]:
    """Predict Gaussians from a batch of images."""
    if not images:
        return []

    internal_shape = (1536, 1536)
    
    # Prepare batch tensors
    # Stack numpy arrays: (B, H, W, C) -> permute to (B, C, H, W)
    batch_np = np.stack(images)
    batch_pt = torch.from_numpy(batch_np).float().to(device).permute(0, 3, 1, 2) / 255.0
    
    batch_size, _, height, width = batch_pt.shape
    
    # Disparity factor: (B,)
    disparity_val = f_px / width
    disparity_factor = torch.full((batch_size,), disparity_val, device=device, dtype=torch.float32)

    # Resize batch
    batch_resized_pt = F.interpolate(
        batch_pt,
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Inference
    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            gaussians_ndc = predictor(batch_resized_pt, disparity_factor)
    else:
        gaussians_ndc = predictor(batch_resized_pt, disparity_factor)

    # Post-processing intrinsics
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Unproject whole batch
    gaussians_batch = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    # Split batched Gaussians3D into list of individual Gaussians3D
    results = []
    for i in range(batch_size):
        results.append(Gaussians3D(
            mean_vectors=gaussians_batch.mean_vectors[i:i+1],
            singular_values=gaussians_batch.singular_values[i:i+1],
            quaternions=gaussians_batch.quaternions[i:i+1],
            colors=gaussians_batch.colors[i:i+1],
            opacities=gaussians_batch.opacities[i:i+1]
        ))
        
    return results


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/outputs")
def outputs():
    """List generated PLY files for the WebUI Library."""
    try:
        limit = int(request.args.get("limit", 60))
    except ValueError:
        limit = 60
    limit = max(1, min(limit, 200))
    return jsonify({"items": get_output_items(limit=limit)})


@app.route("/outputs/delete", methods=["POST"])
def delete_output():
    """Delete a generated output item from the WebUI Library."""
    try:
        data = request.get_json(silent=True) or {}
        item_type = str(data.get("type", "")).strip()
        filename = str(data.get("filename", "")).strip()
        if not item_type or not filename:
            return jsonify({"error": "Missing output type or filename."}), 400

        result = delete_output_item(item_type, filename)
        return jsonify({"success": True, **result})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("Failed to delete output")
        return jsonify({"error": str(exc)}), 500


@app.route("/codex/status")
def codex_status():
    """Return local Codex CLI availability without exposing credentials."""
    return jsonify(get_codex_status())


@app.route("/imagegen/extend", methods=["POST"])
def imagegen_extend():
    """Start a local Codex image expansion/editing job."""
    if "image" not in request.files:
        return jsonify({"error": "No image selected."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}."}), 400

    aspect_mode = request.form.get("aspect_mode", request.form.get("aspect_ratio", "keep")).strip()
    if aspect_mode not in SUPPORTED_IMAGEGEN_ASPECTS:
        return jsonify({"error": f"Unsupported Image Extension aspect setting: {aspect_mode}."}), 400

    try:
        directions = normalize_imagegen_directions(
            request.form.get("directions", request.form.get("direction", "all"))
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    guard_band_enabled = request.form.get("guard_band_enabled", "true").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    preserve_original_details = request.form.get("preserve_original_details", "true").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }

    padding_percent = 15
    if guard_band_enabled:
        try:
            padding_percent = int(request.form.get("padding_percent", "15"))
        except ValueError:
            return jsonify({"error": "Guard Band must be a whole number."}), 400
        min_padding, max_padding = IMAGEGEN_PADDING_RANGE
        if padding_percent < min_padding or padding_percent > max_padding:
            return jsonify({"error": f"Guard Band must be between {min_padding}% and {max_padding}%."}), 400

    try:
        variants = int(request.form.get("variants", "1"))
    except ValueError:
        return jsonify({"error": "Image Count must be a whole number."}), 400
    min_variants, max_variants = IMAGEGEN_VARIANT_RANGE
    if variants < min_variants or variants > max_variants:
        return jsonify({"error": f"Image Count must be between {min_variants} and {max_variants}."}), 400

    try:
        parallel_agents = int(request.form.get("parallel_agents", "1"))
    except ValueError:
        return jsonify({"error": "Parallel Agent must be a whole number."}), 400
    min_parallel_agents, max_parallel_agents = IMAGEGEN_PARALLEL_AGENT_RANGE
    if parallel_agents < min_parallel_agents or parallel_agents > min(max_parallel_agents, variants):
        return jsonify({"error": f"Parallel Agent must be between 1 and {variants}."}), 400

    prompt = request.form.get("prompt", "").strip()
    codex = get_codex_status()
    if not codex["available"]:
        return jsonify({"error": codex["error"] or "Codex CLI is unavailable.", "codex": codex}), 400
    if not codex["ready"]:
        return jsonify({"error": codex["error"], "codex": codex}), 400

    job_id = f"ig_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = get_imagegen_job_dir(job_id)
    if job_dir is None:
        return jsonify({"error": "Unable to create the Image Extension job."}), 500

    try:
        job_dir.mkdir(parents=True, exist_ok=False)
        raw_input_path = job_dir / f"source{ext}"
        input_path = job_dir / "input.png"
        file.save(raw_input_path)

        try:
            with Image.open(raw_input_path) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.save(input_path, "PNG")
        except Exception as exc:
            raise ValueError("Unable to prepare this image format for Codex.") from exc

        _imagegen_jobs[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "job_dir": job_dir,
            "input_path": input_path,
            "aspect_mode": aspect_mode,
            "directions": directions,
            "guard_band_enabled": guard_band_enabled,
            "preserve_original_details": preserve_original_details,
            "padding_percent": padding_percent,
            "variants": variants,
            "parallel_agents": parallel_agents,
            "prompt": prompt,
            "process": None,
            "thread_id": None,
            "codex_turn_completed_at": None,
            "last_agent_message": None,
            "generated_source_path": None,
            "outputs": [],
            "selected_output": None,
            "copied_source_paths": [],
            "workers": [],
            "lock": threading.Lock(),
            "stop_signal": False,
            "error_msg": None,
        }

        thread = threading.Thread(target=process_imagegen_job, args=(job_id,), daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
        })

    except Exception as exc:
        LOGGER.exception("Failed to start imagegen job")
        if job_dir.exists():
            safe_rmtree(job_dir, IMAGEGEN_DIR)
        return jsonify({"error": str(exc)}), 500


@app.route("/imagegen/status/<job_id>")
def imagegen_status(job_id):
    """Return status for a local Codex imagegen job."""
    job = _imagegen_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Image Extension job not found."}), 404

    outputs = []
    for item in job.get("outputs") or []:
        filename = item.get("filename")
        if not filename:
            continue
        outputs.append({
            "index": item.get("index"),
            "filename": filename,
            "url": f"/imagegen/output/{urllib.parse.quote(job_id)}/{urllib.parse.quote(filename)}",
            "selected": filename == job.get("selected_output"),
        })

    workers = job.get("workers") or []
    failed_agents = sum(1 for worker in workers if worker.get("status") in {"error", "timeout"})
    finished_agents = sum(
        1 for worker in workers if worker.get("status") in {"done", "error", "timeout", "stopped"}
    )
    total_agents = len(workers) or job.get("parallel_agents", 1)
    worker_failures = get_imagegen_worker_failures(workers)
    return jsonify({
        "status": job["status"],
        "outputs": outputs,
        "error": job.get("error_msg"),
        "variants": job.get("variants"),
        "failed_agents": failed_agents,
        "finished_agents": finished_agents,
        "total_agents": total_agents,
        "worker_failures": worker_failures,
    })


@app.route("/imagegen/stop/<job_id>", methods=["POST"])
def imagegen_stop(job_id):
    """Stop a local Codex imagegen job."""
    job = _imagegen_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Image Extension job not found."}), 404

    if job.get("status") not in {"queued", "running"}:
        return jsonify({"success": True, "status": job["status"]})

    job["stop_signal"] = True
    stop_imagegen_job_processes(job)
    job["status"] = "stopped"
    candidate_count = count_imagegen_outputs(job)
    image_label = "image" if candidate_count == 1 else "images"
    job["error_msg"] = (
        f"Image Extension stopped. Kept {candidate_count} generated {image_label}."
        if candidate_count > 0
        else "Image Extension stopped before any images were generated."
    )
    return jsonify({"success": True, "status": job["status"]})


@app.route("/imagegen/output/<job_id>")
def imagegen_output(job_id):
    """Serve the selected local Codex imagegen variant."""
    job = _imagegen_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Image Extension job not found."}), 404

    output_path = get_selected_imagegen_variant_path(job)
    if output_path is None:
        return jsonify({"error": "Image Extension image not found."}), 404

    return send_file(output_path, mimetype="image/png")


@app.route("/imagegen/output/<job_id>/<path:filename>")
def imagegen_variant_output(job_id, filename):
    """Serve a generated local Codex imagegen variant."""
    job = _imagegen_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Image Extension job not found."}), 404

    variant_path = (job["job_dir"] / filename).resolve()
    job_dir = job["job_dir"].resolve()
    allowed = {item.get("filename") for item in job.get("outputs") or []}
    if (
        filename not in allowed
        or not _is_relative_to(variant_path, job_dir)
        or not variant_path.exists()
    ):
        return jsonify({"error": "Image Extension image not found."}), 404

    return send_file(variant_path, mimetype="image/png")


@app.route("/imagegen/select/<job_id>", methods=["POST"])
def imagegen_select(job_id):
    """Select which imagegen variant should feed Gaussian generation."""
    job = _imagegen_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Image Extension job not found."}), 404

    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename", "")).strip()
    match = None
    for item in job.get("outputs") or []:
        if item.get("filename") == filename:
            match = item
            break
    if not match:
        return jsonify({"error": "Image Extension image not found."}), 404

    variant_path = (job["job_dir"] / filename).resolve()
    if not _is_relative_to(variant_path, job["job_dir"].resolve()) or not variant_path.exists():
        return jsonify({"error": "Image Extension image not found."}), 404

    validate_image_file(variant_path)
    job["selected_output"] = filename
    job["generated_source_path"] = match.get("source_path")
    return jsonify({"success": True})


@app.route("/generate_from_imagegen", methods=["POST"])
def generate_from_imagegen():
    """Generate a PLY from the selected imagegen variant without browser re-upload."""
    try:
        data = request.get_json(silent=True) or {}
        job_id = str(data.get("job_id", "")).strip()
        quality = str(data.get("quality", "balanced")).strip()
        if quality not in {"fast", "balanced"}:
            return jsonify({"error": "Unsupported quality mode."}), 400
        use_fp16 = quality == "fast"

        job = _imagegen_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Image Extension job not found."}), 404
        output_path = get_selected_imagegen_variant_path(job)
        if output_path is None:
            return jsonify({"error": "Image Extension has not produced an image yet."}), 400

        predictor, device = get_predictor()
        image, _, f_px = io.load_rgb(output_path)
        height, width = image.shape[:2]
        gaussians = predict_image(predictor, image, f_px, device, use_fp16=use_fp16)

        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"{job_id}_{unique_id}.ply"
        ply_path = OUTPUT_DIR / output_filename
        save_ply(gaussians, f_px, (height, width), ply_path)

        thumbnail_path = save_thumbnail(image, ply_path)
        thumbnail_url = None
        if thumbnail_path is not None:
            thumbnail_relative = thumbnail_path.relative_to(THUMBNAIL_DIR).as_posix()
            thumbnail_url = f"/thumbnail/{urllib.parse.quote(thumbnail_relative)}"

        return jsonify({
            "success": True,
            "filename": output_filename,
            "download_url": f"/download/{urllib.parse.quote(output_filename)}",
            "view_url": f"/ply/{urllib.parse.quote(output_filename)}",
            "thumbnail_url": thumbnail_url,
        })
    except Exception as exc:
        LOGGER.exception("Error generating PLY from imagegen output")
        return jsonify({"error": str(exc)}), 500


@app.route("/thumbnail/<path:filename>")
def serve_thumbnail(filename: str):
    """Serve a generated thumbnail image."""
    file_path = (THUMBNAIL_DIR / filename).resolve()
    thumbnail_root = THUMBNAIL_DIR.resolve()
    if not _is_relative_to(file_path, thumbnail_root) or not file_path.is_file():
        return jsonify({"error": "Thumbnail not found."}), 404
    return send_file(file_path, mimetype="image/jpeg")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate a 3DGS PLY file from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image selected."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    quality = request.form.get('quality', 'balanced')
    if quality not in {'fast', 'balanced'}:
        return jsonify({"error": "Unsupported quality mode."}), 400
    use_fp16 = (quality == 'fast')

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}."}), 400

    try:
        LOGGER.info(f"Processing uploaded file: {file.filename} | Quality: {quality} | FP16: {use_fp16}")

        predictor, device = get_predictor()
        result = generate_ply_from_upload(file, predictor, device, use_fp16)

        return jsonify({
            "success": True,
            **result,
        })

    except Exception as e:
        LOGGER.exception("Error during generation")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_batch", methods=["POST"])
def generate_batch():
    """Generate multiple 3DGS PLY files from uploaded images."""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images selected."}), 400

    quality = request.form.get('quality', 'balanced')
    if quality not in {'fast', 'balanced'}:
        return jsonify({"error": "Unsupported quality mode."}), 400
    use_fp16 = (quality == 'fast')

    valid_files = []
    errors = []
    for file in files:
        if file.filename == "":
            errors.append({"filename": "", "error": "Empty filename."})
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            errors.append({"filename": file.filename, "error": f"Unsupported file type: {ext}."})
            continue

        valid_files.append(file)

    if not valid_files:
        return jsonify({"error": "No supported images selected.", "errors": errors}), 400

    LOGGER.info(
        "Processing %d uploaded image files | Quality: %s | FP16: %s",
        len(valid_files),
        quality,
        use_fp16,
    )

    items = []
    try:
        predictor, device = get_predictor()
        for index, file in enumerate(valid_files, start=1):
            try:
                LOGGER.info("Batch item %d/%d: %s", index, len(valid_files), file.filename)
                item = generate_ply_from_upload(file, predictor, device, use_fp16)
                items.append(item)
            except Exception as exc:
                LOGGER.exception("Batch item failed: %s", file.filename)
                errors.append({"filename": file.filename, "error": str(exc)})

        if not items:
            return jsonify({"error": "Unable to generate PLY files.", "errors": errors}), 500

        return jsonify({
            "success": True,
            "items": items,
            "errors": errors,
        })
    except Exception as e:
        LOGGER.exception("Error during batch generation")
        return jsonify({"error": str(e), "errors": errors}), 500


def _process_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, batch_size):
    """Background worker to process video frames into PLY sequence."""
    reader = None
    try:
        total_frames = 0
        tmp_reader = None
        try:
            tmp_reader = iio.get_reader(tmp_path)
            total_frames = tmp_reader.count_frames()
            _active_jobs[job_id]['total_frames'] = total_frames
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0
        finally:
            if tmp_reader:
                try:
                    tmp_reader.close()
                except Exception:
                    pass

        reader = iio.get_reader(tmp_path)
        
        # Atomically reserve a unique output directory for this PLY Sequence.
        job_prefix, work_dir = reserve_numbered_output_dir(original_stem, "plys")
        _active_jobs[job_id]['output_path'] = work_dir.relative_to(OUTPUT_DIR).as_posix()
        try:
            (work_dir / SEQUENCE_METADATA_FILENAME).write_text(
                json.dumps({"fps": fps, "format_version": 1}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            LOGGER.warning("Unable to persist sequence metadata for job %s", job_id)
        
        LOGGER.info(f"Job {job_id} [{job_prefix}] | Batch: {batch_size} | Folder: {work_dir}")

        # Batch accumulation
        current_batch_frames = []
        current_batch_indices = []
        
        frames_iterator = enumerate(reader)
        
        # Helper to process what's in buffer
        def process_batch(frames, indices):
            try:
                processed_frames = [ensure_rgb_frame(frame) for frame in frames]
                
                # Assume all frames in video are same size
                h, w = processed_frames[0].shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)
                
                # Predict
                gaussians_list = predict_batch(predictor, processed_frames, f_px, device, use_fp16=use_fp16)
                
                # Save Individual PLYs
                for k, g in enumerate(gaussians_list):
                    frame_idx = indices[k]
                    frame_filename = f"{original_stem}_{unique_id}_f{frame_idx:06d}.ply"
                    
                    # Save to subfolder
                    output_path = work_dir / frame_filename
                    save_ply(g, f_px, (h, w), output_path)
                    if not _active_jobs[job_id]['files']:
                        save_thumbnail(processed_frames[k], output_path)
                    
                    # Append file path relative to OUTPUT_DIR so webui works
                    # ex: "001_myvideo_plys/myvideo_uuid_f0001.ply"
                    relative_path = f"{work_dir.name}/{frame_filename}"
                    
                    _active_jobs[job_id]['files'].append(relative_path)
                    _active_jobs[job_id]['processed_frames'] = frame_idx + 1
                
            except RuntimeError as e:
                # OOM Fallback logic
                if "out of memory" in str(e).lower() and len(frames) > 1:
                    LOGGER.warning(f"OOM detected with batch size {len(frames)}. Switching to batch size 1.")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        torch.mps.empty_cache()
                    # Recursive retry one by one
                    for idx, single_frame in enumerate(frames):
                        process_batch([single_frame], [indices[idx]])
                    return True # Signal that OOM happened
                else:
                    raise
            return False # No OOM

        # Main Loop
        fallback_mode = False
        
        for i, frame in frames_iterator:
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break

            current_batch_frames.append(frame)
            current_batch_indices.append(i)
            
            # If batch full or force fallback
            effective_bs = 1 if fallback_mode else batch_size
            
            if len(current_batch_frames) >= effective_bs:
                oom_occurred = process_batch(current_batch_frames, current_batch_indices)
                if oom_occurred:
                    fallback_mode = True # Permamently switch to 1 for this job
                current_batch_frames = []
                current_batch_indices = []
                
            if i % 10 == 0:
                LOGGER.info(f"Job {job_id}: Processed frame {i+1} / {total_frames}")

        # Process remaining
        if current_batch_frames and not _active_jobs[job_id]['stop_signal']:
            process_batch(current_batch_frames, current_batch_indices)

        if not _active_jobs[job_id]['stop_signal']:
            _active_jobs[job_id]['status'] = 'done'
        elif not _active_jobs[job_id].get('error_msg'):
            frame_count = len(_active_jobs[job_id].get('files') or [])
            frame_label = "frame" if frame_count == 1 else "frames"
            _active_jobs[job_id]['error_msg'] = f"PLY Sequence stopped. Kept {frame_count} generated {frame_label} in Library."

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        if reader:
            try:
                reader.close()
            except Exception:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _process_sbs_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, opacity_threshold, stereo_offset, brightness_factor, batch_size, frame_skip=1):
    """Background worker for SBS Movie generation with batching and unique folders.
    
    Args:
        frame_skip: Process every Nth frame (1=all frames, 2=every 2nd, etc.)
    """
    
    reader = None

    # Verify CUDA before allocating an output directory, but still release the upload.
    if device.type != 'cuda':
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = "SBS Movie generation requires a CUDA GPU; CPU and MPS are not supported."
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                LOGGER.warning("Unable to remove temporary upload %s", tmp_path)
        return

    try:
        job_prefix, work_dir = reserve_numbered_output_dir(original_stem, "frames")
        _active_jobs[job_id]['output_path'] = work_dir.relative_to(OUTPUT_DIR).as_posix()
    except Exception as exc:
        LOGGER.exception("Unable to reserve an SBS Movie frame directory for job %s", job_id)
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                LOGGER.warning("Unable to remove temporary upload %s", tmp_path)
        return
    
    audio_path = work_dir / "audio.aac"
    
    LOGGER.info(f"Job {job_id} [{job_prefix}] | Batch: {batch_size} | Folder: {work_dir}")
    
    try:
        # 1. Extract Audio
        LOGGER.info(f"Job {job_id}: Extracting audio...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        audio_cmd = [
            ffmpeg_exe, "-y", "-i", str(tmp_path), 
            "-vn", "-c:a", "aac", "-b:a", "192k", str(audio_path)
        ]
        has_audio = False
        try:
            subprocess.run(audio_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                has_audio = True
        except subprocess.CalledProcessError:
            LOGGER.warning(f"Job {job_id}: No audio track found or extraction failed.")

        # 2. Setup Processing
        try:
            reader = iio.get_reader(tmp_path)
            total_frames_raw = reader.count_frames()
            # Adjust total_frames to reflect actual frames to render after skip
            total_frames = (total_frames_raw + frame_skip - 1) // frame_skip  # Ceiling division
            _active_jobs[job_id]['total_frames'] = total_frames
            LOGGER.info(f"Job {job_id}: {total_frames_raw} total frames, {total_frames} to render (skip={frame_skip})")
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0
            if reader:
                try:
                    reader.close()
                except Exception:
                    pass
            reader = iio.get_reader(tmp_path)

        # Initialize Renderer
        render_w, render_h = 1920, 1080
        renderer = create_gsplat_renderer(color_space="linearRGB")
            
        png_files = [] # Tuples of (index, path) to ensure sort order if needed, but append is sequential here
        
        # Batch Containers
        batch_frames = []
        batch_indices = []
        
        # Performance: Pre-compute stereo camera extrinsics (used for all frames)
        _cached_ext_left = torch.eye(4, device=device)
        _cached_ext_left[0, 3] = stereo_offset
        _cached_ext_right = torch.eye(4, device=device)
        _cached_ext_right[0, 3] = -stereo_offset
        # Cache intrinsics template (f_px will vary per frame size, but we cache the structure)
        _cached_intrinsics_template = None
        _cached_frame_dims = None

        def process_sbs_batch(frames, indices):
            nonlocal _cached_intrinsics_template, _cached_frame_dims
            try:
                clean_frames = [ensure_rgb_frame(frame) for frame in frames]
                
                h, w = clean_frames[0].shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)
                
                # Predict Batch
                gaussians_list = predict_batch(predictor, clean_frames, f_px, device, use_fp16=use_fp16)
                
                # Performance: Cache intrinsics if frame dimensions match
                if _cached_frame_dims != (h, w):
                    f_px_render = f_px * (render_w / w)
                    _cached_intrinsics_template = torch.tensor([
                        [f_px_render, 0, render_w / 2, 0],
                        [0, f_px_render, render_h / 2, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ], device=device, dtype=torch.float32)
                    _cached_frame_dims = (h, w)
                
                if _cached_intrinsics_template is None:
                    raise RuntimeError("Unable to initialize SBS camera intrinsics.")
                intrinsics = _cached_intrinsics_template
                ext_left = _cached_ext_left
                ext_right = _cached_ext_right
                
                # Render Loop (Rendering is still sequential per gaussian, but prediction was batched)
                # Note: We could technically batch render if gsplat supports it, but gsplat renderer
                # usually takes one scene at a time. The speedup comes from the UNet prediction.
                
                for k, gaussians in enumerate(gaussians_list):
                    idx = indices[k]
                    
                    # Halo Removal
                    if opacity_threshold > 0.0:
                        mask = gaussians.opacities[0] > opacity_threshold
                        gaussians = Gaussians3D(
                            mean_vectors=gaussians.mean_vectors[:, mask],
                            singular_values=gaussians.singular_values[:, mask],
                            quaternions=gaussians.quaternions[:, mask],
                            colors=gaussians.colors[:, mask],
                            opacities=gaussians.opacities[:, mask]
                        )
                    
                    # Render Left
                    out_left = renderer(
                        gaussians, 
                        extrinsics=ext_left.unsqueeze(0), 
                        intrinsics=intrinsics.unsqueeze(0),
                        image_width=render_w, image_height=render_h
                    )
                    img_left_tensor = (out_left.color[0] * brightness_factor).clamp(0, 1)
                    img_left = (img_left_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

                    # Render Right
                    out_right = renderer(
                        gaussians, 
                        extrinsics=ext_right.unsqueeze(0), 
                        intrinsics=intrinsics.unsqueeze(0),
                        image_width=render_w, image_height=render_h
                    )
                    img_right_tensor = (out_right.color[0] * brightness_factor).clamp(0, 1)
                    img_right = (img_right_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

                    # Stitch
                    img_sbs = np.concatenate((img_left, img_right), axis=1)
                    
                    frame_name = f"frame_{idx:05d}.png"
                    frame_path = work_dir / frame_name
                    Image.fromarray(img_sbs).save(frame_path)
                    png_files.append(frame_path)

                    _active_jobs[job_id]['processed_frames'] = idx + 1
                    
                    del gaussians, out_left, out_right, img_left_tensor, img_right_tensor, img_left, img_right, img_sbs
                
            except RuntimeError as e:
                # Fallback logic
                if "out of memory" in str(e).lower() and len(frames) > 1:
                    LOGGER.warning(f"OOM in SBS job with batch {len(frames)}. Switching to 1.")
                    torch.cuda.empty_cache()
                    for i in range(len(frames)):
                        process_sbs_batch([frames[i]], [indices[i]])
                    return True
                else:
                    raise
            return False

        # 3. Frame Loop
        fallback_mode = False
        output_frame_idx = 0  # Separate counter for output frame numbering

        
        for i, frame in enumerate(reader):
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break
            
            # Frame skip logic: only process every Nth frame
            if i % frame_skip != 0:
                continue

            batch_frames.append(frame)
            batch_indices.append(output_frame_idx)  # Use output index for sequential frame naming
            output_frame_idx += 1
            
            effective_bs = 1 if fallback_mode else batch_size
            
            if len(batch_frames) >= effective_bs:
                oom = process_sbs_batch(batch_frames, batch_indices)
                if oom: fallback_mode = True
                batch_frames = []
                batch_indices = []
            
            if i % 10 == 0:
                 LOGGER.info(f"Job {job_id}: SBS Rendered frame {i+1}")
        
        # Remainder
        if batch_frames and not _active_jobs[job_id]['stop_signal']:
            process_sbs_batch(batch_frames, batch_indices)

        # 4. Final Assembly (Allowed even if stopped)
        if png_files:
            # Save Video to root OUTPUT_DIR with prefix
            output_filename = f"{job_prefix}_{original_stem}_SBS.mp4"
            output_path = OUTPUT_DIR / output_filename
            _active_jobs[job_id]['output_path'] = output_filename
            LOGGER.info(f"Job {job_id}: Encoding SBS Movie to {output_filename}...")
            
            # Adjust output FPS based on frame_skip to maintain original video duration
            output_fps = fps / frame_skip
            LOGGER.info(f"Job {job_id}: Output FPS adjusted from {fps} to {output_fps} (frame_skip={frame_skip})")
            
            input_pattern = str(work_dir / "frame_%05d.png")
            cmd = [ffmpeg_exe, "-y", "-framerate", str(output_fps), "-i", input_pattern]
            if has_audio: cmd.extend(["-i", str(audio_path)])
            cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast"])
            
            # If audio exists, shortest cuts it to video length (useful if we stopped early)
            if has_audio: cmd.append("-shortest")
            
            cmd.append(str(output_path))
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                _active_jobs[job_id]['files'].append(output_filename)
                if _active_jobs[job_id]['stop_signal']:
                    _active_jobs[job_id]['status'] = 'stopped'
                    _active_jobs[job_id]['error_msg'] = f"SBS Movie stopped. A partial result is available in Library: {output_filename}."
                else:
                    _active_jobs[job_id]['status'] = 'done'
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"FFmpeg encoding failed: {e.stderr.decode()}")
                _active_jobs[job_id]['status'] = 'error'
                _active_jobs[job_id]['error_msg'] = "Unable to encode the SBS Movie."
        elif _active_jobs[job_id]['stop_signal']:
             _active_jobs[job_id]['status'] = 'stopped'
             _active_jobs[job_id]['error_msg'] = "SBS Movie stopped before any frames were generated."
        elif not _active_jobs[job_id]['stop_signal']:
             _active_jobs[job_id]['status'] = 'error'
             _active_jobs[job_id]['error_msg'] = "No SBS Movie frames were generated."

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        if reader:
            try:
                reader.close()
            except Exception:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.route("/preview_sbs_frame", methods=["POST"])
def preview_sbs_frame():
    """Generate a single SBS preview frame from a video for testing settings."""
    if "video" not in request.files:
        return jsonify({"error": "No video selected."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video selected."}), 400

    try:
        frame_number = int(request.form.get('frame_number', 0))
        if frame_number < 0:
            raise ValueError("Frame number must be zero or greater.")
        opacity_threshold = parse_finite_form_float('opacity_threshold', 0.0, 0.0, 1.0)
        stereo_offset = parse_finite_form_float('stereo_offset', 0.015, 0.0, 0.2)
        brightness_factor = parse_finite_form_float('brightness_factor', 1.0, 0.1, 3.0)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    quality = request.form.get('quality', 'balanced')
    if quality not in {'fast', 'balanced'}:
        return jsonify({"error": "Unsupported quality mode."}), 400
    use_fp16 = (quality == 'fast')

    LOGGER.info(f"SBS Preview: frame={frame_number}, opacity={opacity_threshold}, offset={stereo_offset}, brightness={brightness_factor}")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}."}), 400

    tmp_path = None
    out_tmp_path = None
    reader = None
    try:
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            file.save(tmp.name)

        # Get predictor and device
        predictor, device = get_predictor()
        
        # Verify CUDA for rendering
        if device.type != 'cuda':
            return jsonify({"error": "SBS Preview requires a CUDA GPU."}), 400

        # Open video and extract frame
        reader = iio.get_reader(tmp_path)
        try:
            frame = reader.get_data(frame_number)
        except IndexError:
            return jsonify({"error": f"Frame {frame_number} is out of range."}), 400

        frame = ensure_rgb_frame(frame)

        h, w = frame.shape[:2]
        f_px = io.convert_focallength(w, h, 30.0)

        # Predict Gaussians
        gaussians = predict_image(predictor, frame, f_px, device, use_fp16=use_fp16)

        # Apply halo removal
        if opacity_threshold > 0.0:
            mask = gaussians.opacities[0] > opacity_threshold
            gaussians = Gaussians3D(
                mean_vectors=gaussians.mean_vectors[:, mask],
                singular_values=gaussians.singular_values[:, mask],
                quaternions=gaussians.quaternions[:, mask],
                colors=gaussians.colors[:, mask],
                opacities=gaussians.opacities[:, mask]
            )

        # Setup rendering
        render_w, render_h = 1920, 1080
        renderer = create_gsplat_renderer(color_space="linearRGB")

        f_px_render = f_px * (render_w / w)
        intrinsics = torch.tensor([
            [f_px_render, 0, render_w / 2, 0],
            [0, f_px_render, render_h / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], device=device, dtype=torch.float32)

        ext_left = torch.eye(4, device=device)
        ext_left[0, 3] = stereo_offset

        ext_right = torch.eye(4, device=device)
        ext_right[0, 3] = -stereo_offset

        # Render left eye
        out_left = renderer(
            gaussians,
            extrinsics=ext_left.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            image_width=render_w, image_height=render_h
        )
        img_left_tensor = (out_left.color[0] * brightness_factor).clamp(0, 1)
        img_left = (img_left_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

        # Render right eye
        out_right = renderer(
            gaussians,
            extrinsics=ext_right.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            image_width=render_w, image_height=render_h
        )
        img_right_tensor = (out_right.color[0] * brightness_factor).clamp(0, 1)
        img_right = (img_right_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()

        # Stitch side-by-side (3840x1080)
        img_sbs = np.concatenate((img_left, img_right), axis=1)

        # Cleanup GPU memory
        del gaussians, out_left, out_right, img_left_tensor, img_right_tensor

        # Save to temp file and return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as out_tmp:
            out_tmp_path = Path(out_tmp.name)
            Image.fromarray(img_sbs).save(out_tmp.name, "JPEG", quality=90)

        response = send_file(
            out_tmp_path,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
        # Schedule cleanup after response (Flask handles this)
        @response.call_on_close
        def cleanup():
            try:
                out_tmp_path.unlink()
            except OSError:
                pass

        return response

    except Exception as e:
        LOGGER.exception("Error generating SBS preview")
        if out_tmp_path and out_tmp_path.exists():
            try:
                out_tmp_path.unlink()
            except OSError:
                pass
        return jsonify({"error": str(e)}), 500
    finally:
        if reader:
            try:
                reader.close()
            except Exception:
                pass
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@app.route("/generate_video", methods=["POST"])
def generate_video():
    """Start async video generation."""
    if "video" not in request.files:
        return jsonify({"error": "No video selected."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No video selected."}), 400

    quality = request.form.get('quality', 'balanced')
    if quality not in {'fast', 'balanced'}:
        return jsonify({"error": "Unsupported quality mode."}), 400
    use_fp16 = (quality == 'fast')
    output_mode = request.form.get('output_mode', 'ply_seq')
    if output_mode not in {'ply_seq', 'sbs_movie'}:
        return jsonify({"error": "Unsupported video output mode."}), 400

    try:
        opacity_threshold = parse_finite_form_float('opacity_threshold', 0.0, 0.0, 1.0)
        stereo_offset = parse_finite_form_float('stereo_offset', 0.015, 0.0, 0.2)
        brightness_factor = parse_finite_form_float('brightness_factor', 1.0, 0.1, 3.0)
        batch_size = int(request.form.get('batch_size', 1))
        frame_skip = int(request.form.get('frame_skip', 1))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Invalid video setting."}), 400

    if not 1 <= batch_size <= 8:
        return jsonify({"error": "Batch size must be between 1 and 8."}), 400
    if not 1 <= frame_skip <= 5:
        return jsonify({"error": "Frame skip must be between 1 and 5."}), 400

    LOGGER.info(f"Starting video generation | Mode: {output_mode} | Batch Size: {batch_size} | Frame Skip: {frame_skip} | Quality: {quality}")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}."}), 400

    tmp_path = None
    job_id = None
    worker_started = False
    metadata_reader = None

    try:
        unique_id = str(uuid.uuid4())[:8]
        original_stem = sanitize_output_stem(file.filename, "video")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            file.save(tmp.name)

        try:
            metadata_reader = iio.get_reader(tmp_path)
            meta = metadata_reader.get_meta_data()
            fps = float(meta.get('fps', 30.0))
            if not math.isfinite(fps) or fps <= 0 or fps > 240:
                fps = 30.0
        except Exception:
            fps = 30.0
        finally:
            if metadata_reader:
                try:
                    metadata_reader.close()
                except Exception:
                    pass

        predictor, device = get_predictor()
        if output_mode == 'sbs_movie' and device.type != 'cuda':
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    LOGGER.warning("Unable to remove temporary upload %s", tmp_path)
            return jsonify({"error": "SBS Movie generation requires a CUDA GPU."}), 400

        job_id = str(uuid.uuid4())
        _active_jobs[job_id] = {
            "status": "running",
            "files": [],
            "total_frames": 0,
            "processed_frames": 0,
            "stop_signal": False,
            "error_msg": "",
            "fps": fps,
            "mode": output_mode
        }

        if output_mode == 'sbs_movie':
            thread = threading.Thread(
                target=_process_sbs_video_job,
                args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, opacity_threshold, stereo_offset, brightness_factor, batch_size, frame_skip),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=_process_video_job,
                args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16, batch_size),
                daemon=True,
            )

        thread.start()
        worker_started = True

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "running"
        })

    except Exception as e:
        LOGGER.exception("Error starting video generation")
        if job_id and not worker_started:
            _active_jobs.pop(job_id, None)
        if tmp_path and not worker_started and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                LOGGER.warning("Unable to remove temporary upload %s", tmp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/job_status/<job_id>")
def job_status(job_id):
    """Check status of a background job."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Generation job not found."}), 404
    
    return jsonify({
        "status": job["status"],
        "processed": job["processed_frames"],
        "total": job["total_frames"],
        "files": job["files"], # Returns full list so client can see what's new
        "fps": job["fps"],
        "error": job["error_msg"],
        "mode": job.get("mode", "ply_seq"),
        "base_url": "/download/" if job.get("mode") == "sbs_movie" else "/ply/"
    })


@app.route("/stop_job/<job_id>", methods=["POST"])
def stop_job(job_id):
    """Signal a job to stop."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Generation job not found."}), 404
    
    job["stop_signal"] = True
    return jsonify({"success": True, "status": "stopping"})


@app.route("/download/<path:filename>")
def download(filename: str):
    """Download a generated file."""
    file_path = resolve_output_file(filename)
    if file_path is None or not file_path.is_file():
        return jsonify({"error": "File not found."}), 404

    # Determine mimetype based on extension
    mime = "application/octet-stream"
    if file_path.suffix.lower() == ".mp4":
        mime = "video/mp4"

    return send_file(
        file_path,
        as_attachment=True,
        download_name=Path(filename).name,
        mimetype=mime,
    )


@app.route("/ply/<path:filename>")
def serve_ply(filename: str):
    """Serve a PLY file for the viewer."""
    file_path = resolve_output_file(filename)
    if file_path is None or not file_path.is_file() or file_path.suffix.lower() != ".ply":
        return jsonify({"error": "File not found."}), 404

    return send_file(
        file_path,
        mimetype="application/octet-stream",
    )


@app.route("/status")
def status():
    """Get server status."""
    device = get_device()
    model_loaded = _model_cache["predictor"] is not None
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
    })


@app.route("/clear_memory_cache", methods=["POST"])
def clear_memory_cache():
    """Clear finished in-memory job records without deleting output files."""
    result = clear_finished_memory_jobs()
    return jsonify({"success": True, **result})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ml-sharp WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")

    args = parser.parse_args()

    if args.preload:
        LOGGER.info("Preloading model...")
        get_predictor()

    LOGGER.info(f"Starting WebUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

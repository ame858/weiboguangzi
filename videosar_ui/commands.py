"""Command construction and process execution for the GUI.

The GUI only ever shells out to ``python -m videosar ...`` with the exact flag
names defined in videosar/__main__.py. It never imports or edits core logic.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEMO_CONFIG = REPO_ROOT / "examples" / "urban-shadow.json"
SAR_CONFIG = REPO_ROOT / "examples" / "ka-simulation.json"
UNET_MODEL = REPO_ROOT / "deliverables" / "proposal" / "models" / "unet_temporal_post.json"
SNL_MODEL = REPO_ROOT / "deliverables" / "snl-causal-study" / "models" / "stacked-center.json"

FFMPEG_INSTALL = r".\.venv\Scripts\python.exe -m pip install imageio imageio-ffmpeg"
TORCH_INSTALL = (r".\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 "
                 r"--index-url https://download.pytorch.org/whl/cpu")


def gui_run_dir(task, root=REPO_ROOT):
    """All GUI-generated artefacts go under runs/gui/<timestamp>_<task>/."""
    # Milliseconds keep repeated runs in the same second from colliding:
    # run_directory() deliberately refuses to reuse an existing directory.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    return Path(root) / "runs" / "gui" / f"{stamp}_{task}"


def _python():
    return sys.executable


def demo_command(config, frames, output_dir):
    return [_python(), "-m", "videosar", "demo",
            "--config", str(config), "--frames", str(int(frames)),
            "--output-dir", str(output_dir)]


def sar_chain_commands(bandwidth_hz, method, model, root):
    """image -> detect -> track -> evaluate for one live SAR run."""
    image_dir, detect_dir = root / "image", root / "detect"
    track_dir, evaluate_dir = root / "track", root / "evaluate"
    commands = [
        [_python(), "-m", "videosar", "image", "--config", str(SAR_CONFIG),
         "--bandwidth-hz", str(int(bandwidth_hz)), "--output-dir", str(image_dir)],
        [_python(), "-m", "videosar", "detect", "--input", str(image_dir),
         "--method", method, *(["--model", str(model)] if method == "learned" else []),
         "--output-dir", str(detect_dir)],
        [_python(), "-m", "videosar", "track", "--input", str(image_dir),
         "--predictions", str(detect_dir), "--output-dir", str(track_dir)],
        [_python(), "-m", "videosar", "evaluate", "--input", str(image_dir),
         "--predictions", str(detect_dir), "--tracks", str(track_dir),
         "--match-iou", "0.50", "--output-dir", str(evaluate_dir)],
    ]
    return {"commands": commands, "sequence": image_dir, "predictions": detect_dir,
            "tracks": track_dir, "metrics": evaluate_dir}


def compare_commands(sequence_dir, root):
    """Run traditional and learned detection on the same frames.npy."""
    traditional = root / "detect-threshold"
    learned = root / "detect-learned"
    return {
        "commands": [
            [_python(), "-m", "videosar", "detect", "--input", str(sequence_dir),
             "--method", "threshold", "--output-dir", str(traditional)],
            [_python(), "-m", "videosar", "detect", "--input", str(sequence_dir),
             "--method", "learned", "--model", str(UNET_MODEL), "--output-dir", str(learned)],
        ],
        "predictions": [traditional, learned],
    }


def detect_video_command(input_path, output_dir, crop=None, start_frame=0,
                         max_frames=24, model=None, ffmpeg=None):
    command = [_python(), "-m", "videosar", "detect-video", "--input", str(input_path)]
    if model:
        command += ["--model", str(model)]
    if ffmpeg:
        command += ["--ffmpeg", str(ffmpeg)]
    if crop:
        command += ["--crop", *(str(int(value)) for value in crop)]
    command += ["--start-frame", str(int(start_frame)),
                "--max-frames", str(int(max_frames)), "--output-dir", str(output_dir)]
    return command


def run_commands(commands, cwd, emit):
    """Run commands in order, streaming combined stdout/stderr to ``emit``."""
    for command in commands:
        emit("> " + " ".join(command[2:]))
        try:
            process = subprocess.Popen(
                command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as error:
            emit(f"[启动失败] {error}")
            return False
        with process:  # Popen.__exit__ closes the pipe and waits for the child
            assert process.stdout is not None
            for line in process.stdout:
                emit(line.rstrip())
            code = process.wait()
        emit(f"[退出码 {code}]")
        if code != 0:
            return False
    return True


def ffmpeg_path():
    found = shutil.which("ffmpeg")
    if found:
        return found
    if find_spec("imageio_ffmpeg"):
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, OSError):
            return None
    return None


def environment_report():
    return {
        "ffmpeg": ffmpeg_path(),
        "torch": bool(find_spec("torch")),
        "torchvision": bool(find_spec("torchvision")),
        "imageio": bool(find_spec("imageio")),
        "imageio_ffmpeg": bool(find_spec("imageio_ffmpeg")),
        "python": sys.executable,
    }


def extract_first_frame(video, ffmpeg, destination):
    """Decode the first video frame to a PNG for the --crop preview."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ffmpeg), "-nostdin", "-v", "error", "-i", str(video),
               "-frames:v", "1", "-y", str(target)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not target.is_file():
        return None
    return target

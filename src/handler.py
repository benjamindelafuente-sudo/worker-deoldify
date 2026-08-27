"""DeOldify ONNX Video colorizer — RunPod Serverless worker.

Uses the ONNX export of DeOldify (jantic/DeOldify → ONNX via
instant-high/deoldify-onnx). NO fastai, NO wandb, just torch +
onnxruntime. Fits cleanly in a slim Docker image.

Bake strategy:
- Base: runpod/pytorch:2.4.0 (torch 2.4 + CUDA 12.4)
- Install: opencv-python-headless, onnxruntime-gpu, Pillow, requests
- Bake the ONNX model (~122 MB) into the image

INPUT (job["input"]):
    {
        "video_url": "https://...   |   s3://bucket/key",
        "render_factor": 21,         # opcional, default 21
        "watermark": False,          # opcional, default False
    }

OUTPUT:
    {
        "output_url": "https://litter.catbox.moe/...mp4",
        "duration_sec": 5.0,
        "frames_processed": 120,
        "elapsed_sec": 32.4
    }
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort

import runpod
from rp_schemas import INPUT_SCHEMA, OUTPUT_SCHEMA


# ---------------------------------------------------------------------------
# Init: load ONNX session ONCE per worker (1-2 sec)
# ---------------------------------------------------------------------------
print("[init] Loading DeOldify ONNX Video colorizer...", flush=True)
_init_t = time.time()

MODEL_PATH = Path("/app/models/deoldify_fp16.onnx")
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

so = ort.SessionOptions()
so.intra_op_num_threads = 2  # each worker uses 2 threads; 4 parallel workers = 8 threads
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
INP_NAME = session.get_inputs()[0].name
print(f"[init] ONNX session loaded in {time.time()-_init_t:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# ONNX colorizer (mirrors the sandbox implementation that worked)
# ---------------------------------------------------------------------------

def colorize_frame(bgr: np.ndarray, render_size: int = 256) -> np.ndarray:
    """Colorize a single BGR frame (uint8, HxWx3)."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    target_LAB = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    target_L = target_LAB[:, :, 0]

    rgb_3ch = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    rgb_resized = cv2.resize(rgb_3ch, (render_size, render_size))
    x = rgb_resized.astype(np.float16).transpose(2, 0, 1)[None, ...]

    y = session.run(None, {INP_NAME: x})[0][0]
    out_hwc = y.transpose(1, 2, 0).astype(np.float32)
    out_rgb = cv2.cvtColor(out_hwc, cv2.COLOR_BGR2RGB).astype(np.uint8)
    out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    out_rgb = cv2.GaussianBlur(out_rgb, (13, 13), 0)
    out_LAB = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB)
    _, A, B = cv2.split(out_LAB)
    merged = cv2.merge((target_L, A, B))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _run(cmd, *, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(map(str, cmd[:6]))} — {(r.stderr or r.stdout or '')[-1500:]}")
    return r


def _upload_to_litterbox(local_path: Path, expiry: str = "24h") -> str:
    import uuid
    boundary = f"----x{uuid.uuid4().hex}"
    body = []
    for k, v in (("reqtype", "fileupload"), ("time", expiry)):
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"fileToUpload\"; "
        f"filename=\"{local_path.name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode()
    )
    with open(local_path, "rb") as f:
        body.append(f.read())
    body.append(f"\r\n--{boundary}--\r\n".encode())
    req = requests.post(
        "https://litterbox.catbox.moe/resources/internals/api.php",
        data=b"".join(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=180,
    )
    url = req.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"litterbox returned non-URL: {url!r}")
    return url


def _download(url: str, dst: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)


def _probe_fps(path: Path) -> float:
    r = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1", str(path),
    ])
    raw = r.stdout.strip()
    if "/" in raw:
        n, d = raw.split("=")[1].split("/")
        return float(n) / float(d)
    return float(raw)


def _probe_duration(path: Path) -> float:
    r = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nokey=1", str(path),
    ])
    return float(r.stdout.strip())


def colorize_video(in_path: Path, out_path: Path, *,
                   render_factor: int, watermark: bool,
                   job=None) -> tuple[int, float]:
    """Extract frames with ffmpeg, colorize each via ONNX, recompose."""
    t0 = time.time()
    fps = _probe_fps(in_path)

    work = Path(tempfile.mkdtemp(prefix="colorize_"))
    frames_in = work / "in"
    frames_out = work / "out"
    frames_in.mkdir()
    frames_out.mkdir()

    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-vf", "scale=iw:ih",
        str(frames_in / "%06d.png"),
    ], timeout=120)

    frames = sorted(frames_in.glob("*.png"))
    n = len(frames)
    if n == 0:
        raise RuntimeError("no frames extracted")

    if job is not None:
        runpod.serverless.progress_update(job, f"extracted {n} frames @ {fps:.2f}fps")

    for i, fp in enumerate(frames, 1):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        col = colorize_frame(bgr, render_size=render_factor)
        cv2.imwrite(str(frames_out / fp.name), col, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if job is not None and i % 10 == 0:
            runpod.serverless.progress_update(job, f"colorized {i}/{n}")

    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_out / "%06d.png"),
        "-i", str(in_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ], timeout=300)

    # watermark param accepted but not implemented (would burn text via ffmpeg drawtext)
    del watermark

    return n, time.time() - t0


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job):
    job_input = job["input"]
    job_id = job.get("id", "unknown")

    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "missing 'video_url' in input"}

    render_factor = int(job_input.get("render_factor", 21))
    watermark = bool(job_input.get("watermark", False))

    work = Path(tempfile.mkdtemp(prefix="job_"))
    in_path = work / "in.mp4"
    out_path = work / "out.mp4"

    try:
        print(f"[job {job_id}] downloading {video_url[:80]}...", flush=True)
        _download(video_url, in_path)
        print(f"[job {job_id}] downloaded {in_path.stat().st_size//1024} KB", flush=True)

        print(f"[job {job_id}] colorizing render={render_factor}...", flush=True)
        n_frames, elapsed = colorize_video(
            in_path, out_path,
            render_factor=render_factor,
            watermark=watermark,
            job=job,
        )
        dur = _probe_duration(out_path)
        print(f"[job {job_id}] done: {n_frames} frames, {elapsed:.1f}s wall, {dur:.1f}s video", flush=True)

        print(f"[job {job_id}] uploading to litterbox...", flush=True)
        output_url = _upload_to_litterbox(out_path, expiry="24h")

        return {
            "output_url": output_url,
            "duration_sec": round(dur, 2),
            "frames_processed": n_frames,
            "elapsed_sec": round(elapsed, 1),
            "size_bytes": out_path.stat().st_size,
        }
    except Exception as e:
        print(f"[job {job_id}] ERROR: {e}", flush=True)
        return {"error": str(e), "job_id": job_id}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


print("[handler] starting RunPod serverless", flush=True)
runpod.serverless.start({"handler": handler})
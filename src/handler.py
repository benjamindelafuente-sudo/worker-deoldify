"""worker-deoldify-video — RunPod Serverless handler.

Colorizes B&W video clips with the DeOldify Video model (Artistic=false),
which is optimized for flicker-free temporal consistency (better than the
Artistic model for video).

INPUT (job["input"]):
    {
        "video_url": "https://...   |   s3://bucket/key",
        "model":     "Video",        # opcional, default "Video"
        "render_factor": 21,         # opcional, default 21
        "watermark": False,          # opcional, default False
    }

OUTPUT:
    {
        "output_url": "https://litter.catbox.moe/...mp4",
        "duration_sec": 5.0,
        "frames_processed": 120,
        "model": "Video",
        "elapsed_sec": 32.4
    }

PIPELINE:
    1. download video from URL to /tmp
    2. extract frames with ffmpeg
    3. colorize each frame with DeOldify Video
    4. re-encode frames back to mp4 with original audio
    5. upload to litterbox.catbox.moe (24h expiry), return URL
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

# DeOldify lives at /DeOldify (baked in by Dockerfile).
sys.path.insert(0, "/DeOldify")
sys.path.insert(0, "/app")

from deoldify import device as deoldify_device  # noqa: E402
from deoldify.device_id import DeviceId  # noqa: E402
deoldify_device.set(device=DeviceId.GPU0)

from deoldify.visualize import get_image_colorizer  # noqa: E402

import runpod  # noqa: E402
from rp_schemas import INPUT_SCHEMA, OUTPUT_SCHEMA  # noqa: E402


# ---------------------------------------------------------------------------
# Init: load colorizer ONCE per worker
# ---------------------------------------------------------------------------
print("[init] Cargando DeOldify Video colorizer (1 sola vez por worker)...", flush=True)
_init_t = time.time()
colorizer = get_image_colorizer(artistic=False)  # artistic=False = Video model
print(f"[init] Colorizer listo en {time.time()-_init_t:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd, *, timeout=300, capture=True):
    """Run subprocess, raise on non-zero."""
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "<no output>")[-2000:]
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(map(str, cmd[:6]))}... — {err}")
    return r


def _upload_to_litterbox(local_path: Path, expiry: str = "24h") -> str:
    """Upload to litterbox.catbox.moe via multipart form-data.
    Returns the public URL.
    """
    import urllib.request
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

    req = urllib.request.Request(
        "https://litterbox.catbox.moe/resources/internals/api.php",
        data=b"".join(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        url = resp.read().decode().strip()
    if not url.startswith("http"):
        raise RuntimeError(f"litterbox returned non-URL: {url!r}")
    return url


def _download(url: str, dst: Path) -> Path:
    """Download a video from http(s) URL to dst. Streams to disk."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "worker-deoldify-video/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dst


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


def _colorize_one(pil_path: Path, out_path: Path, render_factor: int, watermark: bool):
    """Colorize a single frame with DeOldify Video model."""
    pil_image = colorizer.get_transformed_image(
        str(pil_path),
        render_factor=render_factor,
        watermarked=watermark,
    )
    pil_image.save(str(out_path))


def colorize_video(in_path: Path, out_path: Path, *,
                   render_factor: int, watermark: bool,
                   job=None) -> tuple[int, float]:
    """Extract frames, colorize each, recompose. Returns (n_frames, elapsed_sec)."""
    t0 = time.time()
    fps = _probe_fps(in_path)

    work = Path(tempfile.mkdtemp(prefix="colorize_"))
    frames_in = work / "in"
    frames_out = work / "out"
    frames_in.mkdir()
    frames_out.mkdir()

    # 1) Extract frames
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

    # 2) Colorize each
    for i, fp in enumerate(frames, 1):
        _colorize_one(fp, frames_out / fp.name, render_factor, watermark)
        if job is not None and i % 10 == 0:
            runpod.serverless.progress_update(job, f"colorized {i}/{n}")

    # 3) Recompose with original audio
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

    model = job_input.get("model", "Video")
    render_factor = int(job_input.get("render_factor", 21))
    watermark = bool(job_input.get("watermark", False))

    work = Path(tempfile.mkdtemp(prefix="job_"))
    in_path = work / "in.mp4"
    out_path = work / "out.mp4"

    try:
        print(f"[job {job_id}] downloading {video_url[:80]}...", flush=True)
        _download(video_url, in_path)
        print(f"[job {job_id}] downloaded {in_path.stat().st_size//1024} KB", flush=True)

        print(f"[job {job_id}] colorizing with model={model} render={render_factor}...", flush=True)
        n_frames, elapsed = colorize_video(
            in_path, out_path,
            render_factor=render_factor,
            watermark=watermark,
            job=job,
        )
        dur = _probe_duration(out_path)
        print(f"[job {job_id}] done: {n_frames} frames, {elapsed:.1f}s, {dur:.1f}s video", flush=True)

        print(f"[job {job_id}] uploading to litterbox...", flush=True)
        output_url = _upload_to_litterbox(out_path, expiry="24h")

        return {
            "output_url": output_url,
            "duration_sec": round(dur, 2),
            "frames_processed": n_frames,
            "model": model,
            "elapsed_sec": round(elapsed, 1),
            "size_bytes": out_path.stat().st_size,
        }
    except Exception as e:
        print(f"[job {job_id}] ERROR: {e}", flush=True)
        return {"error": str(e), "job_id": job_id}
    finally:
        # Cleanup workdir
        import shutil
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

print("[handler] starting RunPod serverless", flush=True)
runpod.serverless.start({"handler": handler})
"""Minimalist DeOldify Video colorizer."""
import os, sys, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
import requests

print("[handler] starting", flush=True)

# Load ONNX model
MODEL_PATH = Path("/app/models/deoldify_fp16.onnx")
print(f"[handler] loading model from {MODEL_PATH}", flush=True)
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
SESSION = ort.InferenceSession(
    str(MODEL_PATH), sess_options=so,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
INP_NAME = SESSION.get_inputs()[0].name
print(f"[handler] model loaded, input={INP_NAME}", flush=True)


def colorize_frame(bgr):
    h, w = bgr.shape[:2]
    target_lab = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2LAB)
    target_l = target_lab[:, :, 0]
    gray_rgb = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    x = cv2.resize(gray_rgb, (256, 256)).astype(np.float16).transpose(2, 0, 1)[None, ...]
    y = SESSION.run(None, {INP_NAME: x})[0][0]
    out_rgb = cv2.cvtColor(y.transpose(1, 2, 0).astype(np.float32), cv2.COLOR_BGR2RGB).astype(np.uint8)
    out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    out_rgb = cv2.GaussianBlur(out_rgb, (13, 13), 0)
    _, A, B = cv2.split(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB))
    return cv2.cvtColor(cv2.merge((target_l, A, B)), cv2.COLOR_LAB2BGR)


def upload_litterbox(p, expiry="24h"):
    import uuid
    boundary = f"--x{uuid.uuid4().hex}"
    body = []
    for k, v in (("reqtype", "fileupload"), ("time", expiry)):
        body.append(f"{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.append(f"{boundary}\r\nContent-Disposition: form-data; name=\"fileToUpload\"; filename=\"{p.name}\"\r\nContent-Type: video/mp4\r\n\r\n".encode())
    with open(p, "rb") as f:
        body.append(f.read())
    body.append(f"\r\n{boundary}--\r\n".encode())
    r = requests.post("https://litterbox.catbox.moe/resources/internals/api.php",
                      data=b"".join(body),
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=180)
    url = r.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"bad url: {url!r}")
    return url


def run(cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd fail: {' '.join(cmd[:6])}\n{(r.stderr or r.stdout or '')[-1000:]}")
    return r


def handler(job):
    job_input = job["input"]
    job_id = job.get("id", "?")
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "missing video_url"}

    work = Path(tempfile.mkdtemp(prefix="job_"))
    in_path = work / "in.mp4"
    out_path = work / "out.mp4"
    try:
        print(f"[{job_id}] downloading {video_url[:60]}", flush=True)
        with requests.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(in_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64*1024):
                    if chunk:
                        f.write(chunk)

        print(f"[{job_id}] extracting frames", flush=True)
        frames_in = work / "in"; frames_out = work / "out"
        frames_in.mkdir(); frames_out.mkdir()
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_path),
             "-vf", "scale=iw:ih", str(frames_in / "%06d.png")], timeout=120)
        frames = sorted(frames_in.glob("*.png"))
        n = len(frames)
        print(f"[{job_id}] {n} frames to colorize", flush=True)
        if n == 0:
            return {"error": "no frames extracted"}

        for i, fp in enumerate(frames, 1):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            cv2.imwrite(str(frames_out / fp.name), colorize_frame(bgr),
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if i % 10 == 0:
                print(f"[{job_id}] colorized {i}/{n}", flush=True)

        print(f"[{job_id}] recomposing", flush=True)
        run(["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", "24", "-i", str(frames_out / "%06d.png"),
             "-i", str(in_path),
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-b:a", "128k",
             "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-shortest", str(out_path)], timeout=300)

        output_url = upload_litterbox(out_path)
        return {"output_url": output_url, "frames_processed": n, "size_bytes": out_path.stat().st_size}
    except Exception as e:
        print(f"[{job_id}] ERROR: {e}", flush=True)
        return {"error": str(e)}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


print("[handler] importing runpod", flush=True)
import runpod
print(f"[handler] runpod version: {runpod.__version__}", flush=True)
print("[handler] starting serverless", flush=True)
runpod.serverless.start({"handler": handler})

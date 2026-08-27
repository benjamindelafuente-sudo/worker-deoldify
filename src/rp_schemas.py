"""Input/output schemas for the video colorizer worker."""


INPUT_SCHEMA = {
    "video_url": {
        "type": str,
        "required": True,
        "description": "Public http(s) URL to a B&W video (mp4/mov/webm).",
    },
    "render_factor": {
        "type": int,
        "required": False,
        "default": 21,
        "description": "Render factor (10-40). Higher = more detail. 21 default.",
    },
    "watermark": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "If true, burn a small watermark (not yet implemented).",
    },
}


OUTPUT_SCHEMA = {
    "output_url": {"type": str, "description": "Public URL to the colorized video (24h expiry)."},
    "duration_sec": {"type": float, "description": "Output duration."},
    "frames_processed": {"type": int, "description": "Frames colorized."},
    "elapsed_sec": {"type": float, "description": "Colorize wall time (excludes download/upload)."},
    "size_bytes": {"type": int, "description": "Output MP4 size in bytes."},
}

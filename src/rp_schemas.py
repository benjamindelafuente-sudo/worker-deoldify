"""Input/output schemas for worker-deoldify-video.

These are used by runpod's rp_schemas validator (optional but helpful
for catching malformed input early and providing a clean OpenAPI spec).
"""


INPUT_SCHEMA = {
    "video_url": {
        "type": str,
        "required": True,
        "description": "Public http(s) URL to a B&W video (mp4/mov/webm). Will be downloaded.",
    },
    "model": {
        "type": str,
        "required": False,
        "default": "Video",
        "description": "DeOldify model name. Currently only 'Video' (flicker-free) is supported; 'Stable' and 'Artistic' accepted but map to Video for now.",
    },
    "render_factor": {
        "type": int,
        "required": False,
        "default": 21,
        "description": "Render factor (10-40). Higher = more detail but slower. 21 is the sweet spot.",
    },
    "watermark": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "If true, embeds the DeOldify watermark on the output frames. Default false.",
    },
}


OUTPUT_SCHEMA = {
    "output_url": {
        "type": str,
        "description": "Public URL to the colorized video (24h expiry on litterbox.catbox.moe).",
    },
    "duration_sec": {
        "type": float,
        "description": "Duration of the output video in seconds.",
    },
    "frames_processed": {
        "type": int,
        "description": "Number of frames colorized.",
    },
    "model": {
        "type": str,
        "description": "The model used (always 'Video' for now).",
    },
    "elapsed_sec": {
        "type": float,
        "description": "Wall time spent in the colorize step (excludes download/upload).",
    },
    "size_bytes": {
        "type": int,
        "description": "Size of the output MP4 in bytes.",
    },
}
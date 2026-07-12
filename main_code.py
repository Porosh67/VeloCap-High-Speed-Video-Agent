"""
VeloCap — High-Speed Video Captioning Agent (v4)
AMD Developer Hackathon (ACT II) — Track 2

Single-model pipeline: Minimax M3 (vision) looks at the sampled frames and
produces all four styled captions directly, in one call. No second model,
no optional "polish" stage — that experiment is removed. This is the same
proven approach that worked for you originally, with only these kept:

  - Frame extraction tries to stream directly from the video URL first (no
    full download to disk), falling back to a temp-file download only if
    that isn't possible. Both paths were already tested working in your logs.
  - A slightly larger frame sample (24 vs the original 20) — this was
    already confirmed working and faster in your own test run.
  - Tighter, more explicit prompting and stricter validation (rejects
    duplicate/degenerate captions and retries).
"""

import sys
import os
import json
import base64
import time
import re
import random
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cv2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VeloCapAgent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("FIREWORKS_API_KEY")
if not API_KEY:
    logger.error("FATAL: FIREWORKS_API_KEY is not set in environment or .env file.")
    sys.exit(1)

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

# Single model, vision-capable, does all the work in one call.
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/minimax-m3")

NUM_FRAMES = int(os.getenv("NUM_FRAMES", "24"))
FRAME_SIZE = (640, 360)
JPEG_QUALITY = 78
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

STYLE_GUIDE = {
    "formal": "Professional, objective, factual tone.",
    "sarcastic": "Dry, ironic, lightly mocking tone.",
    "humorous_tech": "Funny, weaving in technology or programming references.",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon.",
}

# A session with connection pooling + retries is used ONLY for downloading
# video files (plain GET, safe to retry on flaky network). API calls to
# Fireworks use plain requests.post with our own retry loop below, so a
# single slow/failed call can't silently balloon into minutes of hidden
# urllib3-level retries.
DOWNLOAD_SESSION = requests.Session()
_retry = Retry(
    total=2,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=20, pool_maxsize=20)
DOWNLOAD_SESSION.mount("https://", _adapter)
DOWNLOAD_SESSION.mount("http://", _adapter)


def clean_json_string(text):
    """Strip markdown code fences so json.loads gets clean JSON."""
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def extract_json_object(text):
    """Pull the outermost {...} out of a response, tolerating any stray
    text a model puts before or after the JSON."""
    text = clean_json_string(text)
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return text[start:end + 1]


def call_fireworks(payload, timeout):
    """Plain, single-shot POST — no hidden auto-retry layer. Retries are
    handled explicitly in generate_captions() so timing stays predictable."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Frame extraction — streams directly from the URL when possible, avoiding a
# full download of large 4K clips. Falls back to download-then-extract.
# ---------------------------------------------------------------------------
def _encode_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buffer).decode('utf-8')


def _download_video(video_url, task_id):
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with DOWNLOAD_SESSION.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return tmp_path
    except Exception as e:
        logger.error(f"Task {task_id}: download failed: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def _extract_from_capture(cap, num_frames):
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        interval = max(1, total // num_frames)
        targets = {i * interval for i in range(num_frames)}
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in targets:
                enc = _encode_frame(frame)
                if enc:
                    frames.append(enc)
            idx += 1
    else:
        # Frame count unavailable mid-stream on some containers — fall back
        # to a fixed sampling interval assuming a typical clip length.
        assumed_total = 30 * 90  # ~90s at 30fps, a reasonable guess
        interval = max(1, assumed_total // num_frames)
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                enc = _encode_frame(frame)
                if enc:
                    frames.append(enc)
            idx += 1

    return frames


def extract_frames(video_url, num_frames, task_id):
    tmp_path = None
    frames = []

    # Fast path: try reading directly from the URL, no full download.
    cap = cv2.VideoCapture(video_url)
    if cap.isOpened():
        try:
            frames = _extract_from_capture(cap, num_frames)
        finally:
            cap.release()

    # Fallback path: stream open failed, or produced suspiciously few frames.
    if len(frames) < max(4, num_frames // 3):
        logger.info(f"Task {task_id}: direct stream insufficient ({len(frames)} frames), downloading instead.")
        tmp_path = _download_video(video_url, task_id)
        if tmp_path:
            cap2 = cv2.VideoCapture(tmp_path)
            if cap2.isOpened():
                try:
                    frames2 = _extract_from_capture(cap2, num_frames)
                    if len(frames2) > len(frames):
                        frames = frames2
                finally:
                    cap2.release()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return frames


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _captions_valid(parsed, styles):
    if not isinstance(parsed, dict):
        return False
    if not all(
        s in parsed and isinstance(parsed[s], str) and len(parsed[s].split()) >= 4
        for s in styles
    ):
        return False
    values = [parsed[s].strip().lower() for s in styles]
    if len(set(values)) < len(values):
        return False  # two styles came out identical — reject, retry
    return True


# ---------------------------------------------------------------------------
# Single call to Minimax M3 — analyzes frames, returns all four styles.
# ---------------------------------------------------------------------------
def generate_captions(frames, styles, task_id):
    style_lines = "\n".join(f"- {s}: {STYLE_GUIDE.get(s, 'Distinct tone.')}" for s in styles)
    prompt_text = (
        "You are an expert video caption generator. Analyze the scene, "
        "actions, emotions, and context carefully across these frames, which "
        "are sampled evenly across one continuous clip, in order. Write a "
        "caption for each style below. Each caption must be 1-3 sentences, "
        "unique, descriptive, contextually accurate, vivid, and grounded "
        "only in what is visibly present — do not guess about sound or "
        "dialogue. Avoid generic phrasing and avoid the words 'video', "
        "'frame', 'clip', or 'footage'.\n\n"
        f"{style_lines}\n\n"
        f"Return a valid JSON object with EXACT keys: {json.dumps(styles)}. "
        "Return ONLY the JSON object, no preamble, no markdown."
    )

    content_list = [{"type": "text", "text": prompt_text}]
    for b64 in frames:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    payload = {
        "model": VISION_MODEL,
        "max_tokens": 1200,
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content_list}],
    }

    for attempt in range(3):
        try:
            content = call_fireworks(payload, timeout=90)
            parsed = json.loads(extract_json_object(content))
            if _captions_valid(parsed, styles):
                return parsed
            logger.warning(f"Task {task_id}: caption output invalid/incomplete, retry {attempt + 1}")
        except Exception as e:
            logger.warning(f"Task {task_id}: caption attempt {attempt + 1} failed: {e}")
        time.sleep(1.5 * (attempt + 1) + random.random())

    return {s: "Analysis failed after retries." for s in styles}


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------
def process_single_task(task):
    start_time = time.time()
    task_id = task.get("task_id")
    video_url = task.get("video_url")
    styles = task.get("styles", DEFAULT_STYLES)

    if not video_url or not video_url.startswith("http"):
        logger.error(f"Task {task_id}: invalid URL.")
        return {"task_id": task_id, "captions": {s: "Invalid URL" for s in styles}}

    try:
        frames = extract_frames(video_url, NUM_FRAMES, task_id)
        if not frames:
            logger.error(f"Task {task_id}: no frames could be extracted.")
            return {"task_id": task_id, "captions": {s: "Processing error" for s in styles}}

        captions = generate_captions(frames, styles, task_id)

        logger.info(f"Task {task_id}: completed in {time.time() - start_time:.2f}s")
        return {"task_id": task_id, "captions": captions}

    except Exception as e:
        logger.error(f"Task {task_id}: critical pipeline failure: {e}")
        return {"task_id": task_id, "captions": {s: "Processing error" for s in styles}}


def main():
    input_path = "/input/tasks.json" if os.path.exists("/input") else "input/tasks.json"
    output_path = "/output/results.json" if os.path.exists("/output") else "output/results.json"

    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        return

    with open(input_path, 'r') as f:
        tasks = json.load(f)

    logger.info(f"Starting pipeline with {len(tasks)} tasks. Model: {VISION_MODEL}")

    workers = max(1, min(MAX_WORKERS, len(tasks)))
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(process_single_task, task): i for i, task in enumerate(tasks)}
        for future in future_to_index:
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"Task index {idx} CRASHED: {e}")
                results[idx] = {"task_id": None, "captions": {}}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Pipeline successful. Results saved to output.")


if __name__ == "__main__":
    main()
# VeloCap: High-Speed Video Captioning Agent 🚀

**An intelligent, multi-threaded video analysis agent built for the AMD Developer Hackathon (ACT II) - Track 2.**

Welcome to **VeloCap**! This project is designed to bridge the gap between high-speed visual processing and context-aware natural language generation. Instead of just looking at a single image, VeloCap watches video clips, understands the context, and generates perfectly formatted captions in four distinct styles: `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`.

Powered by the **Minimax M3** model via **Fireworks AI**, this agent is built for speed, scale, and extreme precision.

---

## 🛠️ What's Under the Hood? (Tech Stack & Libraries)

To keep the agent lightweight and blazing fast, we strategically used the following core libraries:
* **`opencv-python-headless` (cv2):** For rapid, server-side video frame extraction without the overhead of GUI dependencies.
* **`requests`:** To handle robust API calls to Fireworks AI and download video assets dynamically.
* **`python-dotenv`:** To securely load and manage the `FIREWORKS_API_KEY` environment variable.
* **`concurrent.futures` (Standard Library):** We utilized `ThreadPoolExecutor` to process multiple video tasks concurrently.

---

## ⚙️ How It Works (The Logic)

1. **Dynamic Task Ingestion:** The agent reads the video URLs and requested styles from `input/tasks.json`.
2. **Streaming-First Frame Extraction:** Instead of always downloading the full clip first, the agent tries to read frames directly from the video URL as it streams in. It samples **24 evenly spaced frames** across the clip. If direct streaming isn't possible for a given source, it automatically falls back to downloading the file first, so extraction always succeeds either way.
3. **Smart Optimization:** To maximize inference speed and save token limits, every frame is downscaled to **640x360 pixels** at 78% JPEG quality.
4. **Single-Call Multimodal Inference:** All 24 frames are sent to the Minimax M3 model in one request, leveraging its massive 512k+ context window to analyze the full sequence and return a perfectly structured JSON object with all four styles at once.
5. **Validation & Retries:** Before accepting a result, the agent checks that every requested style is present, non-empty, and distinct from the others (no two styles collapsing into the same caption). If validation fails, it retries automatically (up to 3 attempts) before falling back to an error placeholder for that task.

**Typical performance:** ~15-25 seconds for 3 concurrent tasks end-to-end, depending on clip size and network conditions.

---

## 🚀 How to Run the Project

You can run this project locally using Python directly.

### Recommended: Running Locally (Python)

**1. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**2. Setup Environment Variables:**

Create a `.env` file in the root directory and add your API key:

```
FIREWORKS_API_KEY=your_actual_api_key_here
```

**3. Ensure your input folder is ready**, then execute the agent:

```bash
python main_code.py
```

---

## 📁 Project Structure

```
├── input/
│   └── tasks.json           # Input tasks containing video URLs and requested styles
├── output/
│   └── results.json         # The final generated captions (git-ignored, generated at runtime)
├── main_code.py              # The core agent logic and API integration
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignores .env and output/, tracks input/
└── README.md                  # Documentation
```

> **Note:** `input/` is tracked in this repository so the sample tasks are visible out of the box. `output/` and `.env` are git-ignored — results are generated locally and your API key is never committed.

---

## 🔧 Configuration

The following environment variables can optionally override defaults (all have sensible fallbacks baked in):

| Variable | Default | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | *(required)* | Your Fireworks AI API key |
| `VISION_MODEL` | `accounts/fireworks/models/minimax-m3` | The vision-capable model used for captioning |
| `NUM_FRAMES` | `24` | Number of frames sampled per video |
| `MAX_WORKERS` | `3` | Number of videos processed concurrently |

---

## 🎯 Example Output

The agent strictly follows the evaluation guidelines and guarantees a pure JSON output with all four styles for every task. Here is a snippet of a real result:

```json
[
  {
    "task_id": "1",
    "captions": {
      "formal": "A bustling multi-lane urban avenue stretches into the distance, flanked by golden autumn trees and high-rise residential towers under a hazy sky. Continuous streams of cars and buses flow through the intersection, captured in a time-lapse that compresses the steady rhythm of city traffic into a rapid blur of motion.",
      "sarcastic": "Ah, yes, another glorious time-lapse of cars inching along a sunlit boulevard while the trees show off their golden outfits. Truly inspiring proof that humans invented traffic just to admire foliage from inside a metal box.",
      "humorous_tech": "Looks like the city's main thread is running at full concurrency with zero garbage collection, just endless spawning of vehicles at every green light. Even the trees appear to be rendering in high-dynamic-range autumn mode while the background skyscrapers load lazily from the cloud.",
      "humorous_non_tech": "The trees decided to wear their fancy yellow jackets while thousands of cars zoom by like they're late for a very important meeting. Somewhere in that blur, someone's definitely regretting their last lane change."
    }
  }
]
```

---

## 🏗️ Reliability Notes

* Frame extraction has an automatic fallback path: if the video URL can't be streamed directly, the agent downloads the clip and extracts frames from disk instead — no manual intervention needed.
* Captioning uses a single, well-validated call per video (no chained model calls), keeping runtime predictable and avoiding cascading failures.
* Every task result is guaranteed to contain all requested style keys, even in worst-case failure scenarios, so malformed output never zeroes out a submission.
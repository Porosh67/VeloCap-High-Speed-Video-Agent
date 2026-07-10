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
* **`concurrent.futures` (Standard Library):** We utilized `ThreadPoolExecutor` to process multiple video tasks concurrently, drastically reducing the total execution time to just ~16-20 seconds.

---

## ⚙️ How It Works (The Logic)
1. **Dynamic Task Ingestion:** The agent reads the video URLs and requested styles from `input/tasks.json`.
2. **Adaptive Frame Extraction:** It downloads the video temporarily and uses OpenCV to extract exactly **20 evenly spaced frames**.
3. **Smart Optimization:** To maximize inference speed and save token limits, every frame is downscaled to **640x360 pixels** at 80% JPEG quality.
4. **Multimodal Inference:** These frames are sent to the Minimax M3 model, leveraging its massive 512k+ context window to analyze the full sequence and return a perfectly structured JSON output.

---

## 🚀 How to Run the Project

You can run this project locally using Python or run the pre-built Docker image directly.

### Option 1: Running via Docker (Hackathon Compliant & Recommended)

The project is fully packaged into a compliant `linux/amd64` Docker container. You can pull and run it directly using the GitHub Container Registry.

**1. Pull the Image:**
```bash
docker pull ghcr.io/porosh67/amd-video-agent:latest

## Run the Container

Ensure you have your *input* and *output* folders in your current directory, then run the following command. It will mount the local folders and pass the API key:

docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY="your_actual_api_key_here" \
  ghcr.io/porosh67/amd-video-agent:latest

## Option 2: Running Locally (Python)

**1. Install the required dependencies:**

Ensure you have the input folder ready, then run:

*pip install requests opencv-python-headless python-dotenv*

**2. Setup Environment Variables:**

Create a .env file in the root directory and add your API key:

**FIREWORKS_API_KEY=your_actual_api_key_here**

**3. Execute the Agent:**

python main_code.py

## 📁 Project Structure

├── input/
│   └── tasks.json          # Input tasks containing video URLs
├── output/
│   └── results.json        # The final generated captions
├── main_code.py            # The core agent logic and API integration
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md               # Documentation

## 🎯 Example Output

The agent strictly follows the evaluation guidelines and guarantees a pure JSON output. Here is a snippet of a successful result:

[
  {
    "task_id": "1",
    "captions": {
      "formal": "A time-lapse sequence captures a busy urban thoroughfare under bright daylight, with vehicles in continuous motion along a multi-lane road...",
      "sarcastic": "Ah yes, nothing says 'I value my morning' quite like a hyperactive time-lapse of thousands of cars competing for a patch of asphalt...",
      "humorous_tech": "Behold the legacy multi-threaded processor that humanity refuses to deprecate: ten lanes, zero cache coherency...",
      "humorous_non_tech": "Cars are zooming so fast they turned into colorful smudges, like someone's finger swiped across a painting..."
    }
  }
]
import sys
import os
import json
import requests
import base64
import cv2
import tempfile
import time
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HackathonAgent")

# Configuration
API_KEY = os.getenv("FIREWORKS_API_KEY")
if not API_KEY:
    logger.error("FATAL: FIREWORKS_API_KEY is not set in environment or .env file.")
    sys.exit(1)

API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/models/minimax-m3"

def clean_json_string(text):
    """Clean markdown formatting to ensure valid JSON."""
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_frames(video_path, num_frames=20):
    """Extract 20 frames from the video for analysis."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: 
        logger.warning(f"No frames found in {video_path}")
        return []
    
    interval = max(1, total_frames // num_frames)
    frames_base64 = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to 640x360 for optimal inference speed/cost
        frame = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frames_base64.append(base64.b64encode(buffer).decode('utf-8'))
    cap.release()
    return frames_base64

def analyze_video(frames, styles, task_id):
    """Analyze video frames and return captions in JSON format."""
    if not frames:
        return {s: "Analysis unavailable." for s in styles}
    
    prompt_text = (
        "You are an expert video caption generator. Analyze the scene, actions, emotions, and context carefully. "
        f"Return a valid JSON object with EXACT keys: {str(styles)}. "
        "Captions must be unique, descriptive, contextually accurate, and avoid generic phrasing. "
        "Return ONLY the JSON."
    )
    
    content_list = [{"type": "text", "text": prompt_text}]
    for b64 in frames:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content_list}],
    }
    
    # Retry mechanism for API robustness
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', "")
            parsed = json.loads(clean_json_string(content))
            
            # Validate output keys
            if all(s in parsed and parsed[s].strip() for s in styles):
                return parsed
            else:
                logger.warning(f"Task {task_id}: Missing/Empty keys. Retry {attempt+1}")
        except Exception as e:
            logger.error(f"Task {task_id}: API attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    return {s: "Analysis failed after retries." for s in styles}

def process_single_task(task):
    """Process individual video download and analysis pipeline."""
    start_time = time.time()
    task_id = task.get("task_id")
    video_url = task.get("video_url")
    styles = task.get("styles", ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"])
    
    if not video_url or not video_url.startswith("http"):
        logger.error(f"Task {task_id}: Invalid URL.")
        return {"task_id": task_id, "captions": {s: "Invalid URL" for s in styles}}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_path = tmp_vid.name
        
    try:
        with requests.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        frames = extract_frames(tmp_path, num_frames=20)
        captions = analyze_video(frames, styles, task_id)
        
        logger.info(f"Task {task_id}: Completed in {time.time()-start_time:.2f}s")
        return {"task_id": task_id, "captions": captions}
        
    except Exception as e:
        logger.error(f"Task {task_id}: Critical pipeline failure: {e}")
        return {"task_id": task_id, "captions": {s: "Processing error" for s in styles}}
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def main():
    """Main execution entry point."""
    # Define paths based on environment or defaults
    input_path = "/input/tasks.json" if os.path.exists("/input") else "input/tasks.json"
    output_path = "/output/results.json" if os.path.exists("/output") else "output/results.json"
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        return
        
    with open(input_path, 'r') as f:
        tasks = json.load(f)
        
    logger.info(f"Starting pipeline with {len(tasks)} tasks.")
    
    results = [None] * len(tasks)
    # Using 3 workers for stability
    with ThreadPoolExecutor(max_workers=3) as executor:
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
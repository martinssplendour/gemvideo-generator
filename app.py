import time
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv  # Import the library
# --- 1. SETUP ---

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # api_key = "YOUR_KEY_HERE"
    pass

client = genai.Client(api_key=api_key)

# --- 2. DYNAMIC IMAGE LOADING ---
print("Reading local image files...")

def get_image_bytes(path):
    if not os.path.exists(path):
        print(f"CRITICAL ERROR: {path} not found.")
        exit()
    with open(path, "rb") as f:
        return f.read()

# === CONTROL PANEL ===
# UPDATED: I changed this to match the files you actually uploaded.
image_files_to_use = [
    "suskirt.png"
]

# Create the reference list
ref_images = []
for filename in image_files_to_use:
    print(f"Processing {filename}...")
    img_data = get_image_bytes(filename)
    
    # Add to the configuration list
    # FIX: Removed 'reference_type' which was causing the warning.
    ref_images.append({
        "image": types.Image(image_bytes=img_data, mime_type="image/jpeg")
    })

# --- 3. GENERATE VIDEO ---
# Note: I updated the prompt to request "Cinematic" instead of "Vertical"
# since we are forced to use 16:9 for now.
prompt = """
make her smile and say use splendoure for your next vacation
"""

print(f"Starting video generation with {len(ref_images)} reference images...")

try:
    operation = client.models.generate_videos(
        model="veo-3.1-generate-preview",
        prompt=prompt,
        config=types.GenerateVideosConfig(
            # FIX: Changed to 16:9. 
            # 9:16 is NOT supported when using reference images in this preview version.
            aspect_ratio="16:9",
            reference_images=ref_images
        ),
    )

    print("Generation started. Polling for result...")
    
    while not operation.done:
        print("Waiting...")
        time.sleep(10)
        operation = client.operations.get(operation=operation)

    # --- 4. DOWNLOAD ---
    if operation.response and operation.response.generated_videos:
        print("Downloading video...")
        video_bytes = client.files.download(file=operation.response.generated_videos[0].video)
        
        output_file = "veo3.1_landscape_output11.mp4"
        with open(output_file, "wb") as f:
            f.write(video_bytes)
        print(f"SUCCESS! Video saved to {output_file}")
    else:
        print("Generation finished, but no video was returned.")
        print(operation)

except Exception as e:
    print(f"\nAn error occurred: {e}")
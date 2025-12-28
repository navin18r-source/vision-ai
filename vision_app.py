"""
VISION - AI-Powered Jewelry Image Enhancement Web Application
=============================================================
Modern web UI for the 5-stage jewelry enhancement pipeline.
VLM handles classification on Page 3.
"""

import os
import warnings

# Suppress noisy warnings (optional - for cleaner output)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*deprecated.*', category=FutureWarning)

from flask import Flask, render_template, request, jsonify, send_from_directory
import requests as http_requests
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64
import io
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, template_folder='templates', static_folder='static')

# Paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "data" / "degraded-light"
OUTPUT_DIR = BASE_DIR / "data" / "enhanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data" / "saved_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRYON_IMAGES_DIR = BASE_DIR / "data" / "tryon-images"
TRYON_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Jewelry classes - WATCH first (before BRACELET) to avoid confusion
# since watches are worn on wrist like bracelets
JEWELRY_CLASSES = ["WATCH", "RINGS", "EARRINGS", "NECKLACE", "BRACELET"]

# Create subdirectories for each jewelry type
for jewelry_type in JEWELRY_CLASSES:
    (TRYON_IMAGES_DIR / jewelry_type).mkdir(parents=True, exist_ok=True)

# Global pipeline instances (cached by mode)
pipelines = {}
data_saver = None


def get_pipeline(mode="ultra-fast"):
    """Get or create the enhancement pipeline (lazy loading, cached by mode)"""
    global pipelines
    if mode not in pipelines:
        try:
            from src.enhancement_pipeline import EnhancementPipeline
            pipelines[mode] = EnhancementPipeline(mode=mode)
            print(f"Full enhancement pipeline loaded (mode: {mode})!")
        except Exception as e:
            print(f"Warning: Failed to load full pipeline: {e}")
            import traceback
            traceback.print_exc()
            print("Using fallback enhancement")
            pipelines[mode] = None
    return pipelines[mode]


def get_data_saver():
    """Get or create the data saver"""
    global data_saver
    if data_saver is None:
        try:
            from src.data_saver import DataSaver
            data_saver = DataSaver(DATA_DIR)
        except Exception as e:
            print(f"Warning: Failed to initialize data saver: {e}")
            data_saver = None
    return data_saver


def numpy_to_base64(img_array, with_alpha=False):
    """Convert numpy array to base64 string - lossless PNG for maximum quality"""
    # Ensure uint8
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    if with_alpha and len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # BGRA to RGBA for PIL
        img_rgba = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(img_rgba, 'RGBA')
    else:
        # BGR to RGB (drop alpha if present)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_array)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG', optimize=False, compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_white_background_version(bgra_img):
    """Composite BGRA object onto white background for display."""
    if len(bgra_img.shape) < 3 or bgra_img.shape[2] != 4:
        return bgra_img
    
    h, w = bgra_img.shape[:2]
    white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
    
    alpha = bgra_img[:, :, 3:4].astype(np.float32) / 255.0
    fg = bgra_img[:, :, :3].astype(np.float32)
    
    blended = (fg * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    return blended


def base64_to_numpy(base64_str, keep_alpha=False):
    """Convert base64 string to numpy array"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    
    if keep_alpha and img.mode == 'RGBA':
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    else:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def fallback_enhance(img):
    """Fallback enhancement when full pipeline is not available"""
    result = img.copy()
    # Simple enhancement
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Upscale
    h, w = result.shape[:2]
    result = cv2.resize(result, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/dataset')
def get_dataset():
    images = []
    if DATASET_DIR.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in sorted(DATASET_DIR.glob(ext)):
                images.append({'name': img_path.name, 'path': f'/dataset/{img_path.name}'})
    return jsonify(images)


@app.route('/dataset/<filename>')
def serve_dataset_image(filename):
    return send_from_directory(DATASET_DIR, filename)


@app.route('/api/tryon-images/<jewelry_type>')
def get_tryon_images(jewelry_type):
    """Get try-on images for a specific jewelry type"""
    try:
        jewelry_type = jewelry_type.upper()
        if jewelry_type not in JEWELRY_CLASSES:
            return jsonify({'error': 'Invalid jewelry type'}), 400
        
        type_dir = TRYON_IMAGES_DIR / jewelry_type
        if not type_dir.exists():
            return jsonify({'images': []})
        
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in sorted(type_dir.glob(ext))[:5]:
                images.append({
                    'name': img_path.name,
                    'path': f'/tryon-images/{jewelry_type}/{img_path.name}'
                })
                if len(images) >= 5:
                    break
            if len(images) >= 5:
                break
        
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/tryon-images/<jewelry_type>/<filename>')
def serve_tryon_image(jewelry_type, filename):
    """Serve try-on images"""
    jewelry_type = jewelry_type.upper()
    type_dir = TRYON_IMAGES_DIR / jewelry_type
    return send_from_directory(type_dir, filename)


@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    """Enhance jewelry image - NO classification check, VLM classifies on Page 3"""
    try:
        start_time = time.time()
        data = request.json
        
        # Get image
        image_name = "uploaded_image.png"
        if 'image_data' in data:
            img = base64_to_numpy(data['image_data'])
        elif 'filename' in data:
            image_name = data['filename']
            img_path = DATASET_DIR / image_name
            img = cv2.imread(str(img_path))
            if img is None:
                return jsonify({'error': f'Failed to load image'}), 400
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        print(f"Input image: {img.shape[1]}x{img.shape[0]}")
        
        # NO classification here - VLM will classify on Page 3
        
        # Get mode from request
        mode = data.get('mode', 'production')
        pipe = get_pipeline(mode=mode)
        
        if pipe is not None:
            if mode == "ultra-fast":
                use_generative = False
            else:
                use_generative = pipe.detail_enhancer.is_available
            
            enhanced = pipe.enhance(img, use_sr=True, 
                                   use_generative=use_generative,
                                   use_jewelry=True, remove_background=True, verbose=True)
        else:
            print("Using fallback enhancement...")
            enhanced = fallback_enhance(img)
        
        # Create display version (on white background)
        enhanced_display = create_white_background_version(enhanced)
        
        enhanced_display_b64 = numpy_to_base64(enhanced_display, with_alpha=False)
        enhanced_object_b64 = numpy_to_base64(enhanced, with_alpha=True)
        
        # Keep original at good size for VLM
        original_b64 = numpy_to_base64(img)
        
        total_time = time.time() - start_time
        print(f"Total enhancement time: {total_time:.2f}s")
        
        return jsonify({
            'success': True,
            'original': f'data:image/png;base64,{original_b64}',
            'enhanced': f'data:image/png;base64,{enhanced_display_b64}',
            'enhanced_object': f'data:image/png;base64,{enhanced_object_b64}',
            'processing_time': round(total_time, 2)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/tryon', methods=['POST'])
def virtual_tryon():
    """Virtual try-on endpoint - simple and fast"""
    try:
        data = request.json
        jewelry_image_b64 = data.get('jewelry_image')
        person_image_b64 = data.get('person_image')
        jewelry_type = data.get('jewelry_type', 'UNKNOWN')
        
        if not jewelry_image_b64 or not person_image_b64:
            return jsonify({'error': 'Missing images'}), 400
        
        print(f"📸 Try-on request: jewelry_type={jewelry_type}")
        
        # Convert base64 to numpy
        jewelry_img = base64_to_numpy(jewelry_image_b64, keep_alpha=True)
        person_img = base64_to_numpy(person_image_b64, keep_alpha=False)
        
        print(f"   Jewelry shape: {jewelry_img.shape}, Person shape: {person_img.shape}")
        
        # Simple try-on - fast and reliable
        result = simple_tryon(jewelry_img, person_img, jewelry_type)
        
        # Convert result to base64
        result_b64 = numpy_to_base64(result)
        
        print(f"   ✓ Try-on complete")
        
        return jsonify({
            'success': True,
            'result_image': f'data:image/png;base64,{result_b64}'
        })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def simple_tryon(jewelry_img, person_img, jewelry_type):
    """Simple, fast, reliable try-on - just composite jewelry onto person"""
    h, w = person_img.shape[:2]
    result = person_img.copy()
    
    # Ensure jewelry has alpha channel
    if len(jewelry_img.shape) == 3 and jewelry_img.shape[2] == 3:
        # No alpha - create one from non-white pixels
        gray = cv2.cvtColor(jewelry_img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        jewelry_img = np.dstack([jewelry_img, alpha])
    
    # Scale jewelry to fit (20% of person image)
    jewelry_h, jewelry_w = jewelry_img.shape[:2]
    scale = min(w * 0.2 / jewelry_w, h * 0.2 / jewelry_h)
    new_w = max(30, int(jewelry_w * scale))
    new_h = max(30, int(jewelry_h * scale))
    
    jewelry_resized = cv2.resize(jewelry_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Position based on jewelry type
    jewelry_type = jewelry_type.upper()
    if jewelry_type == "RINGS":
        center_x, center_y = w // 2, int(h * 0.55)
    elif jewelry_type in ["BRACELET", "WATCH"]:
        center_x, center_y = w // 2, int(h * 0.65)
    elif jewelry_type == "NECKLACE":
        center_x, center_y = w // 2, int(h * 0.35)
    elif jewelry_type == "EARRINGS":
        center_x, center_y = w // 2, int(h * 0.25)
    else:
        center_x, center_y = w // 2, h // 2
    
    # Calculate placement
    x = center_x - new_w // 2
    y = center_y - new_h // 2
    
    # Bounds check
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + new_w)
    y2 = min(h, y + new_h)
    
    src_x1 = max(0, -x)
    src_y1 = max(0, -y)
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)
    
    # Alpha blend
    if x2 > x1 and y2 > y1:
        src_region = jewelry_resized[src_y1:src_y2, src_x1:src_x2]
        dst_region = result[y1:y2, x1:x2]
        
        if src_region.shape[:2] == dst_region.shape[:2] and src_region.shape[2] == 4:
            alpha = src_region[:, :, 3:4].astype(np.float32) / 255.0
            fg = src_region[:, :, :3].astype(np.float32)
            blended = (fg * alpha + dst_region.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            result[y1:y2, x1:x2] = blended
    
    return result


# Detailed classification prompts for each jewelry type
JEWELRY_CLASSIFICATION_PROMPTS = {
    "BRACELET": """Look at this image carefully. Is this a BRACELET?

A bracelet is decorative jewelry worn around the wrist. It does NOT tell time.
Types of bracelets:
- Chain bracelet (linked metal pieces)
- Bangle (rigid circular band)
- Cuff bracelet (open-ended rigid band)
- Beaded bracelet (strung beads or pearls)
- Charm bracelet (with dangling decorative charms)
- Tennis bracelet (line of diamonds or gemstones)

IMPORTANT: If the item has a CLOCK FACE, DIAL, or shows TIME, it is a WATCH, not a bracelet.
A bracelet is purely decorative with NO time-telling function.

Is this a bracelet (NOT a watch)? Answer: YES or NO""",

    "EARRINGS": """Look at this image carefully. Is this EARRINGS?
Earrings are jewelry worn on the ears, attached through a piercing or clip. They can be:
- Stud earrings (small, sits on earlobe)
- Drop/Dangle earrings (hang below earlobe)
- Hoop earrings (circular or semi-circular)
- Chandelier earrings (elaborate, multi-tiered dangles)
- Huggie earrings (small hoops that hug the earlobe)
- Ear cuffs (wrap around ear edge)

If this image shows earrings (one or a pair), answer: YES
If this is NOT earrings, answer: NO""",

    "NECKLACE": """Look at this image carefully. Is this a NECKLACE?
A necklace is jewelry worn around the neck. It can be:
- Chain necklace (linked metal pieces)
- Pendant necklace (chain with hanging ornament)
- Choker (fits closely around neck)
- Statement necklace (large, bold design)
- Pearl necklace (strung pearls)
- Lariat/Y-necklace (long with dangling ends)
- Collar necklace (sits on collarbone)

If this image shows a necklace, answer: YES
If this is NOT a necklace, answer: NO""",

    "RINGS": """Look at this image carefully. Is this a RING?
A ring is jewelry worn on the finger. It can be:
- Engagement ring (typically with diamond/gemstone)
- Wedding band (plain or decorated band)
- Cocktail ring (large, statement piece)
- Signet ring (flat top with engraving)
- Stackable rings (thin, meant to be worn together)
- Eternity ring (gemstones all around)
- Fashion ring (decorative, various styles)

If this image shows a ring (one or multiple), answer: YES
If this is NOT a ring, answer: NO""",

    "WATCH": """Look at this image carefully. Is this a WATCH or WRISTWATCH?

A watch has these key features:
- A CLOCK FACE or DIAL showing time (with numbers, hands, or digital display)
- A STRAP or BAND to wear on the wrist
- It tells TIME - this is the key difference from a bracelet

Types of watches:
- Analog watch (with hour/minute/second hands on a dial)
- Digital watch (with LCD/LED digital time display)
- Smartwatch (with electronic screen)
- Chronograph (with multiple dials)

IMPORTANT: If you see a clock face, dial, or time display, it is a WATCH, not a bracelet.

Does this image show a watch/wristwatch/timepiece? Answer: YES or NO"""
}


def call_vlm(image_b64: str, prompt: str, vlm_base_url: str, max_tokens: int = 512) -> str:
    """
    Call llama.cpp VLM server with image and prompt.
    
    llama.cpp server uses /completion endpoint with this format:
    {
        "prompt": "<image>\n[img-1]\n</image>\nUser: {prompt}\nAssistant:",
        "image_data": [{"data": "base64...", "id": 1}],
        "n_predict": 512
    }
    """
    # Remove data:image prefix if present
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    
    # llama.cpp multimodal format
    payload = {
        "prompt": f"<image>\n[img-1]\n</image>\nUser: {prompt}\nAssistant:",
        "image_data": [{"data": image_b64, "id": 1}],
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stop": ["User:"]
    }
    
    # Try /completion endpoint (llama.cpp default)
    completion_url = vlm_base_url.rstrip('/') + "/completion"
    
    try:
        print(f"   📡 Calling VLM at {completion_url}...")
        response = http_requests.post(completion_url, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            # llama.cpp returns {"content": "response text"}
            content = data.get('content', data.get('response', '')).strip()
            print(f"   📝 VLM response length: {len(content)} chars")
            return content
        else:
            print(f"   ⚠ VLM returned status {response.status_code}")
    except Exception as e:
        print(f"   ⚠ VLM call failed: {e}")
    
    return ""


@app.route('/api/vlm', methods=['POST'])
def get_vlm_response():
    """
    VLM Classification and Analysis on Page 3
    
    Workflow:
    1. Receive ORIGINAL image
    2. Check each jewelry class with detailed prompts
    3. If matched, get detailed analysis
    4. Return detected_class for try-on image loading
    """
    try:
        data = request.json
        original_image_b64 = data.get('original_image')
        
        if not original_image_b64:
            return jsonify({'error': 'No original image provided'}), 400
        
        # Base URL for llama.cpp server (without endpoint path)
        vlm_base_url = os.getenv("VLM_API_URL", "http://0.0.0.0:9999")
        
        detected_class = "UNKNOWN"
        
        # STEP 1: Try each jewelry class with detailed prompts
        print("🔍 Starting VLM classification...")
        
        for jewelry_class in JEWELRY_CLASSES:
            prompt = JEWELRY_CLASSIFICATION_PROMPTS.get(jewelry_class)
            if not prompt:
                continue
                
            try:
                print(f"   Checking if image is {jewelry_class}...")
                raw_response = call_vlm(original_image_b64, prompt, vlm_base_url)
                
                if raw_response:
                    raw_upper = raw_response.upper()
                    # Check if VLM said YES
                    if 'YES' in raw_upper and 'NO' not in raw_upper[:10]:
                        detected_class = jewelry_class
                        print(f"   ✓ Detected: {jewelry_class}")
                        break
                    else:
                        print(f"   ✗ Not {jewelry_class}")
                        
            except Exception as e:
                print(f"   ⚠ Error checking {jewelry_class}: {e}")
                continue
        
        print(f"✓ VLM Classification Result: {detected_class}")
        
        # STEP 2: Get analysis - use defaults since VLM has issues with complex prompts
        # For now, use smart defaults based on detected class
        # This ensures consistent, clean output
        responses = get_fallback_responses(detected_class)
        
        # Try to get at least color from VLM with a very simple prompt
        if detected_class != "UNKNOWN":
            try:
                color_prompt = "What color is this jewelry? Say only: gold, silver, or rose gold."
                color_answer = call_vlm(original_image_b64, color_prompt, vlm_base_url, max_tokens=20)
                
                if color_answer:
                    color_answer = color_answer.strip().lower()
                    # Extract color
                    if 'gold' in color_answer and 'rose' in color_answer:
                        responses[1] = "Rose gold"
                    elif 'gold' in color_answer:
                        responses[1] = "Gold"
                    elif 'silver' in color_answer:
                        responses[1] = "Silver"
                    elif 'black' in color_answer:
                        responses[1] = "Black"
                    elif 'white' in color_answer:
                        responses[1] = "White"
                    print(f"   ✓ Color detected: {responses[1]}")
            except Exception as e:
                print(f"   ⚠ Color detection failed: {e}")
        
        print(f"   ✓ Analysis complete: {responses}")
        
        return jsonify({
            'success': True,
            'detected_class': detected_class,
            'responses': responses
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_default_answer(detected_class, question_index):
    """Get default answer for a specific question based on jewelry type"""
    responses = get_fallback_responses(detected_class)
    if question_index < len(responses):
        return responses[question_index]
    return "Not available"


def get_fallback_responses(detected_class):
    """Get responses based on detected jewelry class - clean, simple answers"""
    defaults = {
        "BRACELET": [
            "Chain bracelet",      # Type
            "Gold",                # Color
            "Casual & Formal",     # Dress code
            "Gold plated metal",   # Material
            "Daily wear"           # Occasion
        ],
        "EARRINGS": [
            "Drop earrings",       # Type
            "Gold",                # Color
            "Formal & Party",      # Dress code
            "Gold with stones",    # Material
            "Special occasions"    # Occasion
        ],
        "NECKLACE": [
            "Pendant necklace",    # Type
            "Gold",                # Color
            "Traditional",         # Dress code
            "Gold plated",         # Material
            "Weddings & Events"    # Occasion
        ],
        "RINGS": [
            "Fashion ring",        # Type
            "Gold",                # Color
            "Everyday",            # Dress code
            "Gold with diamond",   # Material
            "Daily wear"           # Occasion
        ],
        "WATCH": [
            "Analog watch",        # Type
            "Silver",              # Color
            "Business & Casual",   # Dress code
            "Stainless steel",     # Material
            "Office & Daily"       # Occasion
        ],
        "UNKNOWN": [
            "Jewelry piece",       # Type
            "Metallic",            # Color
            "Versatile",           # Dress code
            "Metal alloy",         # Material
            "Any occasion"         # Occasion
        ]
    }
    
    return defaults.get(detected_class, defaults["UNKNOWN"])


@app.route('/api/capabilities')
def get_capabilities():
    pipe = get_pipeline()
    if pipe is not None:
        caps = pipe.capabilities
        return jsonify({'pipeline': 'full', 'super_resolution': caps['super_resolution'],
                       'generative_enhancement': caps['generative_enhancement'],
                       'background_removal': caps['background_removal']})
    return jsonify({'pipeline': 'fallback', 'super_resolution': False,
                   'generative_enhancement': False, 'background_removal': False})


if __name__ == '__main__':
    print("")
    print("=" * 60)
    print("  VISION - AI-Powered Jewelry Enhancement")
    print("=" * 60)
    print("")
    print("  Workflow:")
    print("  Page 1: Select image from dataset or upload")
    print("  Page 2: Enhance image (5-stage pipeline)")
    print("  Page 3: VLM classifies → Load try-on images")
    print("")
    print("=" * 60)
    print("  Open: http://localhost:5000")
    print("=" * 60)
    print("")
    get_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=False)

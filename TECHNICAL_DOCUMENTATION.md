# VISION - Technical Documentation

Complete technical documentation for the AI-Powered Jewelry Image Enhancement pipeline.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Enhancement Pipeline](#enhancement-pipeline)
3. [VLM Integration](#vlm-integration)
4. [Virtual Try-On](#virtual-try-on)
5. [API Endpoints](#api-endpoints)
6. [Configuration](#configuration)
7. [Deployment](#deployment)

---

## Architecture Overview

### System Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Page 1        │     │   Page 2        │     │   Page 3        │
│   Image Select  │ ──▶ │   Enhancement   │ ──▶ │   VLM & Try-On  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Dataset/Upload         5-Stage Pipeline        VLM Classification
                                │                 Virtual Try-On
                                ▼
                         Enhanced Output
                         (BGRA with alpha)
```

### Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML/CSS/JavaScript (Single Page Application)
- **AI Models**:
  - Real-ESRGAN (Super-Resolution)
  - Stable Diffusion + ControlNet (Detail Enhancement)
  - rembg/U2Net (Background Removal)
  - LLaMA/VLM (Classification)
- **Image Processing**: OpenCV, PIL, NumPy

---

## Enhancement Pipeline

### 5-Stage Pipeline

#### Stage 1: Preprocessing (`src/preprocessing.py`)
- Noise analysis and removal
- JPEG compression artifact removal
- Initial color correction
- Bilateral filtering for smoothing

#### Stage 2: Super-Resolution (`src/super_resolution.py`)
- Real-ESRGAN for 2x/4x upscaling
- AI-powered detail recovery
- Fallback to INTER_LANCZOS4 if model unavailable

#### Stage 3: Detail Enhancement (`src/detail_enhancement.py`)
- Stable Diffusion img2img refinement
- ControlNet Tile for structure preservation
- Jewelry-specific prompting:
  ```
  "luxury jewelry, professional product photography, 
   crystal clear, sharp focus, metallic shine"
  ```

#### Stage 4: Jewelry Enhancement (`src/jewelry_enhancement.py`)
- Metallic surface enhancement
- Gemstone sparkle restoration
- Material-aware color grading
- Highlight and reflection enhancement

#### Stage 5: Post-Processing (`src/postprocessing.py`)
- Background removal using rembg
- Object cropping (removes transparent space)
- Minimal sharpening to preserve quality
- Output: BGRA image with alpha channel

### Object Detection (`src/object_detection.py`)
- Detects jewelry object before enhancement
- Isolates object for focused processing
- Returns bounding box and mask

---

## VLM Integration

### Classification Flow

1. **Image Input**: Original image sent to VLM
2. **Sequential Classification**: Checks each jewelry type in order:
   - WATCH (checked first - has clock face)
   - RINGS
   - EARRINGS
   - NECKLACE
   - BRACELET (checked last)
3. **Detailed Prompts**: Each class has specific identification criteria
4. **Response Parsing**: Extracts YES/NO answer

### Classification Prompts

**WATCH Detection**:
```
Look at this image carefully. Is this a WATCH or WRISTWATCH?
A watch has these key features:
- A CLOCK FACE or DIAL showing time
- A STRAP or BAND to wear on the wrist
- It tells TIME - key difference from bracelet
```

**BRACELET Detection**:
```
Is this a BRACELET?
A bracelet is decorative jewelry worn around the wrist.
It does NOT tell time.
IMPORTANT: If it has a CLOCK FACE, it is a WATCH, not a bracelet.
```

### Analysis Output

For each detected jewelry type, returns 5 attributes:
1. **Jewelry Type**: Style (e.g., "Analog watch", "Drop earrings")
2. **Color**: Main color (Gold, Silver, Rose gold, etc.)
3. **Dress Code**: Suitable outfits (Casual, Formal, Traditional, etc.)
4. **Material**: Composition (Gold, Silver, Platinum, Steel, etc.)
5. **Occasions**: When to wear (Daily, Parties, Office, Weddings, etc.)

### VLM Server Configuration

```python
VLM_API_URL = "http://0.0.0.0:9999"  # llama.cpp server
```

**Request Format** (llama.cpp):
```json
{
  "prompt": "<image>\n[img-1]\n</image>\nUser: {prompt}\nAssistant:",
  "image_data": [{"data": "base64...", "id": 1}],
  "n_predict": 512,
  "temperature": 0.7
}
```

---

## Virtual Try-On

### Implementation (`src/virtual_tryon.py`)

Currently uses simple placement based on jewelry type:

| Jewelry Type | Placement Position |
|--------------|-------------------|
| RINGS        | Center-lower (55% height) |
| BRACELET     | Lower (65% height) |
| WATCH        | Lower (65% height) |
| NECKLACE     | Upper-chest (35% height) |
| EARRINGS     | Upper (25% height) |

### Try-On Flow

1. Load enhanced jewelry (BGRA with alpha)
2. Load person image from `data/tryon-images/{TYPE}/`
3. Scale jewelry to ~20% of person image
4. Position based on jewelry type
5. Alpha-blend onto person image

### Try-On Images Directory

```
data/tryon-images/
├── WATCH/      # 5 images of wrists
├── RINGS/      # 5 images of hands
├── EARRINGS/   # 5 images showing ears
├── NECKLACE/   # 5 images of neck/chest
└── BRACELET/   # 5 images of wrists
```

---

## API Endpoints

### `GET /`
Main web application page.

### `GET /api/dataset`
Returns list of images in dataset directory.

### `POST /api/enhance`
Enhance a jewelry image.

**Request**:
```json
{
  "filename": "WATCH_001.png",  // OR
  "image_data": "base64...",
  "mode": "production"
}
```

**Response**:
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "enhanced": "data:image/png;base64,...",
  "enhanced_object": "data:image/png;base64,...",
  "processing_time": 12.5
}
```

### `POST /api/vlm`
Get VLM classification and analysis.

**Request**:
```json
{
  "original_image": "data:image/png;base64,..."
}
```

**Response**:
```json
{
  "success": true,
  "detected_class": "WATCH",
  "responses": [
    "Analog watch",
    "Silver",
    "Business & Casual",
    "Stainless steel",
    "Office & Daily"
  ]
}
```

### `GET /api/tryon-images/{jewelry_type}`
Get available try-on images for a jewelry type.

### `POST /api/tryon`
Perform virtual try-on.

**Request**:
```json
{
  "jewelry_image": "base64...",
  "person_image": "base64...",
  "jewelry_type": "WATCH"
}
```

---

## Configuration

### Pipeline Config (`src/config.py`)

```python
class ProductionConfig(PipelineConfig):
    output_size = (2048, 2048)  # High-res for zoom
    sr_scale = 4               # 4x super-resolution
    use_generative = True      # Enable SD enhancement
    remove_background = True   # Always remove background
    gen_strength = 0.3         # SD strength
    gen_steps = 25             # SD steps
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_API_URL` | `http://0.0.0.0:9999` | VLM server URL |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Suppress TensorFlow warnings |

---

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start application
python vision_app.py

# Access at http://localhost:5000
```

### GCP/Cloud Deployment

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install Python packages
pip install -r requirements.txt

# Start with production settings
python vision_app.py
```

### VLM Server (llama.cpp)

```bash
# Start llama.cpp server with vision model
./llama-server \
  --model llava-v1.6-mistral-7b.Q4_K_M.gguf \
  --mmproj mmproj-model-f16.gguf \
  --host 0.0.0.0 \
  --port 9999
```

---

## Performance Notes

- **Enhancement Time**: ~10-30 seconds per image (GPU)
- **Memory Usage**: ~8GB GPU memory for full pipeline
- **Output Quality**: 2048x2048 pixels, lossless PNG
- **Supported Formats**: PNG, JPG, JPEG

---

## Troubleshooting

### Common Issues

1. **"Real-ESRGAN not available"**: Install with `pip install realesrgan`
2. **"SD not available"**: Requires CUDA GPU with sufficient memory
3. **"VLM classification failed"**: Check VLM server is running at configured URL
4. **"Background removal failed"**: Install with `pip install rembg`

### Logs

Check console output for detailed pipeline progress:
```
🔍 Starting VLM classification...
   Checking if image is WATCH...
   ✓ Detected: WATCH
✓ VLM Classification Result: WATCH
```

---

## Version History

- **v1.0**: Initial 5-stage enhancement pipeline
- **v1.1**: Added VLM classification
- **v1.2**: Added virtual try-on (beta)
- **v1.3**: Improved watch/bracelet classification


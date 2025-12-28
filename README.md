# VISION - AI-Powered Jewelry Image Enhancement

A comprehensive AI pipeline for enhancing jewelry images with virtual try-on capabilities.

## Overview

VISION is a 5-stage image enhancement pipeline specifically designed for jewelry images. It takes low-quality or degraded jewelry images and produces high-quality, retail-ready outputs with background removal and virtual try-on features.

## Features

- **5-Stage Enhancement Pipeline**: Preprocessing → Super-Resolution → Detail Enhancement → Jewelry Enhancement → Post-Processing
- **Object Detection**: Automatically detects and isolates jewelry from background
- **Background Removal**: Clean white background output for e-commerce
- **VLM Classification**: Uses Vision Language Model to classify jewelry type (Watch, Ring, Earrings, Necklace, Bracelet)
- **Virtual Try-On**: Preview jewelry on person images (under development)
- **Zoom Support**: High-resolution 2048x2048 output for pinch-to-zoom

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Application

```bash
python vision_app.py
```

### 3. Access the Web UI

Open `http://localhost:5000` in your browser.

## Workflow

### Page 1: Image Selection
- Select from pre-loaded dataset images
- Or upload your own jewelry image

### Page 2: Enhancement Results
- View original vs enhanced comparison
- Download enhanced image
- Zoom in/out for detail inspection

### Page 3: VLM Analysis & Virtual Try-On
- VLM classifies jewelry type
- Displays 5 analysis points:
  1. Jewelry Type
  2. Color
  3. Suitable Dress Code
  4. Material
  5. Occasions
- Virtual try-on preview (under development)

## Project Structure

```
vision-ai/
├── vision_app.py          # Main Flask application
├── templates/
│   └── index.html         # Web UI (3-page flow)
├── src/
│   ├── config.py          # Pipeline configuration
│   ├── preprocessing.py   # Stage 1: Noise removal, artifact cleanup
│   ├── super_resolution.py # Stage 2: Real-ESRGAN upscaling
│   ├── detail_enhancement.py # Stage 3: Stable Diffusion refinement
│   ├── jewelry_enhancement.py # Stage 4: Metallic/sparkle enhancement
│   ├── postprocessing.py  # Stage 5: Background removal, final output
│   ├── object_detection.py # Jewelry object detection
│   └── virtual_tryon.py   # Virtual try-on module
├── data/
│   ├── degraded-light/    # Input images (degraded)
│   ├── enhanced/          # Output images
│   └── tryon-images/      # Virtual try-on person images
│       ├── WATCH/
│       ├── RINGS/
│       ├── EARRINGS/
│       ├── NECKLACE/
│       └── BRACELET/
└── scripts/
    ├── create_light_degraded.py  # Create degraded dataset
    └── generate_tryon_images.py  # Generate try-on images
```

## Configuration

Environment variables:
- `VLM_API_URL`: VLM server URL (default: `http://0.0.0.0:9999`)

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for faster processing)
- ~8GB GPU memory for full pipeline

## License

Proprietary - All rights reserved.


"""
Generate Try-On Images Using Pollination (Flux)
===============================================
Generates realistic person/hand images for virtual try-on.
Creates 5 images per jewelry type using Flux model via Pollination API.
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict
import base64
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
BASE_DIR = Path(__file__).parent.parent
TRYON_IMAGES_DIR = BASE_DIR / "data" / "tryon-images"

# Pollination API Configuration
POLLINATION_API_KEY = os.getenv("POLLINATION_API_KEY", "")
# Pollination uses simple GET request with prompt as parameter
POLLINATION_API_URL = "https://image.pollinations.ai/prompt"

# Jewelry type prompts for realistic image generation
JEWELRY_PROMPTS = {
    "RINGS": [
        "Ultra-realistic professional product photography of a human hand, elegant fingers gracefully extended, no jewelry or objects visible, only bare hand, natural skin tone with subtle texture, soft beige background, perfect studio lighting with soft shadows, extremely detailed fingers and knuckles, photorealistic, 8k quality, commercial photography, high-end catalog style",
        "Close-up macro photography of a bare hand showing fingers, no rings or jewelry, only natural hand, natural skin texture with fine details, professional product photography, warm off-white background, dramatic soft lighting, detailed finger joints and nail beds, ultra-realistic, 4k resolution, luxury brand aesthetic",
        "Human hand in elegant natural pose, fingers clearly visible and well-positioned, no objects or jewelry, only bare hand, realistic skin tone with natural variations, professional studio photography with three-point lighting, light cream colored background, extremely detailed skin pores and texture, photorealistic, high-end commercial quality, 8k",
        "Hand photography masterpiece, fingers clearly visible in perfect composition, no jewelry visible, only bare hand, natural skin texture with realistic imperfections, professional lighting setup with soft key light, soft beige background, ultra-detailed, photorealistic, high resolution, luxury product photography style, 4k",
        "Close-up professional macro photo of a bare hand, fingers spread naturally in elegant pose, no rings or objects, only hand, realistic skin tone with natural highlights, studio lighting with rim lighting effect, warm sand colored background, extremely detailed, photorealistic quality, commercial grade, 8k resolution"
    ],
    "BRACELET": [
        "Ultra-realistic professional product photography of a person's elegant arm and wrist, upper body gracefully visible, no bracelets or jewelry visible, only bare arm and wrist, natural skin tone with realistic texture, elegant light beige background, perfect studio lighting with soft shadows, extremely detailed wrist area and arm structure, photorealistic, 8k quality, luxury catalog style",
        "Person showing wrist and lower arm in elegant pose, no bracelets or objects, only bare arm, professional product photography style, natural lighting with soft fill, realistic skin texture with fine details, warm off-white background, ultra-detailed wrist bones and tendons, photorealistic, high-end commercial quality, 4k resolution",
        "Human arm and wrist in natural elegant pose, no jewelry visible, only bare arm, realistic skin tone with natural highlights and shadows, professional studio photography with three-point lighting, soft cream colored background, extremely detailed wrist area with visible anatomy, photorealistic, luxury brand aesthetic, 8k",
        "Professional macro photography of person's wrist area, arm elegantly positioned, no bracelets or objects, only bare wrist, natural skin texture with realistic pores and fine lines, professional lighting setup with soft key and fill lights, light taupe background, ultra-realistic, extremely detailed, commercial grade, 4k resolution",
        "Upper body shot showing wrist and arm in perfect composition, no jewelry visible, only bare arm, professional photography with dramatic lighting, natural skin tone with realistic variations, studio lighting with rim light effect, warm beige background, extremely detailed, photorealistic quality, high-end photography, 8k resolution"
    ],
    "NECKLACE": [
        "Ultra-realistic professional product photography of a person's elegant upper body, neck and chest gracefully visible, no necklaces or jewelry visible, only bare neck and chest, natural skin tone with realistic texture and subtle variations, soft beige background, perfect studio lighting with soft shadows, extremely detailed neck area and collarbone, photorealistic, 8k quality, luxury catalog style",
        "Person showing neck and upper chest area in elegant pose, no necklaces or objects, only bare neck, professional product photography style, natural lighting with soft directional light, realistic skin texture with fine details and natural pores, warm off-white background, ultra-detailed neck anatomy, photorealistic, high-end commercial quality, 4k resolution",
        "Human upper body in natural elegant pose, neck and chest beautifully visible, no jewelry visible, only bare neck and chest, realistic skin tone with natural highlights and shadows, professional studio photography with three-point lighting setup, light cream colored background, extremely detailed skin texture and anatomy, photorealistic, luxury brand aesthetic, 8k",
        "Professional macro photography of person's neck and chest area, elegant composition, no necklaces or objects, only bare neck, natural skin texture with realistic fine lines and pores, professional lighting setup with soft key and fill lights creating depth, soft beige background, ultra-realistic, extremely detailed, commercial grade, 4k resolution",
        "Upper body shot showing neck area in perfect composition, no jewelry visible, only bare neck, professional photography with dramatic lighting, natural skin tone with realistic color variations, studio lighting with rim light effect creating elegant separation, warm sand colored background, extremely detailed, photorealistic quality, high-end photography, 8k resolution"
    ],
    "EARRINGS": [
        "Ultra-realistic professional product photography of a person's elegant face and head in side profile view, ear clearly and beautifully visible from the side, no earrings or jewelry visible, only bare ear in side view, natural skin tone with realistic texture and subtle variations, warm off-white background, perfect studio lighting with soft shadows, extremely detailed ear anatomy in side profile, photorealistic, 8k quality, luxury catalog style",
        "Person showing side profile of face with ear clearly visible from the side, no earrings or objects, only bare ear in side view, professional product photography style, natural lighting with soft directional light creating depth, realistic skin texture with fine details and natural pores, soft beige background, ultra-detailed ear structure in side profile view, photorealistic, high-end commercial quality, 4k resolution",
        "Human face in elegant side profile pose, ear beautifully visible from the side, no jewelry visible, only bare ear, realistic skin tone with natural highlights and shadows, professional studio photography with three-point lighting setup, light cream colored background, extremely detailed skin texture and ear anatomy in profile view, photorealistic, luxury brand aesthetic, 8k",
        "Professional macro photography of person's face and ear in side profile view, elegant composition showing ear from the side, no earrings or objects, only bare ear, natural skin texture with realistic fine lines and pores, professional lighting setup with soft key and fill lights, warm sand colored background, ultra-realistic, extremely detailed ear structure in side view, commercial grade, 4k resolution",
        "Headshot showing face and ear in perfect side profile composition, no jewelry visible, only bare ear from side, professional photography with dramatic lighting, natural skin tone with realistic color variations, studio lighting with rim light effect creating elegant separation, elegant light beige background, extremely detailed, photorealistic quality, high-end photography, 8k resolution"
    ],
    "WATCH": [
        "Ultra-realistic professional product photography of a human hand and wrist, wrist area clearly and elegantly visible, no watches or jewelry visible, only bare hand and wrist, natural skin tone with realistic texture and subtle variations, soft beige background, perfect studio lighting with soft shadows, extremely detailed wrist anatomy and hand structure, photorealistic, 8k quality, luxury catalog style",
        "Close-up macro shot of wrist and lower hand in elegant pose, no watches or objects, only bare wrist and hand, natural lighting with soft directional light, realistic skin texture with fine details and natural pores, professional product photography style, warm off-white background, ultra-detailed wrist bones and tendons, photorealistic, high-end commercial quality, 4k resolution",
        "Human wrist in natural elegant pose, hand beautifully visible, no jewelry visible, only bare wrist and hand, realistic skin tone with natural highlights and shadows, professional studio photography with three-point lighting setup, light cream colored background, extremely detailed wrist area with visible anatomy, photorealistic, luxury brand aesthetic, 8k",
        "Wrist photography masterpiece, clearly visible wrist area in perfect composition, no watches or objects, only bare wrist, natural skin texture with realistic imperfections and fine lines, professional lighting setup with soft key and fill lights, soft beige background, ultra-realistic, extremely detailed, commercial grade, 4k resolution",
        "Close-up professional macro photo of wrist area, hand elegantly positioned, no watches or jewelry, only bare wrist and hand, realistic skin tone with natural color variations, studio lighting with rim light effect creating depth, warm sand colored background, extremely detailed, photorealistic quality, high-end photography, 8k resolution"
    ]
}


def generate_image_with_pollination(prompt: str, api_key: str = None) -> bytes:
    """
    Generate image using Pollination API (Flux model).
    
    Args:
        prompt: Text prompt for image generation
        api_key: Pollination API key (optional, usually not needed)
    
    Returns:
        Image bytes
    """
    try:
        # Pollination API - simple GET request
        # Format: https://image.pollinations.ai/prompt/{prompt}?{params}
        
        # URL encode prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Build URL with parameters
        params = {
            "model": "flux",
            "width": 1024,
            "height": 1024,
            "seed": int(time.time() * 1000) % 1000000,  # Random seed
            "nologo": "true",
            "enhance": "true"
        }
        
        # Construct URL - Pollination format: /prompt/{prompt}?params
        url = f"{POLLINATION_API_URL}/{encoded_prompt}"
        
        # Add query parameters
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        if query_string:
            url = f"{url}?{query_string}"
        
        print(f"  📸 Generating with Pollination (Flux)...")
        print(f"  Prompt: {prompt[:80]}...")
        
        # Make GET request (Pollination returns image directly)
        response = requests.get(url, timeout=120, stream=True)
        
        if response.status_code == 200:
            # Pollination returns image directly
            content_type = response.headers.get('content-type', '')
            
            if 'image' in content_type:
                return response.content
            else:
                # Try to read as image anyway
                return response.content
        else:
            raise Exception(f"Pollination API returned status {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f"  ⚠️ Pollination API error: {e}")
        raise e


def save_image(image_bytes: bytes, output_path: Path):
    """Save image bytes to file"""
    try:
        # Verify it's a valid image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, 'JPEG', quality=95)
        print(f"  ✅ Saved: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to save image: {e}")
        return False


def generate_tryon_images(jewelry_type: str, num_images: int = 5, api_key: str = None):
    """
    Generate try-on images for a specific jewelry type.
    
    Args:
        jewelry_type: One of BRACELET, EARRINGS, NECKLACE, RINGS, WATCH
        num_images: Number of images to generate (default: 5)
        api_key: API key for image generation service
    """
    jewelry_type = jewelry_type.upper()
    
    if jewelry_type not in JEWELRY_PROMPTS:
        print(f"❌ Invalid jewelry type: {jewelry_type}")
        return
    
    print(f"\n{'='*60}")
    print(f"🎨 Generating {num_images} images for {jewelry_type}")
    print(f"{'='*60}")
    
    output_dir = TRYON_IMAGES_DIR / jewelry_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = JEWELRY_PROMPTS[jewelry_type]
    
    for i in range(num_images):
        print(f"\n📸 Image {i+1}/{num_images}:")
        
        # Use prompt from list or generate variation
        if i < len(prompts):
            prompt = prompts[i]
        else:
            # Generate variation of first prompt
            prompt = prompts[0] + f", variation {i+1}"
        
        # Generate using Pollination API only
        image_bytes = generate_image_with_pollination(prompt, api_key)
        
        if image_bytes:
            # Determine filename
            if jewelry_type == "RINGS" or jewelry_type == "WATCH":
                filename = f"hand{i+1}.jpg"
            else:
                filename = f"person{i+1}.jpg"
            
            output_path = output_dir / filename
            save_image(image_bytes, output_path)
            
            # Small delay to avoid rate limiting
            time.sleep(2)
        else:
            print(f"  ❌ Failed to generate image {i+1}")
    
    print(f"\n✅ Completed generating images for {jewelry_type}")


def generate_all_tryon_images(api_key: str = None):
    """Generate images for all jewelry types"""
    print("\n" + "="*60)
    print("🎨 GENERATING TRY-ON IMAGES FOR ALL JEWELRY TYPES")
    print("="*60)
    print("\nThis will generate 5 images per jewelry type (25 total images).")
    print("This may take several minutes...\n")
    
    jewelry_types = ["BRACELET", "EARRINGS", "NECKLACE", "RINGS", "WATCH"]
    
    for jewelry_type in jewelry_types:
        generate_tryon_images(jewelry_type, num_images=5, api_key=api_key)
        print()  # Empty line between types
    
    print("="*60)
    print("✅ ALL IMAGES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nImages saved to: {TRYON_IMAGES_DIR}")
    print("\nNext steps:")
    print("1. Verify images in each folder")
    print("2. Run the application: python3 vision_app.py")
    print("3. Test virtual try-on feature!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate try-on images using Pollination/Flux")
    parser.add_argument(
        "--type",
        choices=["BRACELET", "EARRINGS", "NECKLACE", "RINGS", "WATCH", "ALL"],
        default="ALL",
        help="Jewelry type to generate images for (default: ALL)"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of images to generate per type (default: 5)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Pollination API key (or set POLLINATION_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("POLLINATION_API_KEY", "")
    
    if args.type == "ALL":
        generate_all_tryon_images(api_key=api_key)
    else:
        generate_tryon_images(args.type, num_images=args.num, api_key=api_key)


"""
Create Light Degraded Dataset (15% degradation)
==============================================
For fashion tech use case - images are already good quality,
just need slight degradation for upscaling demonstration.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import random

# Add src/dataset to path directly (avoid importing full src package)
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "dataset"))

# Import directly from degradation module
from degradation import LightDegradation

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    
    # Paths
    source_dir = BASE_DIR / "data" / "raw" / "high_quality"
    output_dir = BASE_DIR / "data" / "degraded-light"
    
    print("=" * 60)
    print("Creating Light Degraded Dataset (15% degradation)")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        print("   Please ensure high-quality images are in data/raw/high_quality/")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize degrader
    degrader = LightDegradation()
    
    # Get all image files
    image_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No images found in {source_dir}")
        sys.exit(1)
    
    print(f"Processing {len(image_files)} images...\n")
    
    # Process each image
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Failed to load: {img_path.name}")
            continue
        
        # Apply light degradation
        degraded = degrader.degrade(img)
        
        # Save degraded image
        output_file = output_dir / img_path.name
        cv2.imwrite(str(output_file), degraded)
        print(f"✓ Created: {output_file.name}")
    
    print("\n" + "=" * 60)
    print("✅ Light degradation complete!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print("\n💡 These images are only 15% degraded - perfect for")
    print("   demonstrating upscaling and enhancement capabilities!")


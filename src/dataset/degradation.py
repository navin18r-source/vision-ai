"""
Image Degradation Module
========================
Creates realistic low-quality versions of high-quality jewelry images.
Supports light degradation (15%) for fashion tech use case.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import random


class LightDegradation:
    """
    Light degradation (15%) - for fashion tech use case.
    Images are already good quality, just need slight degradation for upscaling demo.
    """
    
    def __init__(self):
        # Light degradation parameters (15%)
        self.jpeg_quality = (80, 90)  # Good quality with some compression
        self.resize_factor = (0.80, 0.90)  # Slight downscale
        self.blur_kernel = (3, 5)  # Light blur
        self.noise_sigma = (3, 8)  # Light noise
        self.color_shift_strength = 0.03  # Subtle color shift
    
    def apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Apply very light JPEG compression"""
        quality = random.randint(*self.jpeg_quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def apply_resolution_loss(self, img: np.ndarray) -> np.ndarray:
        """Minimal downscale/upscale to simulate slight quality loss"""
        factor = random.uniform(*self.resize_factor)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w*factor), int(h*factor)), 
                          interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def apply_light_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply light Gaussian blur"""
        kernel_size = random.choice(range(self.blur_kernel[0], self.blur_kernel[1] + 1, 2))
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def apply_light_noise(self, img: np.ndarray) -> np.ndarray:
        """Add minimal Gaussian noise"""
        sigma = random.randint(*self.noise_sigma)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_subtle_color_shift(self, img: np.ndarray) -> np.ndarray:
        """Very subtle color shift"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] += random.uniform(-2, 2)  # Minimal hue shift
        hsv[:,:,1] *= random.uniform(0.98, 1.02)  # Minimal saturation
        hsv[:,:,2] *= random.uniform(0.98, 1.02)  # Minimal value
        hsv = np.clip(hsv, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def degrade(self, img: np.ndarray, methods: Optional[list] = None) -> np.ndarray:
        """
        Apply light degradation (15%).
        Uses 2-3 methods randomly for noticeable but light quality loss.
        
        Args:
            img: Input BGR image
            methods: Optional list of methods, or None for random selection
            
        Returns:
            Lightly degraded image
        """
        if methods is None:
            # Randomly select 2-3 degradation methods for 15% degradation
            all_methods = ['jpeg', 'resolution', 'blur', 'noise', 'color']
            num_methods = random.randint(2, 3)  # 2-3 methods for 15% degradation
            methods = random.sample(all_methods, num_methods)
        
        result = img.copy()
        
        for method in methods:
            if method == 'jpeg':
                result = self.apply_jpeg_compression(result)
            elif method == 'resolution':
                result = self.apply_resolution_loss(result)
            elif method == 'blur':
                result = self.apply_light_blur(result)
            elif method == 'noise':
                result = self.apply_light_noise(result)
            elif method == 'color':
                result = self.apply_subtle_color_shift(result)
        
        return result


class AdditionalDegradation:
    """
    Additional degradation for already degraded images.
    Applies: blur, quality reduction, compression, resolution loss.
    Used to further degrade images from degraded-light folder.
    """
    
    def __init__(self):
        # Additional degradation parameters
        self.jpeg_quality = (40, 60)  # Significant compression
        self.resize_factor = (0.5, 0.7)  # Significant resolution loss
        self.blur_kernel = (7, 15)  # Moderate to heavy blur
        self.noise_sigma = (8, 15)  # Moderate noise
        self.compression_iterations = (2, 3)  # Multiple compression passes
    
    def apply_heavy_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """Apply heavy JPEG compression with multiple passes"""
        result = img.copy()
        iterations = random.randint(*self.compression_iterations)
        
        for _ in range(iterations):
            quality = random.randint(*self.jpeg_quality)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', result, encode_param)
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return result
    
    def apply_resolution_loss(self, img: np.ndarray) -> np.ndarray:
        """Apply significant resolution loss"""
        factor = random.uniform(*self.resize_factor)
        h, w = img.shape[:2]
        # Downscale significantly
        small = cv2.resize(img, (int(w*factor), int(h*factor)), 
                          interpolation=cv2.INTER_LINEAR)
        # Upscale back (loses detail)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    def apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply moderate to heavy Gaussian blur"""
        kernel_size = random.choice(range(self.blur_kernel[0], self.blur_kernel[1] + 1, 2))
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Add moderate Gaussian noise"""
        sigma = random.randint(*self.noise_sigma)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def apply_motion_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply motion blur effect"""
        size = random.randint(10, 20)
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Random angle
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        return cv2.filter2D(img, -1, kernel)
    
    def degrade(self, img: np.ndarray, methods: Optional[list] = None) -> np.ndarray:
        """
        Apply additional degradation: blur, compression, quality reduction, resolution loss.
        Uses 3-4 methods for significant quality reduction.
        
        Args:
            img: Input BGR image (already degraded)
            methods: Optional list of methods, or None for random selection
            
        Returns:
            Further degraded image
        """
        if methods is None:
            # Always include compression and resolution loss, plus 1-2 more
            all_methods = ['jpeg', 'resolution', 'blur', 'noise', 'motion_blur']
            # Always apply compression and resolution loss
            methods = ['jpeg', 'resolution']
            # Add 1-2 more methods
            additional = random.sample([m for m in all_methods if m not in methods], 
                                      random.randint(1, 2))
            methods.extend(additional)
        
        result = img.copy()
        
        for method in methods:
            if method == 'jpeg':
                result = self.apply_heavy_jpeg_compression(result)
            elif method == 'resolution':
                result = self.apply_resolution_loss(result)
            elif method == 'blur':
                result = self.apply_blur(result)
            elif method == 'noise':
                result = self.apply_noise(result)
            elif method == 'motion_blur':
                result = self.apply_motion_blur(result)
        
        return result


def create_light_degraded_dataset(
    source_dir: Path,
    output_dir: Path,
    pattern: str = "*.png"
):
    """
    Create lightly degraded dataset (15% degradation).
    
    Args:
        source_dir: Directory containing high-quality images
        output_dir: Directory to save degraded images
        pattern: File pattern to match (default: "*.png")
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    degrader = LightDegradation()
    
    # Get all image files
    image_files = list(source_path.glob(pattern))
    if not image_files:
        # Try other extensions
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Failed to load: {img_path}")
            continue
        
        # Apply light degradation
        degraded = degrader.degrade(img)
        
        # Save degraded image
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), degraded)
        print(f"✓ Created: {output_file.name}")
    
    print(f"\n✅ Light degradation complete! Output: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python degradation.py <source_dir> <output_dir>")
        sys.exit(1)
    
    source_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    create_light_degraded_dataset(source_dir, output_dir)


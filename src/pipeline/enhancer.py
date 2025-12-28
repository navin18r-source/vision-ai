"""
Jewelry Image Enhancement Pipeline
===================================
Transforms low-quality jewelry images into high-quality outputs with:
- Background removal (extract jewelry object only)
- Denoising and artifact removal
- Super-resolution upscaling
- Sharpening and detail enhancement
- Clean plain background output
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
import io


class JewelryEnhancer:
    """
    Complete enhancement pipeline for jewelry images.
    Outputs clean jewelry object on plain background.
    """
    
    def __init__(self, 
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 output_size: Tuple[int, int] = (512, 512),
                 use_gpu: bool = False):
        """
        Args:
            background_color: RGB color for plain background (default white)
            output_size: Output image dimensions (width, height)
            use_gpu: Use GPU acceleration if available
        """
        self.background_color = background_color
        self.output_size = output_size
        self.use_gpu = use_gpu
        self.rembg_session = None
        
        print("🔧 Initializing Jewelry Enhancement Pipeline...")
        self._init_models()
        print("✅ Pipeline ready!")
    
    def _init_models(self):
        """Initialize all required models"""
        # Background removal will be loaded on first use
        try:
            from rembg import new_session
            self.rembg_session = new_session("u2net")
            print("  ✓ Background removal model loaded")
        except ImportError:
            print("  ⚠ rembg not installed, background removal disabled")
            self.rembg_session = None
    
    # =========================================================================
    #                         BACKGROUND REMOVAL
    # =========================================================================
    
    def remove_background(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove background and extract jewelry object only.
        
        Args:
            img: Input image (BGR numpy array)
        
        Returns:
            Tuple of (foreground image BGRA, alpha mask)
        """
        from rembg import remove
        
        # Convert BGR to RGB for rembg
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Remove background
        output = remove(pil_img, session=self.rembg_session)
        
        # Convert back to numpy
        output_np = np.array(output)
        
        # Extract alpha channel as mask
        if output_np.shape[2] == 4:
            alpha_mask = output_np[:, :, 3]
            foreground = output_np
        else:
            # Create mask from non-background areas
            gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
            _, alpha_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            foreground = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGRA)
            foreground[:, :, 3] = alpha_mask
        
        return foreground, alpha_mask
    
    def apply_plain_background(self, 
                                foreground: np.ndarray, 
                                bg_color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Place foreground object on plain colored background.
        
        Args:
            foreground: BGRA image with alpha channel
            bg_color: RGB background color (default: self.background_color)
        
        Returns:
            BGR image with plain background
        """
        if bg_color is None:
            bg_color = self.background_color
        
        h, w = foreground.shape[:2]
        
        # Create plain background (BGR)
        background = np.full((h, w, 3), bg_color[::-1], dtype=np.uint8)  # RGB to BGR
        
        # Extract alpha channel
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3:4] / 255.0
            fg_bgr = foreground[:, :, :3]
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            fg_bgr = foreground
        
        # Blend foreground with background
        result = (fg_bgr * alpha + background * (1 - alpha)).astype(np.uint8)
        
        return result
    
    # =========================================================================
    #                         IMAGE ENHANCEMENT
    # =========================================================================
    
    def denoise(self, img: np.ndarray, strength: int = 10) -> np.ndarray:
        """Remove noise while preserving edges"""
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    
    def remove_jpeg_artifacts(self, img: np.ndarray) -> np.ndarray:
        """Remove JPEG compression artifacts"""
        return cv2.bilateralFilter(img, 9, 75, 75)
    
    def sharpen(self, img: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """Sharpen image using unsharp mask"""
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1 + amount, gaussian, -amount, 0)
        return sharpened
    
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance local contrast using CLAHE"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Auto white balance for accurate colors"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    def upscale(self, img: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Upscale image using high-quality interpolation.
        For better results, use Real-ESRGAN (separate module).
        """
        h, w = img.shape[:2]
        new_size = (w * scale, h * scale)
        
        # Use INTER_LANCZOS4 for best quality without deep learning
        upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        return upscaled
    
    def enhance_metallic(self, img: np.ndarray, strength: float = 1.2) -> np.ndarray:
        """Enhance metallic reflections for jewelry"""
        # Extract highlights
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Enhance bright areas
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        highlight_mask = highlights.astype(np.float32) / 255.0
        l = l.astype(np.float32)
        l = l + (l * highlight_mask * (strength - 1))
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def add_sparkle(self, img: np.ndarray, threshold: int = 245) -> np.ndarray:
        """Add subtle sparkle effect for gemstones"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, sparkles = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Create subtle glow
        glow = cv2.GaussianBlur(sparkles, (5, 5), 0)
        glow = glow.astype(np.float32) / 255.0
        
        result = img.astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] + glow * 20
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # =========================================================================
    #                         MAIN ENHANCEMENT PIPELINE
    # =========================================================================
    
    def enhance(self, 
                img: Union[np.ndarray, str, Image.Image],
                remove_bg: bool = True,
                bg_color: Tuple[int, int, int] = None,
                upscale_factor: int = 2) -> np.ndarray:
        """
        Full enhancement pipeline.
        
        Args:
            img: Input image (numpy array, file path, or PIL Image)
            remove_bg: Whether to remove background
            bg_color: Background color RGB (default white)
            upscale_factor: Upscaling factor (1, 2, or 4)
        
        Returns:
            Enhanced image with plain background
        """
        # Load image if path provided
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise ValueError("Failed to load image")
        
        original_shape = img.shape[:2]
        print(f"📥 Input: {original_shape[1]}x{original_shape[0]}")
        
        # Step 1: Initial cleanup
        print("  1️⃣ Removing artifacts...")
        result = self.remove_jpeg_artifacts(img)
        result = self.denoise(result, strength=8)
        
        # Step 2: White balance
        print("  2️⃣ Color correction...")
        result = self.auto_white_balance(result)
        
        # Step 3: Upscale
        if upscale_factor > 1:
            print(f"  3️⃣ Upscaling {upscale_factor}x...")
            result = self.upscale(result, scale=upscale_factor)
        
        # Step 4: Enhance contrast and details
        print("  4️⃣ Enhancing details...")
        result = self.enhance_contrast(result)
        result = self.sharpen(result, amount=0.8)
        
        # Step 5: Jewelry-specific enhancements
        print("  5️⃣ Jewelry polish...")
        result = self.enhance_metallic(result, strength=1.15)
        result = self.add_sparkle(result)
        
        # Step 6: Background removal
        if remove_bg and self.rembg_session is not None:
            print("  6️⃣ Removing background...")
            foreground, mask = self.remove_background(result)
            result = self.apply_plain_background(foreground, bg_color)
        
        # Step 7: Final resize to output size
        print(f"  7️⃣ Resizing to {self.output_size}...")
        result = cv2.resize(result, self.output_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Step 8: Final sharpening
        result = self.sharpen(result, amount=0.3)
        
        print(f"📤 Output: {result.shape[1]}x{result.shape[0]}")
        print("✅ Enhancement complete!")
        
        return result
    
    def enhance_quick(self, img: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """Quick enhancement without background removal (faster)"""
        return self.enhance(img, remove_bg=False, upscale_factor=2)
    
    def enhance_full(self, img: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """Full enhancement with background removal"""
        return self.enhance(img, remove_bg=True, upscale_factor=2)


# ============================================================================
#                              TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhancer.py <input_image>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "enhanced_output.png"
    
    enhancer = JewelryEnhancer()
    result = enhancer.enhance(input_path)
    
    cv2.imwrite(output_path, result)
    print(f"Saved to: {output_path}")


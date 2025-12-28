"""
Stage 5: Post-Processing
========================
Final refinement and output preparation:
- Background removal and replacement
- Final sharpening and clarity
- Color grading for consistency
- Output formatting and resizing
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from .config import PipelineConfig


class PostProcessor:
    """Stage 5: Final post-processing and output preparation"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.rembg_session = None
        self._init_background_removal()
    
    def _init_background_removal(self):
        """Initialize background removal model"""
        try:
            from rembg import new_session
            self.rembg_session = new_session("u2net")
            print("  ✓ Background removal model loaded")
        except ImportError:
            print("  ⚠ rembg not installed - background removal disabled")
        except Exception as e:
            print(f"  ⚠ Background removal init failed: {e}")
    
    def process(self, 
                img: np.ndarray,
                remove_background: bool = None) -> np.ndarray:
        """
        Full post-processing pipeline - MINIMAL processing to preserve quality.
        
        Args:
            img: Input BGR image
            remove_background: Whether to remove and replace background
            
        Returns:
            Post-processed image (BGRA if background removed, BGR otherwise)
        """
        if remove_background is None:
            remove_background = self.config.remove_background
        
        result = img.copy()
        
        # MINIMAL PROCESSING - preserve original quality
        # Only apply very subtle sharpening (no blur, no color changes)
        result = self._minimal_sharpen(result)
        
        # Remove background (returns BGRA with alpha channel)
        if remove_background and self.rembg_session is not None:
            result = self._remove_and_replace_background(result)
        
        return result
    
    def _minimal_sharpen(self, img: np.ndarray) -> np.ndarray:
        """Very subtle sharpening that preserves quality"""
        # Use unsharp mask with very small sigma for subtle effect
        gaussian = cv2.GaussianBlur(img, (0, 0), 0.5)  # Small sigma
        sharpened = cv2.addWeighted(img, 1.2, gaussian, -0.2, 0)  # Subtle
        return sharpened
    
    def _remove_and_replace_background(self, img: np.ndarray) -> np.ndarray:
        """
        Remove background and return ONLY the object with ALPHA channel (BGRA).
        Crops to the exact bounding box of the object - no extra space.
        """
        from rembg import remove
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Remove background - get RGBA output
        output = remove(pil_img, session=self.rembg_session)
        output_np = np.array(output)
        
        # Convert RGBA to BGRA
        if output_np.shape[2] == 4:
            bgra = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGRA)
        else:
            # If no alpha, create one
            alpha = np.ones((output_np.shape[0], output_np.shape[1], 1), dtype=np.uint8) * 255
            rgb_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            bgra = np.concatenate([rgb_bgr, alpha], axis=2)
        
        # CROP TO OBJECT ONLY - remove all transparent space
        bgra = self._crop_to_object(bgra)
        
        return bgra
    
    def _crop_to_object(self, bgra: np.ndarray, padding: int = 5) -> np.ndarray:
        """
        Crop image to the bounding box of non-transparent pixels.
        
        Args:
            bgra: 4-channel BGRA image
            padding: Pixels of padding around the object
            
        Returns:
            Cropped BGRA image containing only the object
        """
        # Get alpha channel
        alpha = bgra[:, :, 3]
        
        # Find non-transparent pixels (alpha > 10 to ignore near-transparent edges)
        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # No object found, return original
            return bgra
        
        # Get bounding box
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add padding
        h, w = bgra.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding + 1)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding + 1)
        
        # Crop
        cropped = bgra[y_min:y_max, x_min:x_max]
        
        print(f"   → Cropped from {w}x{h} to {cropped.shape[1]}x{cropped.shape[0]} (object only)")
        
        return cropped
    
    def _final_sharpen(self, img: np.ndarray) -> np.ndarray:
        """Apply final sharpening with unsharp mask (handles BGR and BGRA)"""
        amount = self.config.sharpen_amount
        
        # Handle both 3-channel (BGR) and 4-channel (BGRA) images
        if img.shape[2] == 4:
            # Has alpha - sharpen only RGB channels
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            
            gaussian = cv2.GaussianBlur(bgr, (0, 0), 2)
            sharpened = cv2.addWeighted(bgr, 1 + amount, gaussian, -amount, 0)
            
            # Recombine with alpha
            return np.dstack([sharpened, alpha])
        else:
            # Standard BGR sharpening
            gaussian = cv2.GaussianBlur(img, (0, 0), 2)
            sharpened = cv2.addWeighted(img, 1 + amount, gaussian, -amount, 0)
            return sharpened
    
    def _adjust_saturation(self, img: np.ndarray) -> np.ndarray:
        """Adjust saturation for vibrant colors - PRESERVE ORIGINAL COLORS (handles BGR and BGRA)"""
        # PRESERVE COLORS - minimal saturation boost
        boost = 1.02  # Very minimal boost instead of config value
        
        if img.shape[2] == 4:
            # Has alpha - process only BGR channels
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * boost
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return np.dstack([result_bgr, alpha])
        else:
            # Standard BGR processing
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * boost
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _final_color_grade(self, img: np.ndarray) -> np.ndarray:
        """Apply final color grading for consistency (handles BGR and BGRA)"""
        # Handle both 3-channel (BGR) and 4-channel (BGRA) images
        if img.shape[2] == 4:
            # Has alpha - process only BGR channels
            result_bgr = img[:, :, :3].copy()
            alpha = img[:, :, 3]
        else:
            result_bgr = img.copy()
            alpha = None
        
        # Slight curves adjustment for professional look
        # Create lookup table for subtle S-curve
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Subtle S-curve: slight shadow lift, highlight compression
            value = i / 255.0
            # S-curve formula
            curved = 0.5 * (1 + np.tanh(3 * (value - 0.5)))
            # Blend with original (subtle effect)
            blended = 0.85 * value + 0.15 * curved
            lut[i] = int(np.clip(blended * 255, 0, 255))
        
        # Apply LUT to BGR channels
        result_bgr = cv2.LUT(result_bgr, lut)
        
        # Recombine with alpha if present
        if alpha is not None:
            return np.dstack([result_bgr, alpha])
        else:
            return result_bgr
    
    def _resize_output(self, img: np.ndarray) -> np.ndarray:
        """Resize to final output size"""
        target_size = self.config.output_size
        
        if img.shape[:2][::-1] != target_size:
            # Use high-quality interpolation
            return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        return img
    
    def _final_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Final cleanup pass (handles BGR and BGRA)"""
        # Handle both 3-channel (BGR) and 4-channel (BGRA) images
        if img.shape[2] == 4:
            # Has alpha - process only BGR channels
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            
            # Very subtle noise reduction to clean up any artifacts
            result_bgr = cv2.bilateralFilter(bgr, 5, 40, 40)
            
            # Recombine with alpha
            return np.dstack([result_bgr, alpha])
        else:
            # Standard BGR processing
            return cv2.bilateralFilter(img, 5, 40, 40)
    
    @property
    def can_remove_background(self) -> bool:
        """Check if background removal is available"""
        return self.rembg_session is not None


def postprocess(img: np.ndarray, 
                remove_background: bool = True,
                config: PipelineConfig = None) -> np.ndarray:
    """Convenience function for post-processing"""
    return PostProcessor(config).process(img, remove_background)


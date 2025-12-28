"""
Object Detection and Segmentation
==================================
Detects and isolates jewelry object from the image before enhancement.
This ensures we only enhance the jewelry, not the entire image.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from pathlib import Path


class JewelryDetector:
    """Detect and isolate jewelry object from background"""
    
    def __init__(self):
        self.rembg_session = None
        self._init_background_removal()
    
    def _init_background_removal(self):
        """Initialize background removal model for object detection"""
        try:
            from rembg import new_session
            # Use u2net model - best for object segmentation
            self.rembg_session = new_session("u2net")
            print("  ✓ Object detection model loaded")
        except ImportError:
            print("  ⚠ rembg not installed - using fallback detection")
            self.rembg_session = None
        except Exception as e:
            print(f"  ⚠ Object detection init failed: {e}")
            self.rembg_session = None
    
    def detect_object(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Detect jewelry object and return mask + bounding box.
        
        Args:
            img: Input BGR image
            
        Returns:
            Tuple of (object_mask, object_image, bbox)
            bbox: (x, y, width, height) of detected object
        """
        if self.rembg_session is None:
            # Fallback: use entire image
            h, w = img.shape[:2]
            mask = np.ones((h, w), dtype=np.uint8) * 255
            return mask, img, (0, 0, w, h)
        
        from rembg import remove
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Remove background to get mask
        output = remove(pil_img, session=self.rembg_session)
        output_np = np.array(output)
        
        # Extract alpha channel as mask
        if output_np.shape[2] == 4:
            mask = output_np[:, :, 3]
        else:
            # Create mask from non-background areas
            gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find bounding box of the object
        bbox = self._get_bounding_box(mask)
        
        # Crop to bounding box with padding
        x, y, w, h = bbox
        padding = 20  # Add padding around object
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Crop image and mask
        cropped_img = img[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        return cropped_mask, cropped_img, (x, y, w, h)
    
    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of non-zero pixels in mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No object detected, return full image
            h, w = mask.shape[:2]
            return (0, 0, w, h)
        
        # Find largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def apply_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to image, setting background to transparent"""
        # Ensure mask is 3-channel for broadcasting
        if len(mask.shape) == 2:
            mask_3d = np.stack([mask, mask, mask], axis=2)
        else:
            mask_3d = mask
        
        # Normalize mask to 0-1
        mask_norm = mask_3d.astype(np.float32) / 255.0
        
        # Apply mask
        masked_img = (img.astype(np.float32) * mask_norm).astype(np.uint8)
        
        return masked_img


def detect_jewelry_object(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Convenience function for object detection"""
    detector = JewelryDetector()
    return detector.detect_object(img)






"""
Stage 4: Jewelry-Specific Enhancement (KEY DIFFERENTIATOR)
==========================================================
Specialized enhancements for jewelry photography:
- Metallic surface enhancement (gold, silver, platinum)
- Gemstone sparkle and brilliance
- Reflection and highlight restoration
- Material-aware color correction
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .config import PipelineConfig


class JewelryEnhancer:
    """
    Stage 4: Jewelry-specific enhancement techniques.
    
    This is the KEY DIFFERENTIATOR that makes our pipeline unique.
    These techniques are specifically designed for jewelry photography.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
    
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        MINIMAL jewelry enhancement - preserve original quality.
        
        Args:
            img: Input BGR image
            
        Returns:
            Enhanced BGR image with subtle jewelry-specific improvements
        """
        result = img.copy()
        
        # MINIMAL PROCESSING - only subtle enhancements
        # 1. Subtle highlight enhancement for metallic shine
        result = self._subtle_highlight_boost(result)
        
        # 2. Very subtle sparkle (only on very bright areas)
        result = self._subtle_sparkle(result)
        
        return result
    
    def _subtle_highlight_boost(self, img: np.ndarray) -> np.ndarray:
        """Very subtle highlight boost for metallic shine"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Find highlight regions
        _, highlights = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY)
        highlight_mask = highlights.astype(np.float32) / 255.0
        
        # Very subtle boost (1.05 = 5% brighter)
        l_float = l.astype(np.float32)
        l_float = l_float + (l_float * highlight_mask * 0.05)
        l = np.clip(l_float, 0, 255).astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _subtle_sparkle(self, img: np.ndarray) -> np.ndarray:
        """Very subtle sparkle on brightest spots only"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Only very bright spots (threshold 250)
        _, sparkle_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        # Tiny glow
        sparkle_glow = cv2.GaussianBlur(sparkle_mask.astype(np.float32), (3, 3), 0)
        sparkle_glow = sparkle_glow / 255.0 * 10  # Very subtle (10 intensity)
        
        result = img.astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] + sparkle_glow
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _detect_material(self, img: np.ndarray) -> str:
        """
        Detect dominant material type (gold, silver, rose gold, etc.)
        Uses color analysis in LAB and HSV spaces.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Get mean values
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        l_mean = np.mean(lab[:, :, 0])
        a_mean = np.mean(lab[:, :, 1])
        b_mean = np.mean(lab[:, :, 2])
        
        # Gold detection (warm yellow tones)
        if 15 < h_mean < 35 and b_mean > 135:
            return "gold"
        
        # Rose gold detection (pinkish warm)
        if 0 < h_mean < 20 and a_mean > 132:
            return "rose_gold"
        
        # Silver/Platinum detection (cool, desaturated)
        if s_mean < 50 and l_mean > 120:
            return "silver"
        
        # Default to mixed
        return "mixed"
    
    def _enhance_metallic_surfaces(self, img: np.ndarray, material: str) -> np.ndarray:
        """
        Enhance metallic reflections and surface quality.
        Different treatment for different metals.
        """
        result = img.copy()
        
        # Extract luminosity channel
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Find highlight regions (metallic reflections)
        _, highlights = cv2.threshold(l, 180, 255, cv2.THRESH_BINARY)
        
        # Find midtone metallic regions
        midtones = cv2.inRange(l, 100, 180)
        
        # Boost highlights for metallic shine
        boost = self.config.metallic_boost
        l_float = l.astype(np.float32)
        
        # Apply stronger boost to highlights
        highlight_mask = highlights.astype(np.float32) / 255.0
        l_float = l_float + (l_float * highlight_mask * (boost - 1) * 1.5)
        
        # Subtle boost to midtones for consistent metallic look
        midtone_mask = midtones.astype(np.float32) / 255.0
        l_float = l_float + (l_float * midtone_mask * (boost - 1) * 0.3)
        
        l = np.clip(l_float, 0, 255).astype(np.uint8)
        
        # Apply local contrast enhancement for surface detail
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _enhance_gemstones(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance gemstone sparkle and brilliance.
        Adds subtle sparkle effects to bright spots.
        """
        result = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find very bright spots (potential gemstone sparkles)
        threshold = self.config.sparkle_threshold
        _, sparkle_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Create star-like sparkle pattern
        sparkle_kernel = np.zeros((5, 5), dtype=np.float32)
        sparkle_kernel[2, :] = 0.5
        sparkle_kernel[:, 2] = 0.5
        sparkle_kernel[2, 2] = 1.0
        
        # Apply sparkle effect
        sparkle_expanded = cv2.dilate(sparkle_mask, sparkle_kernel.astype(np.uint8), iterations=1)
        sparkle_glow = cv2.GaussianBlur(sparkle_expanded.astype(np.float32), (9, 9), 0)
        sparkle_glow = sparkle_glow / 255.0
        
        # Add sparkle to image
        intensity = self.config.sparkle_intensity
        result_float = result.astype(np.float32)
        
        for i in range(3):
            result_float[:, :, i] = result_float[:, :, i] + sparkle_glow * intensity
        
        # Add color variation to sparkles (slight rainbow effect for diamonds)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        sparkle_colored = sparkle_mask.astype(np.float32) / 255.0
        hsv[:, :, 1] = hsv[:, :, 1] + sparkle_colored * 15  # Slight saturation boost
        
        result_float = np.clip(result_float, 0, 255).astype(np.uint8)
        return result_float
    
    def _enhance_reflections(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance and restore realistic reflections.
        Creates depth and three-dimensionality.
        """
        result = img.copy()
        
        # Create reflection map using gradients
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sobel gradients for edge detection (reflection boundaries)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude of gradients (reflection intensity)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
        
        # Find strong reflections
        _, reflection_mask = cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)
        
        # Subtle enhancement along reflection edges
        reflection_enhance = cv2.GaussianBlur(reflection_mask.astype(np.float32), (5, 5), 0) / 255.0
        
        # Apply selective contrast to reflection areas
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        
        # Boost contrast in reflection areas
        l = l + (l - 128) * reflection_enhance * 0.15
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        lab[:, :, 0] = l
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _material_color_grading(self, img: np.ndarray, material: str) -> np.ndarray:
        """
        Apply color grading specific to jewelry material.
        PRESERVES ORIGINAL COLORS - only subtle enhancements.
        """
        result = img.copy()
        # PRESERVE COLORS - only enhance brightness/contrast, not color shifts
        # Original color preservation - minimal changes
        return result
    
    def _add_luxury_glow(self, img: np.ndarray) -> np.ndarray:
        """
        Add subtle overall glow for luxury feel.
        Creates that 'expensive' look in product photography.
        """
        result = img.copy()
        
        # Create soft glow from bright areas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bright_areas = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Large gaussian blur for glow effect
        glow = cv2.GaussianBlur(bright_areas.astype(np.float32), (31, 31), 0)
        glow = glow / 255.0 * 0.15  # Subtle effect
        
        # Apply glow
        result_float = result.astype(np.float32)
        for i in range(3):
            result_float[:, :, i] = result_float[:, :, i] + glow * 255
        
        return np.clip(result_float, 0, 255).astype(np.uint8)


def enhance_jewelry(img: np.ndarray, config: PipelineConfig = None) -> np.ndarray:
    """Convenience function for jewelry enhancement"""
    return JewelryEnhancer(config).process(img)


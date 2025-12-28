"""
Stage 1: Preprocessing
======================
Prepares degraded images for enhancement:
- Noise analysis and removal
- JPEG compression artifact removal
- Initial color correction
- Contrast normalization
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .config import PipelineConfig


class Preprocessor:
    """Stage 1: Image preprocessing and cleanup"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
    
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline - MINIMAL to preserve quality.
        
        Args:
            img: Input BGR image
            
        Returns:
            Preprocessed BGR image
        """
        result = img.copy()
        
        # MINIMAL PREPROCESSING - only essential cleanup
        # 1. Very light noise removal (only if noisy)
        noise_level = self._estimate_noise(result)
        if noise_level > 5:  # Only if actually noisy
            result = self._light_denoise(result)
        
        # 2. Subtle contrast enhancement (CLAHE with low clip limit)
        result = self._subtle_contrast(result)
        
        return result
    
    def _light_denoise(self, img: np.ndarray) -> np.ndarray:
        """Very light denoising to preserve details"""
        # Use small values to preserve details
        return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    
    def _subtle_contrast(self, img: np.ndarray) -> np.ndarray:
        """Subtle contrast enhancement"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Very low clip limit for subtle effect
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        """Remove noise while preserving edges using Non-local Means"""
        # Estimate noise level
        noise_level = self._estimate_noise(img)
        
        # Adaptive denoising based on noise level
        h = max(3, min(15, int(noise_level * 1.5)))
        
        # Use positional arguments for OpenCV compatibility
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    
    def _estimate_noise(self, img: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = laplacian.var() ** 0.5
        return min(noise / 10, 15)  # Normalize to reasonable range
    
    def _remove_jpeg_artifacts(self, img: np.ndarray) -> np.ndarray:
        """Remove JPEG compression artifacts using bilateral filter"""
        # Bilateral filter preserves edges while smoothing
        return cv2.bilateralFilter(
            img,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )
    
    def _auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """
        Automatic white balance using Gray World assumption.
        Corrects color temperature for accurate jewelry colors.
        """
        result = img.astype(np.float32)
        
        # Calculate mean for each channel
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        # Calculate gray average
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Apply correction
        if avg_b > 0:
            result[:, :, 0] = result[:, :, 0] * (avg_gray / avg_b)
        if avg_g > 0:
            result[:, :, 1] = result[:, :, 1] * (avg_gray / avg_g)
        if avg_r > 0:
            result[:, :, 2] = result[:, :, 2] * (avg_gray / avg_r)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _normalize_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize contrast using CLAHE on L channel.
        Prevents over/under exposure.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size
        )
        
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _remove_color_cast(self, img: np.ndarray) -> np.ndarray:
        """Remove any remaining color cast in LAB space - PRESERVE COLORS"""
        # PRESERVE ORIGINAL COLORS - skip color cast removal
        # Only remove extreme casts (very minimal)
        result = img.copy()
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Only correct extreme color casts (very subtle)
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])
        
        # Very minimal correction (0.1 instead of 0.5) to preserve original colors
        correction_strength = 0.1
        if abs(avg_a - 128) > 20 or abs(avg_b - 128) > 20:  # Only if extreme
            lab[:, :, 1] = lab[:, :, 1] - (avg_a - 128) * correction_strength
            lab[:, :, 2] = lab[:, :, 2] - (avg_b - 128) * correction_strength
        
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(img: np.ndarray, config: PipelineConfig = None) -> np.ndarray:
    """Convenience function for preprocessing"""
    return Preprocessor(config).process(img)


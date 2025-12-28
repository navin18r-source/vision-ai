"""
Utility Functions
=================
Helper functions for the enhancement pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import base64
import io


def load_image(source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load image from various sources.
    
    Args:
        source: File path, numpy array, or PIL Image
        
    Returns:
        BGR numpy array
    """
    if isinstance(source, np.ndarray):
        return source
    elif isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Failed to load image: {source}")
        return img
    elif isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError(f"Unsupported image type: {type(source)}")


def save_image(img: np.ndarray, path: Union[str, Path], quality: int = 95) -> None:
    """
    Save image to file.
    
    Args:
        img: BGR numpy array
        path: Output path
        quality: JPEG quality (1-100)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif path.suffix.lower() == '.png':
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        cv2.imwrite(str(path), img)


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to RGB PIL Image"""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert RGB PIL Image to BGR numpy array"""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def numpy_to_base64(img: np.ndarray, format: str = 'PNG') -> str:
    """
    Convert numpy array to base64 string.
    
    Args:
        img: BGR numpy array
        format: Image format ('PNG' or 'JPEG')
        
    Returns:
        Base64 encoded string
    """
    # Convert BGR to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_numpy(b64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy array.
    
    Args:
        b64_string: Base64 encoded image string
        
    Returns:
        BGR numpy array
    """
    # Handle data URL prefix
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    
    img_data = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_image(img: np.ndarray, 
                 size: Tuple[int, int],
                 keep_aspect: bool = True) -> np.ndarray:
    """
    Resize image with optional aspect ratio preservation.
    
    Args:
        img: Input image
        size: Target size (width, height)
        keep_aspect: Whether to preserve aspect ratio
        
    Returns:
        Resized image
    """
    if not keep_aspect:
        return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    
    h, w = img.shape[:2]
    target_w, target_h = size
    
    # Calculate scale to fit within target while preserving aspect
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create canvas and center image
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index between two images"""
    try:
        from skimage.metrics import structural_similarity
        
        # Convert to grayscale if color
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        return structural_similarity(img1, img2)
    except ImportError:
        # Simplified SSIM implementation
        c1 = 6.5025
        c2 = 58.5225
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return float(np.mean(ssim_map))


"""
Configuration Parameters for Jewelry Enhancement Pipeline
=========================================================
Central configuration for all pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from pathlib import Path


@dataclass
class PipelineConfig:
    """Main configuration class for the enhancement pipeline"""
    
    # Device settings
    device: str = "auto"  # 'cuda', 'mps', 'cpu', or 'auto'
    
    # Output settings - Higher resolution for zoom support
    output_size: Tuple[int, int] = (2048, 2048)  # High-res for zoom without quality loss
    background_color: Tuple[int, int, int] = (255, 255, 255)  # RGB white
    
    # Stage 1: Preprocessing
    denoise_strength: int = 10
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    clahe_clip_limit: float = 2.5
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Stage 2: Super-Resolution
    sr_model: str = "realesrgan"  # 'realesrgan', 'swinir', or 'opencv'
    sr_scale: int = 4  # 2 or 4
    sr_tile_size: int = 400
    
    # Stage 3: Detail Enhancement (Generative)
    use_generative: bool = True
    sd_model: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model: str = "lllyasviel/control_v11f1e_sd15_tile"
    gen_strength: float = 0.25  # Lower = preserve more original
    gen_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_scale: float = 0.8
    
    # Stage 4: Jewelry Enhancement (KEY DIFFERENTIATOR)
    metallic_boost: float = 1.3
    sparkle_threshold: int = 240
    sparkle_intensity: float = 35.0
    highlight_enhance: float = 1.2
    gold_warmth: float = 1.1
    silver_coolness: float = 1.05
    
    # Stage 5: Post-processing
    sharpen_amount: float = 0.6
    saturation_boost: float = 1.1
    remove_background: bool = True
    
    # Paths
    weights_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "weights")
    
    # Jewelry-specific prompts
    jewelry_prompt: str = (
        "professional product photography of luxury jewelry, "
        "studio lighting, extremely detailed, sharp focus, "
        "sparkling gemstones, brilliant reflections, "
        "clean white background, 8k resolution, commercial quality, "
        "perfect metallic surface, high-end jewelry catalog"
    )
    
    negative_prompt: str = (
        "blurry, low quality, distorted, deformed, noisy, grainy, "
        "artifacts, watermark, text, ugly, pixelated, dull, matte, "
        "compression artifacts, jpeg artifacts, oversaturated"
    )


@dataclass
class UltraFastConfig(PipelineConfig):
    """Ultra-fast configuration for <15 second processing"""
    sr_scale: int = 2
    use_generative: bool = False  # Skip - too slow for 15s target
    gen_steps: int = 5
    # Optimized preprocessing for speed
    denoise_strength: int = 5  # Reduced for speed
    bilateral_d: int = 5  # Smaller kernel
    clahe_clip_limit: float = 2.0  # Faster processing
    # Faster post-processing
    sharpen_amount: float = 0.4  # Less intensive
    remove_background: bool = True  # Keep for quality


@dataclass
class QuickConfig(PipelineConfig):
    """Fast processing configuration"""
    sr_scale: int = 2
    use_generative: bool = False
    gen_steps: int = 10


@dataclass 
class BalancedConfig(PipelineConfig):
    """Balanced quality/speed configuration"""
    sr_scale: int = 2
    use_generative: bool = True  # Enabled - will gracefully fallback if models unavailable
    gen_strength: float = 0.2
    gen_steps: int = 15


@dataclass
class MaxQualityConfig(PipelineConfig):
    """Maximum quality configuration"""
    sr_scale: int = 4
    use_generative: bool = True
    gen_strength: float = 0.3
    gen_steps: int = 25
    metallic_boost: float = 1.4
    sparkle_intensity: float = 45.0


@dataclass
class ProductionConfig(PipelineConfig):
    """Production configuration optimized for GCP VM with CUDA GPU - All 5 stages enabled"""
    sr_scale: int = 4  # 4x for maximum quality (object-only, so faster)
    use_generative: bool = True  # Enable generative enhancement
    gen_strength: float = 0.3  # Higher strength for better quality
    gen_steps: int = 25  # More steps for better quality
    guidance_scale: float = 8.0  # Higher guidance for better results
    # Optimized for CUDA GPU
    device: str = "auto"  # Will auto-detect CUDA
    # Full quality settings
    metallic_boost: float = 1.4  # Higher boost
    sparkle_intensity: float = 45.0  # More sparkle
    sharpen_amount: float = 0.8  # More sharpening
    saturation_boost: float = 1.15  # More vibrant
    remove_background: bool = True


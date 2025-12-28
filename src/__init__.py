"""
VISION - Jewelry Image Enhancement Pipeline
============================================
AI-powered enhancement for jewelry photography.

Usage:
    from src import EnhancementPipeline
    
    pipeline = EnhancementPipeline(mode="balanced")
    enhanced = pipeline.enhance("input.jpg")
"""

from .config import PipelineConfig, QuickConfig, BalancedConfig, MaxQualityConfig
from .enhancement_pipeline import EnhancementPipeline, create_pipeline, enhance_image
from .preprocessing import Preprocessor, preprocess
from .super_resolution import SuperResolver, super_resolve
from .detail_enhancement import DetailEnhancer, enhance_details
from .jewelry_enhancement import JewelryEnhancer, enhance_jewelry
from .postprocessing import PostProcessor, postprocess
from .helpers import (
    load_image, save_image,
    numpy_to_pil, pil_to_numpy,
    numpy_to_base64, base64_to_numpy,
    resize_image,
    calculate_psnr, calculate_ssim
)

__version__ = "1.0.0"
__all__ = [
    # Main pipeline
    "EnhancementPipeline",
    "create_pipeline",
    "enhance_image",
    
    # Configuration
    "PipelineConfig",
    "QuickConfig",
    "BalancedConfig",
    "MaxQualityConfig",
    
    # Individual stages
    "Preprocessor",
    "SuperResolver",
    "DetailEnhancer",
    "JewelryEnhancer",
    "PostProcessor",
    
    # Stage functions
    "preprocess",
    "super_resolve",
    "enhance_details",
    "enhance_jewelry",
    "postprocess",
    
    # Utilities
    "load_image",
    "save_image",
    "numpy_to_pil",
    "pil_to_numpy",
    "numpy_to_base64",
    "base64_to_numpy",
    "resize_image",
    "calculate_psnr",
    "calculate_ssim",
]

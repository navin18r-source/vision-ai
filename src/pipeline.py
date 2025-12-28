"""
Complete Enhancement Pipeline Orchestrator
==========================================
Orchestrates all 5 stages of the jewelry enhancement pipeline:

┌─────────────────────────────────────┐
│ LOW-QUALITY JEWELRY IMAGE           │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ STAGE 1: PREPROCESSING              │
│ - Noise Analysis & Removal          │
│ - Compression Artifact Removal      │
│ - Initial Color Correction          │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ STAGE 2: SUPER-RESOLUTION           │
│ - Real-ESRGAN for 2x/4x upscaling   │
│ - AI-powered detail recovery        │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ STAGE 3: DETAIL ENHANCEMENT         │
│ - Stable Diffusion img2img          │
│ - ControlNet for structure          │
│ - Jewelry-specific prompting        │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ STAGE 4: JEWELRY ENHANCEMENT (KEY)  │
│ - Metallic surface enhancement      │
│ - Gemstone sparkle restoration      │
│ - Material-aware color grading      │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ STAGE 5: POST-PROCESSING            │
│ - Background removal                │
│ - Final sharpening & color grade    │
│ - Output formatting                 │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ HIGH-QUALITY RETAIL-READY IMAGE     │
└─────────────────────────────────────┘
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Literal
import time

from .config import PipelineConfig, QuickConfig, BalancedConfig, MaxQualityConfig
from .preprocessing import Preprocessor
from .super_resolution import SuperResolver
from .detail_enhancement import DetailEnhancer
from .jewelry_enhancement import JewelryEnhancer
from .postprocessing import PostProcessor


class EnhancementPipeline:
    """
    Complete 5-Stage Jewelry Image Enhancement Pipeline.
    
    This is the main orchestrator that combines all stages into
    a seamless enhancement workflow.
    """
    
    def __init__(self, 
                 config: PipelineConfig = None,
                 mode: Literal["quick", "balanced", "max"] = "balanced"):
        """
        Initialize the enhancement pipeline.
        
        Args:
            config: Custom configuration (overrides mode)
            mode: Preset mode - "quick", "balanced", or "max"
        """
        # Select configuration
        if config is not None:
            self.config = config
        elif mode == "quick":
            self.config = QuickConfig()
        elif mode == "max":
            self.config = MaxQualityConfig()
        else:
            self.config = BalancedConfig()
        
        print("\n" + "=" * 60)
        print("💎 JEWELRY ENHANCEMENT PIPELINE")
        print("=" * 60)
        print(f"Mode: {mode.upper()}")
        print(f"Output Size: {self.config.output_size}")
        print(f"Super-Resolution: {self.config.sr_scale}x")
        print(f"Generative Enhancement: {'Enabled' if self.config.use_generative else 'Disabled'}")
        print("=" * 60)
        print("\n🔧 Initializing pipeline stages...")
        
        # Initialize all stages
        self.preprocessor = Preprocessor(self.config)
        print("  ✓ Stage 1: Preprocessing ready")
        
        self.super_resolver = SuperResolver(self.config)
        # Status printed in SuperResolver
        
        self.detail_enhancer = DetailEnhancer(self.config)
        # Status printed in DetailEnhancer if enabled
        
        self.jewelry_enhancer = JewelryEnhancer(self.config)
        print("  ✓ Stage 4: Jewelry Enhancement ready")
        
        self.postprocessor = PostProcessor(self.config)
        # Status printed in PostProcessor
        
        print("\n✅ Pipeline ready!\n")
    
    def enhance(self,
                img: Union[np.ndarray, str, Path, Image.Image],
                use_sr: bool = True,
                use_generative: bool = None,
                use_jewelry: bool = True,
                remove_background: bool = None,
                verbose: bool = True) -> np.ndarray:
        """
        Run the full enhancement pipeline.
        
        Args:
            img: Input image (path, numpy array, or PIL Image)
            use_sr: Enable Stage 2 (Super-Resolution)
            use_generative: Enable Stage 3 (Generative Enhancement)
            use_jewelry: Enable Stage 4 (Jewelry Enhancement)
            remove_background: Enable background removal in Stage 5
            verbose: Print progress messages
            
        Returns:
            Enhanced image as BGR numpy array
        """
        if use_generative is None:
            use_generative = self.config.use_generative
        if remove_background is None:
            remove_background = self.config.remove_background
        
        # Load image
        img_array = self._load_image(img)
        
        if verbose:
            print("\n" + "=" * 50)
            print("🚀 Starting Enhancement Pipeline")
            print(f"   Input: {img_array.shape[1]}x{img_array.shape[0]}")
            print("=" * 50)
        
        start_time = time.time()
        result = img_array.copy()
        
        # Stage 1: Preprocessing
        if verbose:
            print("\n📌 Stage 1: Preprocessing...")
        stage_start = time.time()
        result = self.preprocessor.process(result)
        if verbose:
            print(f"   ✓ Done ({time.time() - stage_start:.2f}s)")
        
        # Stage 2: Super-Resolution
        if use_sr:
            if verbose:
                print(f"\n📌 Stage 2: Super-Resolution ({self.config.sr_scale}x)...")
            stage_start = time.time()
            result = self.super_resolver.process(result, scale=self.config.sr_scale)
            if verbose:
                print(f"   ✓ Done ({time.time() - stage_start:.2f}s)")
                print(f"   → Size: {result.shape[1]}x{result.shape[0]}")
        
        # Stage 3: Detail Enhancement (Generative)
        if use_generative and self.detail_enhancer.is_available:
            if verbose:
                print(f"\n📌 Stage 3: Generative Detail Enhancement...")
            stage_start = time.time()
            result = self.detail_enhancer.process(result)
            if verbose:
                print(f"   ✓ Done ({time.time() - stage_start:.2f}s)")
        elif use_generative and verbose:
            print("\n📌 Stage 3: Skipped (SD not available)")
        
        # Stage 4: Jewelry Enhancement (KEY DIFFERENTIATOR)
        if use_jewelry:
            if verbose:
                print("\n📌 Stage 4: Jewelry Enhancement (KEY)...")
            stage_start = time.time()
            result = self.jewelry_enhancer.process(result)
            if verbose:
                print(f"   ✓ Done ({time.time() - stage_start:.2f}s)")
        
        # Stage 5: Post-Processing
        if verbose:
            print("\n📌 Stage 5: Post-Processing...")
        stage_start = time.time()
        result = self.postprocessor.process(result, remove_background=remove_background)
        if verbose:
            print(f"   ✓ Done ({time.time() - stage_start:.2f}s)")
        
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 50)
            print("✅ Enhancement Complete!")
            print(f"   Output: {result.shape[1]}x{result.shape[0]}")
            print(f"   Total time: {total_time:.2f}s")
            print("=" * 50 + "\n")
        
        return result
    
    def _load_image(self, img: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
        """Load image from various input types"""
        if isinstance(img, np.ndarray):
            return img
        elif isinstance(img, (str, Path)):
            loaded = cv2.imread(str(img))
            if loaded is None:
                raise ValueError(f"Failed to load image: {img}")
            return loaded
        elif isinstance(img, Image.Image):
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
    
    def enhance_quick(self, img: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
        """Quick enhancement (no generative, 2x SR)"""
        return self.enhance(
            img, 
            use_sr=True, 
            use_generative=False,
            use_jewelry=True,
            remove_background=True
        )
    
    def enhance_balanced(self, img: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
        """Balanced enhancement (all stages)"""
        return self.enhance(
            img,
            use_sr=True,
            use_generative=True,
            use_jewelry=True,
            remove_background=True
        )
    
    def enhance_max(self, img: Union[np.ndarray, str, Path, Image.Image]) -> np.ndarray:
        """Maximum quality enhancement"""
        return self.enhance(
            img,
            use_sr=True,
            use_generative=True,
            use_jewelry=True,
            remove_background=True
        )
    
    @property
    def capabilities(self) -> dict:
        """Return current pipeline capabilities"""
        return {
            "super_resolution": self.super_resolver.is_ai_available,
            "generative_enhancement": self.detail_enhancer.is_available,
            "background_removal": self.postprocessor.can_remove_background
        }


# Convenience functions
def create_pipeline(mode: str = "balanced") -> EnhancementPipeline:
    """Create an enhancement pipeline with the specified mode"""
    return EnhancementPipeline(mode=mode)


def enhance_image(img: Union[np.ndarray, str, Path, Image.Image],
                  mode: str = "balanced") -> np.ndarray:
    """One-shot image enhancement"""
    pipeline = EnhancementPipeline(mode=mode)
    return pipeline.enhance(img)


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jewelry Image Enhancement Pipeline")
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("--output", "-o", type=str, default="enhanced_output.png",
                       help="Output image path")
    parser.add_argument("--mode", "-m", type=str, default="balanced",
                       choices=["quick", "balanced", "max"],
                       help="Enhancement mode")
    parser.add_argument("--no-sr", action="store_true", help="Skip super-resolution")
    parser.add_argument("--no-gen", action="store_true", help="Skip generative enhancement")
    parser.add_argument("--no-jewelry", action="store_true", help="Skip jewelry enhancement")
    parser.add_argument("--keep-bg", action="store_true", help="Keep original background")
    
    args = parser.parse_args()
    
    pipeline = EnhancementPipeline(mode=args.mode)
    result = pipeline.enhance(
        args.input,
        use_sr=not args.no_sr,
        use_generative=not args.no_gen,
        use_jewelry=not args.no_jewelry,
        remove_background=not args.keep_bg
    )
    
    cv2.imwrite(args.output, result)
    print(f"Saved to: {args.output}")


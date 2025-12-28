"""
FULL Jewelry Image Enhancement Pipeline - 4 Stages
===================================================
Complete implementation as per the architecture:

STAGE 1: PREPROCESSING
    - Noise Analysis & Removal
    - Compression Artifact Removal  
    - Initial Color Correction

STAGE 2: SUPER-RESOLUTION
    - Real-ESRGAN for base upscaling (4x)
    - High-quality detail recovery

STAGE 3: GENERATIVE REFINEMENT
    - Stable Diffusion img2img
    - ControlNet Tile for structure preservation
    - Jewelry-specific prompting

STAGE 4: POST-PROCESSING
    - Material-specific enhancement (metallic reflections)
    - Sparkle/reflection restoration
    - Background removal & standardization
    - Color grading for luxury feel
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
import torch
import os


class FullJewelryEnhancer:
    """
    Complete 4-Stage Enhancement Pipeline for Jewelry Images.
    
    Stage 1: Preprocessing (OpenCV)
    Stage 2: Super-Resolution (Real-ESRGAN)
    Stage 3: Generative Refinement (Stable Diffusion + ControlNet)
    Stage 4: Post-Processing (Background removal + Polish)
    """
    
    def __init__(self, 
                 device: str = "auto",
                 output_size: Tuple[int, int] = (512, 512),
                 background_color: Tuple[int, int, int] = (255, 255, 255),
                 use_generative: bool = True):
        """
        Initialize the full enhancement pipeline.
        
        Args:
            device: 'cuda', 'mps', 'cpu', or 'auto'
            output_size: Final output dimensions
            background_color: RGB background color
            use_generative: Whether to use SD for generative refinement
        """
        self.output_size = output_size
        self.background_color = background_color
        self.use_generative = use_generative
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"\n{'='*60}")
        print("💎 FULL JEWELRY ENHANCEMENT PIPELINE")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Output Size: {output_size}")
        print(f"Generative Refinement: {'Enabled' if use_generative else 'Disabled'}")
        print(f"{'='*60}\n")
        
        self._init_models()
    
    def _init_models(self):
        """Initialize all AI models"""
        print("🔧 Loading models...")
        
        # Stage 2: Real-ESRGAN
        self.esrgan_model = None
        try:
            self._init_realesrgan()
            print("  ✓ Real-ESRGAN loaded")
        except Exception as e:
            print(f"  ⚠ Real-ESRGAN not available: {e}")
        
        # Stage 3: Stable Diffusion + ControlNet
        self.sd_pipe = None
        if self.use_generative:
            try:
                self._init_stable_diffusion()
                print("  ✓ Stable Diffusion + ControlNet loaded")
            except Exception as e:
                print(f"  ⚠ Stable Diffusion not available: {e}")
        
        # Stage 4: Background Removal
        self.rembg_session = None
        try:
            from rembg import new_session
            self.rembg_session = new_session("u2net")
            print("  ✓ Background removal model loaded")
        except Exception as e:
            print(f"  ⚠ Background removal not available: {e}")
        
        print("\n✅ Pipeline ready!\n")
    
    def _init_realesrgan(self):
        """Initialize Real-ESRGAN for 4x super-resolution"""
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # Real-ESRGAN x4 model
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=4
        )
        
        # Model weights path
        weights_dir = Path(__file__).parent.parent.parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        weights_path = weights_dir / "RealESRGAN_x4plus.pth"
        
        # Download weights if not present
        if not weights_path.exists():
            print("  📥 Downloading Real-ESRGAN weights...")
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            urllib.request.urlretrieve(url, str(weights_path))
        
        self.esrgan_model = RealESRGANer(
            scale=4,
            model_path=str(weights_path),
            model=model,
            tile=400,  # Tile size for memory efficiency
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == "cuda" else False,
            device=self.device
        )
    
    def _init_stable_diffusion(self):
        """Initialize Stable Diffusion with ControlNet Tile for enhancement"""
        from diffusers import (
            StableDiffusionControlNetImg2ImgPipeline,
            ControlNetModel,
            AutoencoderKL
        )
        
        # Load ControlNet Tile model (best for image enhancement)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1e_sd15_tile",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Load SD 1.5 with better VAE
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        self.sd_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        if self.device == "cuda":
            self.sd_pipe = self.sd_pipe.to("cuda")
            self.sd_pipe.enable_attention_slicing()
        elif self.device == "mps":
            self.sd_pipe = self.sd_pipe.to("mps")
        
        # Jewelry-specific prompts
        self.jewelry_prompt = (
            "professional product photography of luxury jewelry, "
            "studio lighting, high detail, sharp focus, sparkling, "
            "clean background, 8k resolution, commercial quality, "
            "metallic reflections, gemstone brilliance"
        )
        
        self.negative_prompt = (
            "blurry, low quality, distorted, deformed, noisy, grainy, "
            "artifacts, watermark, text, ugly, pixelated, "
            "compression artifacts, jpeg artifacts"
        )
    
    # =========================================================================
    #                    STAGE 1: PREPROCESSING
    # =========================================================================
    
    def stage1_preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Stage 1: Preprocessing
        - Noise analysis and removal
        - JPEG artifact removal
        - Initial color correction
        """
        print("  📌 Stage 1: Preprocessing...")
        
        # 1.1 Remove JPEG compression artifacts
        result = cv2.bilateralFilter(img, 9, 75, 75)
        
        # 1.2 Denoise while preserving edges
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        
        # 1.3 Auto white balance for color correction
        result = self._auto_white_balance(result)
        
        # 1.4 Initial contrast enhancement
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Automatic white balance correction"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # =========================================================================
    #                    STAGE 2: SUPER-RESOLUTION
    # =========================================================================
    
    def stage2_super_resolution(self, img: np.ndarray, scale: int = 4) -> np.ndarray:
        """
        Stage 2: Super-Resolution using Real-ESRGAN
        - 4x upscaling with detail preservation
        - AI-powered texture enhancement
        """
        print(f"  📌 Stage 2: Super-Resolution ({scale}x)...")
        
        if self.esrgan_model is not None:
            # Use Real-ESRGAN
            output, _ = self.esrgan_model.enhance(img, outscale=scale)
            return output
        else:
            # Fallback: High-quality interpolation
            print("    ⚠ Using fallback upscaling (Real-ESRGAN not available)")
            h, w = img.shape[:2]
            return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
    
    # =========================================================================
    #                    STAGE 3: GENERATIVE REFINEMENT
    # =========================================================================
    
    def stage3_generative_refinement(self, 
                                      img: np.ndarray, 
                                      strength: float = 0.3,
                                      steps: int = 20) -> np.ndarray:
        """
        Stage 3: Generative Refinement using Stable Diffusion + ControlNet
        - Adds realistic details that upscaling can't create
        - Preserves structure using ControlNet Tile
        - Jewelry-specific prompting for optimal results
        """
        print(f"  📌 Stage 3: Generative Refinement (strength={strength})...")
        
        if self.sd_pipe is None:
            print("    ⚠ Skipping (Stable Diffusion not available)")
            return img
        
        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Resize to SD-friendly size (multiple of 8)
        w, h = pil_img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Run SD with ControlNet
        with torch.inference_mode():
            result = self.sd_pipe(
                prompt=self.jewelry_prompt,
                negative_prompt=self.negative_prompt,
                image=pil_img,
                control_image=pil_img,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.8
            ).images[0]
        
        # Convert back to BGR numpy
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        
        # Resize back to original size if needed
        if result_bgr.shape[:2] != img.shape[:2]:
            result_bgr = cv2.resize(result_bgr, (img.shape[1], img.shape[0]), 
                                    interpolation=cv2.INTER_LANCZOS4)
        
        return result_bgr
    
    # =========================================================================
    #                    STAGE 4: POST-PROCESSING
    # =========================================================================
    
    def stage4_postprocess(self, 
                           img: np.ndarray,
                           remove_background: bool = True) -> np.ndarray:
        """
        Stage 4: Post-Processing
        - Material-specific enhancement (metallic reflections)
        - Sparkle/reflection restoration
        - Background removal & standardization
        - Color grading for luxury feel
        """
        print("  📌 Stage 4: Post-Processing...")
        
        result = img.copy()
        
        # 4.1 Enhance metallic reflections
        result = self._enhance_metallic(result)
        
        # 4.2 Add sparkle effect for gemstones
        result = self._add_sparkle(result)
        
        # 4.3 Sharpen details
        result = self._sharpen(result, amount=0.5)
        
        # 4.4 Color grading for luxury feel
        result = self._color_grade_luxury(result)
        
        # 4.5 Background removal and standardization
        if remove_background and self.rembg_session is not None:
            result = self._remove_and_replace_background(result)
        
        # 4.6 Final resize to output size
        result = cv2.resize(result, self.output_size, interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    def _enhance_metallic(self, img: np.ndarray, strength: float = 1.2) -> np.ndarray:
        """Enhance metallic reflections for jewelry"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, highlights = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        highlight_mask = highlights.astype(np.float32) / 255.0
        l = l.astype(np.float32)
        l = l + (l * highlight_mask * (strength - 1))
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _add_sparkle(self, img: np.ndarray, threshold: int = 245) -> np.ndarray:
        """Add subtle sparkle effect for gemstones"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, sparkles = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        glow = cv2.GaussianBlur(sparkles, (7, 7), 0)
        glow = glow.astype(np.float32) / 255.0
        
        result = img.astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] + glow * 30
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _sharpen(self, img: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Unsharp mask sharpening"""
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1 + amount, gaussian, -amount, 0)
    
    def _color_grade_luxury(self, img: np.ndarray) -> np.ndarray:
        """Apply subtle color grading for luxury feel"""
        # Slight warmth and saturation boost
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.1  # Saturation +10%
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _remove_and_replace_background(self, img: np.ndarray) -> np.ndarray:
        """Remove background and replace with plain color"""
        from rembg import remove
        
        # Remove background
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        output = remove(pil_img, session=self.rembg_session)
        output_np = np.array(output)
        
        # Create plain background
        h, w = output_np.shape[:2]
        background = np.full((h, w, 3), self.background_color[::-1], dtype=np.uint8)
        
        # Blend with alpha
        if output_np.shape[2] == 4:
            alpha = output_np[:, :, 3:4] / 255.0
            fg = cv2.cvtColor(output_np[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            fg = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        result = (fg * alpha + background * (1 - alpha)).astype(np.uint8)
        return result
    
    # =========================================================================
    #                    MAIN ENHANCEMENT FUNCTION
    # =========================================================================
    
    def enhance(self,
                img: Union[np.ndarray, str, Image.Image],
                use_sr: bool = True,
                use_generative: bool = None,
                remove_background: bool = True,
                sr_scale: int = 4,
                gen_strength: float = 0.25) -> np.ndarray:
        """
        Full 4-Stage Enhancement Pipeline.
        
        Args:
            img: Input image (path, numpy array, or PIL Image)
            use_sr: Use Stage 2 (Super-Resolution)
            use_generative: Use Stage 3 (Generative Refinement)
            remove_background: Use background removal in Stage 4
            sr_scale: Super-resolution scale factor (2 or 4)
            gen_strength: Generative refinement strength (0.1-0.4)
        
        Returns:
            Enhanced image as numpy array (BGR)
        """
        if use_generative is None:
            use_generative = self.use_generative
        
        # Load image
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if img is None:
            raise ValueError("Failed to load image")
        
        print(f"\n{'='*50}")
        print(f"🚀 Starting Enhancement Pipeline")
        print(f"   Input: {img.shape[1]}x{img.shape[0]}")
        print(f"{'='*50}")
        
        # Stage 1: Preprocessing
        result = self.stage1_preprocess(img)
        
        # Stage 2: Super-Resolution
        if use_sr:
            result = self.stage2_super_resolution(result, scale=sr_scale)
        
        # Stage 3: Generative Refinement
        if use_generative and self.sd_pipe is not None:
            result = self.stage3_generative_refinement(result, strength=gen_strength)
        
        # Stage 4: Post-Processing
        result = self.stage4_postprocess(result, remove_background=remove_background)
        
        print(f"{'='*50}")
        print(f"✅ Enhancement Complete!")
        print(f"   Output: {result.shape[1]}x{result.shape[0]}")
        print(f"{'='*50}\n")
        
        return result
    
    def enhance_quick(self, img: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """Quick enhancement (no generative, 2x SR)"""
        return self.enhance(img, use_sr=True, use_generative=False, 
                           sr_scale=2, remove_background=True)
    
    def enhance_full(self, img: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """Full enhancement (all stages, 4x SR)"""
        return self.enhance(img, use_sr=True, use_generative=True,
                           sr_scale=4, remove_background=True)


# ============================================================================
#                              CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Jewelry Enhancement Pipeline")
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, default="enhanced_output.png",
                       help="Output image path")
    parser.add_argument("--no-sr", action="store_true", help="Skip super-resolution")
    parser.add_argument("--no-gen", action="store_true", help="Skip generative refinement")
    parser.add_argument("--no-bg", action="store_true", help="Keep original background")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4],
                       help="Super-resolution scale")
    
    args = parser.parse_args()
    
    enhancer = FullJewelryEnhancer(use_generative=not args.no_gen)
    result = enhancer.enhance(
        args.input,
        use_sr=not args.no_sr,
        use_generative=not args.no_gen,
        remove_background=not args.no_bg,
        sr_scale=args.scale
    )
    
    cv2.imwrite(args.output, result)
    print(f"Saved to: {args.output}")


"""
Stage 3: Detail Enhancement (Generative Refinement)
====================================================
Uses AI to add realistic details that upscaling can't create:
- Stable Diffusion img2img for detail generation
- ControlNet Tile for structure preservation
- Jewelry-specific prompting for optimal results
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional
import torch

from .config import PipelineConfig


class DetailEnhancer:
    """Stage 3: Generative detail enhancement using Stable Diffusion"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.pipe = None
        self.device = self._get_device()
        
        # Only initialize SD if explicitly enabled (requires large model download)
        if self.config.use_generative:
            try:
                self._init_model()
            except ImportError as e:
                print(f"  ⚠ Generative enhancement: Missing dependencies ({e})")
                print("  → Install: pip install diffusers transformers accelerate")
                self.pipe = None
            except Exception as e:
                print(f"  ⚠ Generative enhancement unavailable: {type(e).__name__}")
                print("  → Will use other enhancement stages")
                self.pipe = None
    
    def _get_device(self) -> str:
        """Auto-detect best available device"""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _init_model(self):
        """Initialize Stable Diffusion with ControlNet Tile"""
        try:
            from diffusers import (
                StableDiffusionControlNetImg2ImgPipeline,
                ControlNetModel,
                AutoencoderKL
            )
            
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # Load ControlNet Tile (best for image enhancement/upscaling)
            print("  📥 Loading ControlNet Tile...")
            controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model,
                torch_dtype=dtype
            )
            
            # Load improved VAE for better quality
            print("  📥 Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=dtype
            )
            
            # Load SD pipeline
            print("  📥 Loading Stable Diffusion...")
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.config.sd_model,
                controlnet=controlnet,
                vae=vae,
                torch_dtype=dtype,
                safety_checker=None
            )
            
            # Move to device and optimize
            if self.device == "cuda":
                self.pipe = self.pipe.to("cuda")
                # Optimize for CUDA (GCP VM)
                self.pipe.enable_attention_slicing()
                # Enable memory efficient attention if available (faster on CUDA)
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("  ✓ XFormers memory efficient attention enabled")
                except:
                    pass
                # Enable VAE slicing for faster processing (new API)
                try:
                    self.pipe.vae.enable_slicing()
                except:
                    # Fallback to old API if new one not available
                    try:
                        self.pipe.enable_vae_slicing()
                    except:
                        pass
            elif self.device == "mps":
                # Enable attention slicing for MPS to reduce memory usage
                self.pipe.enable_attention_slicing(slice_size="max")
                # Move to MPS after optimizations
                self.pipe = self.pipe.to("mps")
            
            print(f"  ✓ Stable Diffusion + ControlNet loaded on {self.device}")
            
        except ImportError as e:
            print(f"  ⚠ Diffusers not available: {e}")
            self.pipe = None
        except Exception as e:
            print(f"  ⚠ Failed to load SD pipeline: {e}")
            self.pipe = None
    
    def process(self, 
                img: np.ndarray,
                strength: float = None,
                steps: int = None) -> np.ndarray:
        """
        Enhance image details using Stable Diffusion.
        
        Args:
            img: Input BGR image
            strength: How much to modify (0.1-0.4, lower = preserve more)
            steps: Number of diffusion steps
            
        Returns:
            Enhanced BGR image
        """
        if self.pipe is None:
            return img
        
        if strength is None:
            strength = self.config.gen_strength
        if steps is None:
            steps = self.config.gen_steps
        
        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Resize to SD-friendly size (multiple of 8)
        w, h = pil_img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        
        # Cap size for memory (smaller for MPS to avoid OOM)
        max_dim = 512 if self.device == "mps" else 768
        if new_w > max_dim or new_h > max_dim:
            ratio = min(max_dim / new_w, max_dim / new_h)
            new_w = int(new_w * ratio) // 8 * 8
            new_h = int(new_h * ratio) // 8 * 8
        
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Clear MPS cache before processing
        if self.device == "mps":
            torch.mps.empty_cache()
        
        # Run Stable Diffusion with ControlNet
        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=self.config.jewelry_prompt,
                    negative_prompt=self.config.negative_prompt,
                    image=pil_img,
                    control_image=pil_img,
                    num_inference_steps=steps,
                    strength=strength,
                    guidance_scale=self.config.guidance_scale,
                    controlnet_conditioning_scale=self.config.controlnet_scale
                ).images[0]
            
            # Clear MPS cache after processing
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Convert back to BGR numpy
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # Resize back to original size if needed
            if result_bgr.shape[:2] != img.shape[:2]:
                result_bgr = cv2.resize(
                    result_bgr, 
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            return result_bgr
            
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
                print(f"  ⚠ MPS out of memory: {e}")
                print("  → Falling back to CPU for generative enhancement...")
                # Try CPU fallback
                try:
                    original_device = self.device
                    self.device = "cpu"
                    if self.pipe is not None:
                        self.pipe = self.pipe.to("cpu")
                    
                    # Further reduce size for CPU
                    max_dim_cpu = 384
                    if new_w > max_dim_cpu or new_h > max_dim_cpu:
                        ratio = min(max_dim_cpu / new_w, max_dim_cpu / new_h)
                        new_w = int(new_w * ratio) // 8 * 8
                        new_h = int(new_h * ratio) // 8 * 8
                        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    
                    with torch.inference_mode():
                        result = self.pipe(
                            prompt=self.config.jewelry_prompt,
                            negative_prompt=self.config.negative_prompt,
                            image=pil_img,
                            control_image=pil_img,
                            num_inference_steps=max(10, steps // 2),  # Fewer steps on CPU
                            strength=strength,
                            guidance_scale=self.config.guidance_scale,
                            controlnet_conditioning_scale=self.config.controlnet_scale
                        ).images[0]
                    
                    # Convert back to BGR numpy
                    result_np = np.array(result)
                    result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                    
                    # Resize back to original size if needed
                    if result_bgr.shape[:2] != img.shape[:2]:
                        result_bgr = cv2.resize(
                            result_bgr, 
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LANCZOS4
                        )
                    
                    # Restore original device
                    self.device = original_device
                    if self.pipe is not None and original_device != "cpu":
                        self.pipe = self.pipe.to(original_device)
                    
                    print("  ✓ CPU fallback successful")
                    return result_bgr
                except Exception as cpu_error:
                    print(f"  ⚠ CPU fallback also failed: {cpu_error}")
                    return img
            else:
                print(f"  ⚠ Generative enhancement failed: {e}")
                return img
        except Exception as e:
            print(f"  ⚠ Generative enhancement failed: {e}")
            return img
    
    @property
    def is_available(self) -> bool:
        """Check if generative enhancement is available"""
        return self.pipe is not None


def enhance_details(img: np.ndarray, 
                    strength: float = 0.25,
                    config: PipelineConfig = None) -> np.ndarray:
    """Convenience function for detail enhancement"""
    return DetailEnhancer(config).process(img, strength)


"""
Stage 2: Super-Resolution
=========================
AI-powered upscaling for maximum detail recovery:
- Real-ESRGAN for photorealistic upscaling
- SwinIR alternative for different characteristics
- Fallback to high-quality interpolation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
import urllib.request
import os
import warnings

from .config import PipelineConfig


class SuperResolver:
    """Stage 2: AI Super-Resolution using Real-ESRGAN"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.device = self._get_device()
        self._init_model()
    
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
        """Initialize Real-ESRGAN model"""
        try:
            # Check NumPy version compatibility first
            import numpy as np
            import warnings
            
            if hasattr(np, '__version__') and int(np.__version__.split('.')[0]) >= 2:
                # NumPy 2.x - basicsr may not be compatible
                print("  ⚠ NumPy 2.x detected - Real-ESRGAN not compatible")
                print("  → Using fallback upscaling (install numpy<2 for Real-ESRGAN)")
                self.model = None
                return
            
            # Suppress NumPy warnings during import
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                except (ImportError, RuntimeError, AttributeError, Exception) as e:
                    # NumPy 2.x incompatibility or other import issues
                    print(f"  ⚠ Real-ESRGAN not available: {type(e).__name__}")
                    print("  → Using fallback upscaling")
                    self.model = None
                    return
            
            # Create model architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            
            # Ensure weights directory exists
            self.config.weights_dir.mkdir(parents=True, exist_ok=True)
            weights_path = self.config.weights_dir / "RealESRGAN_x4plus.pth"
            
            # Download weights if needed
            if not weights_path.exists():
                print("  📥 Downloading Real-ESRGAN weights (64MB)...")
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                urllib.request.urlretrieve(url, str(weights_path))
                print("  ✓ Weights downloaded")
            
            # Adjust tile size for MPS to reduce memory usage
            tile_size = self.config.sr_tile_size
            if self.device == "mps":
                # Smaller tiles for MPS to avoid OOM
                tile_size = min(tile_size, 200)
            
            # Create enhancer
            # Use half precision (FP16) for CUDA for faster processing
            use_half = self.device == "cuda"
            self.model = RealESRGANer(
                scale=4,
                model_path=str(weights_path),
                model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=use_half,  # FP16 on CUDA for 2x speed
                device=self.device
            )
            print(f"  ✓ Real-ESRGAN loaded on {self.device}")
            
        except ImportError as e:
            print(f"  ⚠ Real-ESRGAN not available: {e}")
            print("  → Using fallback upscaling")
            self.model = None
        except Exception as e:
            print(f"  ⚠ Failed to load Real-ESRGAN: {e}")
            self.model = None
    
    def process(self, img: np.ndarray, scale: int = None) -> np.ndarray:
        """
        Upscale image using Real-ESRGAN or fallback.
        
        Args:
            img: Input BGR image
            scale: Output scale (2 or 4, default from config)
            
        Returns:
            Upscaled BGR image
        """
        if scale is None:
            scale = self.config.sr_scale
        
        if self.model is not None:
            return self._upscale_realesrgan(img, scale)
        else:
            return self._upscale_fallback(img, scale)
    
    def _upscale_realesrgan(self, img: np.ndarray, scale: int) -> np.ndarray:
        """Upscale using Real-ESRGAN with MPS memory handling"""
        try:
            # Clear MPS cache before processing
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Real-ESRGAN is 4x, so adjust if needed
            if scale == 2:
                output, _ = self.model.enhance(img, outscale=2)
            else:
                output, _ = self.model.enhance(img, outscale=4)
            
            # Clear MPS cache after processing
            if self.device == "mps":
                torch.mps.empty_cache()
            
            return output
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
                print(f"  ⚠ MPS out of memory in Real-ESRGAN: {e}")
                print("  → Using high-quality fallback upscaling")
                return self._upscale_fallback(img, scale)
            else:
                print(f"  ⚠ Real-ESRGAN failed: {e}, using fallback")
                return self._upscale_fallback(img, scale)
        except (NameError, UnboundLocalError) as e:
            # Handle library bugs like 'output_tile' referenced before assignment
            if "output_tile" in str(e) or "referenced before assignment" in str(e):
                print(f"  ⚠ Real-ESRGAN tile processing error: {e}")
                print("  → Using high-quality fallback upscaling")
                return self._upscale_fallback(img, scale)
            else:
                print(f"  ⚠ Real-ESRGAN failed: {e}, using fallback")
                return self._upscale_fallback(img, scale)
        except Exception as e:
            print(f"  ⚠ Real-ESRGAN failed: {e}, using fallback")
            return self._upscale_fallback(img, scale)
    
    def _upscale_fallback(self, img: np.ndarray, scale: int) -> np.ndarray:
        """
        High-quality fallback upscaling using multiple techniques.
        Better than simple interpolation.
        """
        h, w = img.shape[:2]
        new_size = (w * scale, h * scale)
        
        # Use INTER_LANCZOS4 for best quality
        upscaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply subtle sharpening to compensate for interpolation blur
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 1.5)
        upscaled = cv2.addWeighted(upscaled, 1.3, gaussian, -0.3, 0)
        
        # Edge enhancement
        upscaled = self._enhance_edges(upscaled)
        
        return upscaled
    
    def _enhance_edges(self, img: np.ndarray) -> np.ndarray:
        """Enhance edges after upscaling"""
        # Convert to LAB and enhance L channel edges
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Unsharp mask on L channel only
        gaussian = cv2.GaussianBlur(l, (0, 0), 2)
        l = cv2.addWeighted(l, 1.2, gaussian, -0.2, 0)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @property
    def is_ai_available(self) -> bool:
        """Check if AI upscaling is available"""
        return self.model is not None


def super_resolve(img: np.ndarray, scale: int = 4, config: PipelineConfig = None) -> np.ndarray:
    """Convenience function for super-resolution"""
    return SuperResolver(config).process(img, scale)


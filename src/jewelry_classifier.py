"""
Jewelry Classification Module
=============================
Classifies jewelry images into 5 categories:
- BRACELET
- EARRINGS
- NECKLACE
- RINGS
- WATCH
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
from torchvision import transforms


JEWELRY_CLASSES = [
    "BRACELET",
    "EARRINGS", 
    "NECKLACE",
    "RINGS",
    "WATCH"
]


class JewelryClassifier:
    """
    Classify jewelry images into 5 categories.
    Uses a custom trained model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained model file (.pth, .pt, or .onnx)
                       If None, will look for model in weights/ directory
        """
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self.transform = self._get_transform()
        self.classes = JEWELRY_CLASSES
        
        if model_path or self._find_model():
            self._load_model()
        else:
            print("  ⚠ Classification model not found - classification disabled")
            print("     Place your model in: weights/jewelry_classifier.pth")
    
    def _get_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    
    def _find_model(self) -> Optional[Path]:
        """Find model file in weights directory"""
        weights_dir = Path(__file__).parent.parent / "weights"
        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for common model file names
        model_names = [
            "jewelry_classifier.pth",
            "jewelry_classifier.pt",
            "jewelry_classifier.onnx",
            "classifier.pth",
            "classifier.pt"
        ]
        
        for name in model_names:
            model_path = weights_dir / name
            if model_path.exists():
                self.model_path = str(model_path)
                return model_path
        
        return None
    
    def _load_model(self):
        """Load the classification model"""
        try:
            if self.model_path is None:
                return
            
            model_path = Path(self.model_path)
            if not model_path.exists():
                print(f"  ⚠ Model not found: {model_path}")
                return
            
            # Try loading as PyTorch model
            if model_path.suffix in ['.pth', '.pt']:
                self._load_pytorch_model(model_path)
            elif model_path.suffix == '.onnx':
                self._load_onnx_model(model_path)
            else:
                print(f"  ⚠ Unsupported model format: {model_path.suffix}")
                
        except Exception as e:
            print(f"  ⚠ Failed to load classification model: {e}")
            self.model = None
    
    def _load_pytorch_model(self, model_path: Path):
        """Load PyTorch model"""
        try:
            # Try loading with map_location for device compatibility
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need model architecture - try to infer or use default
                    self.model = self._create_default_model()
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume entire dict is state_dict
                    self.model = self._create_default_model()
                    self.model.load_state_dict(checkpoint)
            else:
                # Assume it's the model itself
                self.model = checkpoint
            
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                print(f"  ✓ Classification model loaded from {model_path.name}")
        except Exception as e:
            print(f"  ⚠ Error loading PyTorch model: {e}")
            self.model = None
    
    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(model_path))
            print(f"  ✓ ONNX classification model loaded from {model_path.name}")
        except ImportError:
            print("  ⚠ onnxruntime not installed - cannot load ONNX model")
            self.model = None
        except Exception as e:
            print(f"  ⚠ Error loading ONNX model: {e}")
            self.model = None
    
    def _create_default_model(self):
        """Create a default ResNet-based model architecture"""
        # Default architecture - user should replace with their actual model
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        return model
    
    def is_available(self) -> bool:
        """Check if classifier is available"""
        return self.model is not None
    
    def classify(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Classify a jewelry image.
        
        Args:
            img: Input BGR image (numpy array)
            
        Returns:
            Tuple of (class_name, confidence)
            Returns ("UNKNOWN", 0.0) if classification fails or model not available
        """
        if not self.is_available():
            return ("UNKNOWN", 0.0)
        
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess
            input_tensor = self.transform(pil_img).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                if isinstance(self.model, type) or hasattr(self.model, 'forward'):
                    # PyTorch model
                    outputs = self.model(input_tensor)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    probs = torch.softmax(outputs, dim=1)
                else:
                    # ONNX model
                    input_name = self.model.get_inputs()[0].name
                    outputs = self.model.run(None, {input_name: input_tensor.cpu().numpy()})
                    probs = torch.softmax(torch.tensor(outputs[0]), dim=1)
            
            # Get top prediction
            top_prob, top_idx = torch.max(probs, 1)
            class_name = self.classes[top_idx.item()]
            confidence = top_prob.item()
            
            return (class_name, confidence)
            
        except Exception as e:
            print(f"  ⚠ Classification error: {e}")
            return ("UNKNOWN", 0.0)
    
    def is_valid_jewelry(self, img: np.ndarray, min_confidence: float = 0.5) -> Tuple[bool, str, float]:
        """
        Check if image is valid jewelry (one of the 5 classes).
        
        Args:
            img: Input BGR image
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (is_valid, class_name, confidence)
        """
        class_name, confidence = self.classify(img)
        
        if class_name == "UNKNOWN" or confidence < min_confidence:
            return (False, class_name, confidence)
        
        return (True, class_name, confidence)


def classify_jewelry(img: np.ndarray, model_path: Optional[str] = None) -> Tuple[str, float]:
    """Convenience function for classification"""
    classifier = JewelryClassifier(model_path)
    return classifier.classify(img)





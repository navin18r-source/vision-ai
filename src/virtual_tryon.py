"""
Virtual Try-On Module
====================
Implements virtual try-on functionality for jewelry.
Supports multiple approaches: MediaPipe-based, API-based, and custom models.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Literal
from pathlib import Path


class VirtualTryOn:
    """
    Virtual try-on for jewelry items.
    Places jewelry on person images at appropriate positions.
    """
    
    def __init__(self, method: Literal["mediapipe", "api", "custom"] = "mediapipe"):
        """
        Initialize virtual try-on.
        
        Args:
            method: Implementation method
                - "mediapipe": Uses MediaPipe for pose/keypoint detection
                - "api": Calls external try-on API
                - "custom": Uses custom trained model
        """
        self.method = method
        self.mp_hands = None
        self.mp_pose = None
        self.mp_face = None
        
        if method == "mediapipe":
            self._init_mediapipe()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe models"""
        try:
            import mediapipe as mp
            self.mp = mp
            
            # Initialize MediaPipe solutions
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
            
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5
            )
            
            self.mp_face = mp.solutions.face_detection
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            
            print("  ✓ MediaPipe initialized for virtual try-on")
        except ImportError:
            print("  ⚠ MediaPipe not installed - install with: pip install mediapipe")
            self.method = "simple"  # Fallback to simple method
    
    def try_on(self, 
               jewelry_img: np.ndarray,
               person_img: np.ndarray,
               jewelry_type: str) -> np.ndarray:
        """
        Place jewelry on person image.
        
        Args:
            jewelry_img: Enhanced jewelry image (BGR or BGRA numpy array)
            person_img: Person image (BGR numpy array)
            jewelry_type: Type of jewelry (BRACELET, EARRINGS, NECKLACE, RINGS, WATCH)
        
        Returns:
            Composite image (BGR numpy array)
        """
        print(f"  🎯 Try-on: {jewelry_type}, method={self.method}")
        print(f"     Jewelry: {jewelry_img.shape}, Person: {person_img.shape}")
        
        # Use simple method for faster, more reliable results
        # MediaPipe can hang on some images
        try:
            if self.method == "mediapipe" and self.mp_hands is not None:
                result = self._try_on_mediapipe(jewelry_img, person_img, jewelry_type)
            else:
                result = self._try_on_simple(jewelry_img, person_img, jewelry_type)
            print(f"  ✓ Try-on complete")
            return result
        except Exception as e:
            print(f"  ⚠ Try-on error: {e}, falling back to simple method")
            return self._try_on_simple(jewelry_img, person_img, jewelry_type)
    
    def _try_on_mediapipe(self, 
                         jewelry_img: np.ndarray,
                         person_img: np.ndarray,
                         jewelry_type: str) -> np.ndarray:
        """Try-on using MediaPipe for detection"""
        result = person_img.copy()
        
        # Convert to RGB for MediaPipe
        person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        if jewelry_type == "RINGS":
            return self._place_ring_mediapipe(jewelry_img, result, person_rgb)
        elif jewelry_type == "BRACELET":
            return self._place_bracelet_mediapipe(jewelry_img, result, person_rgb)
        elif jewelry_type == "NECKLACE":
            return self._place_necklace_mediapipe(jewelry_img, result, person_rgb)
        elif jewelry_type == "EARRINGS":
            return self._place_earrings_mediapipe(jewelry_img, result, person_rgb)
        elif jewelry_type == "WATCH":
            return self._place_watch_mediapipe(jewelry_img, result, person_rgb)
        else:
            return self._try_on_simple(jewelry_img, result, jewelry_type)
    
    def _place_ring_mediapipe(self, 
                              jewelry_img: np.ndarray,
                              person_img: np.ndarray,
                              person_rgb: np.ndarray) -> np.ndarray:
        """Place ring on finger using MediaPipe Hands"""
        if self.hands is None:
            return self._place_ring_simple(jewelry_img, person_img)
        
        results = self.hands.process(person_rgb)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Ring finger tip (landmark 12) or index finger tip (landmark 8)
            finger_tip = hand_landmarks.landmark[12]  # Ring finger tip
            
            h, w = person_img.shape[:2]
            x = int(finger_tip.x * w)
            y = int(finger_tip.y * h)
            
            # Get finger width for sizing
            finger_mcp = hand_landmarks.landmark[9]  # Middle finger MCP
            finger_width = abs(finger_tip.x - finger_mcp.x) * w
            
            # Resize jewelry to fit finger with high-quality interpolation
            ring_size = max(30, int(finger_width * 1.5))
            jewelry_resized = cv2.resize(jewelry_img, (ring_size, ring_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Composite onto person image
            return self._composite_image(person_img, jewelry_resized, (x, y))
        
        return person_img
    
    def _place_bracelet_mediapipe(self,
                                  jewelry_img: np.ndarray,
                                  person_img: np.ndarray,
                                  person_rgb: np.ndarray) -> np.ndarray:
        """Place bracelet on wrist using MediaPipe Hands (more accurate)"""
        if self.hands is None:
            return self._place_bracelet_simple(jewelry_img, person_img)
        
        results = self.hands.process(person_rgb)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            h, w = person_img.shape[:2]
            
            # Get wrist landmark (0) and middle finger MCP (9) for orientation
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            
            # Calculate wrist position
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            
            # Calculate hand orientation (angle from wrist to middle finger MCP)
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate wrist width from hand landmarks
            # Use distance between index finger MCP (5) and pinky MCP (17)
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            wrist_width = np.sqrt(
                (index_mcp.x - pinky_mcp.x)**2 * w**2 + 
                (index_mcp.y - pinky_mcp.y)**2 * h**2
            )
            
            # Resize bracelet to fit wrist (slightly larger than wrist width)
            bracelet_width = int(wrist_width * 1.3)
            bracelet_height = int(bracelet_width * 0.4)  # Aspect ratio
            
            # Resize jewelry with high-quality interpolation
            jewelry_resized = cv2.resize(jewelry_img, (bracelet_width, bracelet_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Rotate bracelet to match hand orientation
            if abs(angle) > 10:  # Only rotate if significant angle
                center = (bracelet_width // 2, bracelet_height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                jewelry_resized = cv2.warpAffine(
                    jewelry_resized, M, 
                    (bracelet_width, bracelet_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_TRANSPARENT
                )
            
            # Position: center on wrist, slightly offset based on hand orientation
            offset_x = int(np.cos(np.radians(angle)) * wrist_width * 0.2)
            offset_y = int(np.sin(np.radians(angle)) * wrist_width * 0.2)
            
            x = wrist_x - bracelet_width // 2 + offset_x
            y = wrist_y - bracelet_height // 2 + offset_y
            
            # Composite onto person image
            return self._composite_image(person_img, jewelry_resized, (x, y))
        
        # Fallback to simple if no hands detected
        return self._place_bracelet_simple(jewelry_img, person_img)
    
    def _place_necklace_mediapipe(self,
                                  jewelry_img: np.ndarray,
                                  person_img: np.ndarray,
                                  person_rgb: np.ndarray) -> np.ndarray:
        """Place necklace on neck/chest using MediaPipe Face"""
        if self.face_detector is None:
            return self._place_necklace_simple(jewelry_img, person_img)
        
        results = self.face_detector.process(person_rgb)
        
        if results.detections:
            # Get first face
            face = results.detections[0]
            bbox = face.location_data.relative_bounding_box
            
            h, w = person_img.shape[:2]
            
            # Neck position (below face, center)
            face_center_x = int((bbox.xmin + bbox.width / 2) * w)
            neck_y = int((bbox.ymin + bbox.height) * h) + 50  # Below face
            
            # Resize necklace with high-quality interpolation
            necklace_width = 200
            necklace_height = 100
            jewelry_resized = cv2.resize(jewelry_img, (necklace_width, necklace_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Composite onto person image
            return self._composite_image(person_img, jewelry_resized, 
                                      (face_center_x - necklace_width // 2, neck_y))
        
        return person_img
    
    def _place_earrings_mediapipe(self,
                                 jewelry_img: np.ndarray,
                                 person_img: np.ndarray,
                                 person_rgb: np.ndarray) -> np.ndarray:
        """Place earrings on ears using MediaPipe Face"""
        if self.face_detector is None:
            return self._place_earrings_simple(jewelry_img, person_img)
        
        results = self.face_detector.process(person_rgb)
        
        if results.detections:
            face = results.detections[0]
            bbox = face.location_data.relative_bounding_box
            
            h, w = person_img.shape[:2]
            
            # Ear positions (left and right side of face)
            face_left = int(bbox.xmin * w)
            face_right = int((bbox.xmin + bbox.width) * w)
            ear_y = int((bbox.ymin + bbox.height * 0.3) * h)  # Upper part of face
            
            # Resize earrings with high-quality interpolation
            earring_size = 60
            jewelry_resized = cv2.resize(jewelry_img, (earring_size, earring_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Place on both ears
            result = person_img.copy()
            result = self._composite_image(result, jewelry_resized, 
                                          (face_left - 20, ear_y))
            result = self._composite_image(result, jewelry_resized, 
                                          (face_right - 40, ear_y))
            
            return result
        
        return person_img
    
    def _place_watch_mediapipe(self,
                               jewelry_img: np.ndarray,
                               person_img: np.ndarray,
                               person_rgb: np.ndarray) -> np.ndarray:
        """Place watch on wrist (similar to bracelet but square shape)"""
        if self.hands is None:
            return self._place_watch_simple(jewelry_img, person_img)
        
        results = self.hands.process(person_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = person_img.shape[:2]
            
            # Get wrist landmark
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            
            # Calculate orientation
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Watch size (square, slightly larger than bracelet)
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            wrist_width = np.sqrt(
                (index_mcp.x - pinky_mcp.x)**2 * w**2 + 
                (index_mcp.y - pinky_mcp.y)**2 * h**2
            )
            
            watch_size = int(wrist_width * 1.5)
            jewelry_resized = cv2.resize(jewelry_img, (watch_size, watch_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Rotate to match wrist orientation
            if abs(angle) > 10:
                center = (watch_size // 2, watch_size // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                jewelry_resized = cv2.warpAffine(
                    jewelry_resized, M, 
                    (watch_size, watch_size),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_TRANSPARENT
                )
            
            x = wrist_x - watch_size // 2
            y = wrist_y - watch_size // 2
            
            return self._composite_image(person_img, jewelry_resized, (x, y))
        
        return self._place_watch_simple(jewelry_img, person_img)
    
    def _place_watch_simple(self, jewelry_img, person_img):
        """Simple watch placement"""
        return self._try_on_simple(jewelry_img, person_img, "WATCH")
    
    def _try_on_simple(self,
                      jewelry_img: np.ndarray,
                      person_img: np.ndarray,
                      jewelry_type: str) -> np.ndarray:
        """Simple try-on without detection (centered placement) - FAST and RELIABLE"""
        h, w = person_img.shape[:2]
        
        # Ensure person_img is BGR (3 channels)
        if len(person_img.shape) == 3 and person_img.shape[2] == 4:
            person_img = person_img[:, :, :3]
        
        # Get jewelry dimensions
        jewelry_h, jewelry_w = jewelry_img.shape[:2]
        
        # Scale jewelry to fit nicely (25% of person image)
        scale = min(w * 0.25 / jewelry_w, h * 0.25 / jewelry_h)
        new_w = max(50, int(jewelry_w * scale))
        new_h = max(50, int(jewelry_h * scale))
        
        # Resize with high quality
        jewelry_resized = cv2.resize(jewelry_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate position based on jewelry type
        if jewelry_type == "RINGS":
            # Center of image (finger area)
            center_x = w // 2
            center_y = int(h * 0.55)
        elif jewelry_type in ["BRACELET", "WATCH"]:
            # Lower center (wrist area)
            center_x = w // 2
            center_y = int(h * 0.65)
        elif jewelry_type == "NECKLACE":
            # Upper center (neck area)
            center_x = w // 2
            center_y = int(h * 0.35)
        elif jewelry_type == "EARRINGS":
            # Upper side (ear area)
            center_x = w // 2
            center_y = int(h * 0.25)
        else:
            # Default: center
            center_x = w // 2
            center_y = h // 2
        
        # Use composite with center position
        return self._composite_image(person_img, jewelry_resized, (center_x, center_y))
    
    def _place_ring_simple(self, jewelry_img, person_img):
        """Simple ring placement"""
        return self._try_on_simple(jewelry_img, person_img, "RINGS")
    
    def _place_bracelet_simple(self, jewelry_img, person_img):
        """Simple bracelet placement"""
        return self._try_on_simple(jewelry_img, person_img, "BRACELET")
    
    def _place_necklace_simple(self, jewelry_img, person_img):
        """Simple necklace placement"""
        return self._try_on_simple(jewelry_img, person_img, "NECKLACE")
    
    def _place_earrings_simple(self, jewelry_img, person_img):
        """Simple earrings placement"""
        return self._try_on_simple(jewelry_img, person_img, "EARRINGS")
    
    def _composite_image(self,
                        background: np.ndarray,
                        foreground: np.ndarray,
                        position: Tuple[int, int],
                        rotation: float = 0,
                        scale: float = 1.0) -> np.ndarray:
        """
        Composite foreground (WITH ALPHA) onto background with proper blending.
        
        Args:
            background: Person image (BGR, 3-channel)
            foreground: Jewelry image (BGRA, 4-channel with transparency)
            position: (x, y) center position for placement
            rotation: Rotation angle in degrees (optional)
            scale: Scale factor (optional)
            
        Returns:
            Composited image (BGR, 3-channel)
        """
        result = background.copy()
        
        # Ensure foreground has alpha channel
        if len(foreground.shape) < 3:
            return result  # Invalid foreground
            
        if foreground.shape[2] == 3:
            # No alpha - create one (assume non-black pixels are object)
            gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            foreground = np.dstack([foreground, alpha])
        
        fg_h, fg_w = foreground.shape[:2]
        
        # Apply scale
        if scale != 1.0:
            new_w = int(fg_w * scale)
            new_h = int(fg_h * scale)
            foreground = cv2.resize(foreground, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            fg_h, fg_w = new_h, new_w
        
        # Apply rotation
        if abs(rotation) > 0.1:
            center = (fg_w // 2, fg_h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            foreground = cv2.warpAffine(
                foreground, M, (fg_w, fg_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)  # Transparent border
            )
        
        # Calculate placement (position is center)
        x = position[0] - fg_w // 2
        y = position[1] - fg_h // 2
        
        # Calculate actual placement area (handle out-of-bounds)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(result.shape[1], x + fg_w)
        y2 = min(result.shape[0], y + fg_h)
        
        # Source region from foreground
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)
        
        # Extract regions
        dst_region = result[y1:y2, x1:x2]
        src_region = foreground[src_y1:src_y2, src_x1:src_x2]
        
        # Alpha blending with feathering
        if src_region.shape[:2] == dst_region.shape[:2] and src_region.size > 0:
            # Extract alpha channel
            alpha = src_region[:, :, 3:4].astype(np.float32) / 255.0
            
            # Apply feathering (smooth edges)
            kernel_size = min(5, min(alpha.shape[:2]))
            if kernel_size >= 3:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                alpha_feathered = cv2.filter2D(alpha, -1, kernel)
            else:
                alpha_feathered = alpha
            
            # Extract BGR from foreground
            fg_rgb = src_region[:, :, :3].astype(np.float32)
            
            # Alpha blending
            blended = (fg_rgb * alpha_feathered + dst_region.astype(np.float32) * (1 - alpha_feathered)).astype(np.uint8)
            result[y1:y2, x1:x2] = blended
        
        return result
    
    def _try_on_api(self,
                   jewelry_img: np.ndarray,
                   person_img: np.ndarray,
                   jewelry_type: str) -> np.ndarray:
        """Try-on using external API"""
        # This would call an external API
        # For now, fallback to simple method
        return self._try_on_simple(jewelry_img, person_img, jewelry_type)


def try_on_jewelry(jewelry_img: np.ndarray,
                   person_img: np.ndarray,
                   jewelry_type: str,
                   method: str = "mediapipe") -> np.ndarray:
    """Convenience function for virtual try-on"""
    tryon = VirtualTryOn(method=method)
    return tryon.try_on(jewelry_img, person_img, jewelry_type)



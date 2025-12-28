"""
Data Saving Module
==================
Saves enhancement results with metadata:
- Image name
- Detected class
- Enhanced image
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import cv2


class DataSaver:
    """Save enhancement results and metadata"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize data saver.
        
        Args:
            output_dir: Directory to save data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.output_dir / "metadata.json"
        self.csv_file = self.output_dir / "enhancements.csv"
        
        # Initialize CSV if it doesn't exist
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'image_name', 'detected_class', 
                    'confidence', 'enhanced_image_path', 'processing_time'
                ])
    
    def save(self, 
             image_name: str,
             detected_class: str,
             confidence: float,
             enhanced_image: bytes,
             processing_time: float = 0.0) -> Dict:
        """
        Save enhancement result.
        
        Args:
            image_name: Original image name
            detected_class: Detected jewelry class
            confidence: Classification confidence
            enhanced_image: Enhanced image as bytes or numpy array
            processing_time: Processing time in seconds
            
        Returns:
            Dictionary with saved data info
        """
        timestamp = datetime.now().isoformat()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate output filename
        safe_name = Path(image_name).stem
        output_filename = f"{timestamp_str}_{safe_name}_enhanced.png"
        output_path = self.images_dir / output_filename
        
        # Save enhanced image
        if isinstance(enhanced_image, bytes):
            with open(output_path, 'wb') as f:
                f.write(enhanced_image)
        else:
            # Assume numpy array
            cv2.imwrite(str(output_path), enhanced_image)
        
        # Create metadata entry
        entry = {
            'timestamp': timestamp,
            'image_name': image_name,
            'detected_class': detected_class,
            'confidence': float(confidence),
            'enhanced_image_path': str(output_path.relative_to(self.output_dir)),
            'processing_time': float(processing_time)
        }
        
        # Append to JSON metadata
        metadata = []
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        
        metadata.append(entry)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                entry['timestamp'],
                entry['image_name'],
                entry['detected_class'],
                entry['confidence'],
                entry['enhanced_image_path'],
                entry['processing_time']
            ])
        
        return entry





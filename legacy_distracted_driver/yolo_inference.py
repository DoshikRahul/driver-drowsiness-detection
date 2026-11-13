import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path

class DistractedDriverDetector:
    def __init__(self, model_path=None):
        """Initialize the distracted driver detector with trained YOLO model"""
        
        # Default model path
        if model_path is None:
            model_path = "runs/detect/distracted_driver_fast/weights/best.pt"
        
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'drinking',
            'hair and makeup', 
            'operating the radio',
            'reaching behind',
            'safe driving',
            'talking on the phone',
            'talking to passenger',
            'texting'
        ]
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
        return True
    
    def preprocess_image(self, image_path):
        """Preprocess image for YOLO inference"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb, image
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None, None
    
    def detect_distracted_behavior(self, image_path, confidence_threshold=0.5):
        """
        Detect distracted driving behavior in an image
        
        Args:
            image_path (str): Path to the image file
            confidence_threshold (float): Minimum confidence for detections
            
        Returns:
            dict: Detection results with predictions and metadata
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded',
                'predictions': []
            }
        
        try:
            # Preprocess image
            image_rgb, image_bgr = self.preprocess_image(image_path)
            if image_rgb is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'predictions': []
                }
            
            # Run inference
            results = self.model(image_path, conf=confidence_threshold)
            
            # Process results
            predictions = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        prediction = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2)
                            }
                        }
                        predictions.append(prediction)
            
            # Determine overall behavior
            behavior_summary = self.analyze_behavior(predictions)
            
            return {
                'success': True,
                'predictions': predictions,
                'behavior_summary': behavior_summary,
                'total_detections': len(predictions),
                'image_path': image_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Detection failed: {str(e)}',
                'predictions': []
            }
    
    def analyze_behavior(self, predictions):
        """Analyze detected behaviors and provide summary"""
        if not predictions:
            return {
                'status': 'no_detection',
                'message': 'No behavior detected',
                'risk_level': 'unknown'
            }
        
        # Get highest confidence prediction
        best_prediction = max(predictions, key=lambda x: x['confidence'])
        class_name = best_prediction['class_name']
        confidence = best_prediction['confidence']
        
        # Determine risk level
        high_risk_behaviors = ['drinking', 'texting', 'talking on the phone']
        medium_risk_behaviors = ['hair and makeup', 'reaching behind', 'operating the radio']
        safe_behaviors = ['safe driving', 'talking to passenger']
        
        if class_name in high_risk_behaviors:
            risk_level = 'high'
            message = f"HIGH RISK: Driver detected {class_name} (confidence: {confidence:.2f})"
        elif class_name in medium_risk_behaviors:
            risk_level = 'medium'
            message = f"MEDIUM RISK: Driver detected {class_name} (confidence: {confidence:.2f})"
        elif class_name in safe_behaviors:
            risk_level = 'low'
            message = f"LOW RISK: Driver appears to be {class_name} (confidence: {confidence:.2f})"
        else:
            risk_level = 'unknown'
            message = f"UNKNOWN: Detected {class_name} (confidence: {confidence:.2f})"
        
        return {
            'status': 'detected',
            'primary_behavior': class_name,
            'confidence': confidence,
            'risk_level': risk_level,
            'message': message
        }
    
    def draw_predictions(self, image_path, predictions, output_path=None):
        """Draw bounding boxes and labels on image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Define colors for different risk levels
            colors = {
                'high': (0, 0, 255),      # Red
                'medium': (0, 165, 255),   # Orange
                'low': (0, 255, 0),       # Green
                'unknown': (128, 128, 128) # Gray
            }
            
            for pred in predictions:
                bbox = pred['bbox']
                class_name = pred['class_name']
                confidence = pred['confidence']
                
                # Determine color based on risk
                high_risk = ['drinking', 'texting', 'talking on the phone']
                medium_risk = ['hair and makeup', 'reaching behind', 'operating the radio']
                
                if class_name in high_risk:
                    color = colors['high']
                elif class_name in medium_risk:
                    color = colors['medium']
                else:
                    color = colors['low']
                
                # Draw bounding box
                x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save annotated image
            if output_path:
                cv2.imwrite(output_path, image)
            
            return image
            
        except Exception as e:
            print(f"Error drawing predictions: {str(e)}")
            return None

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = DistractedDriverDetector()
    
    # Test with sample images (if available)
    test_images = [
        "1.jpg",
        "2.webp", 
        "3.webp",
        "4.webp",
        "5.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nTesting with {image_path}:")
            results = detector.detect_distracted_behavior(image_path)
            
            if results['success']:
                print(f"Detections: {results['total_detections']}")
                print(f"Behavior: {results['behavior_summary']['message']}")
                
                # Draw predictions
                annotated_image = detector.draw_predictions(
                    image_path, 
                    results['predictions'], 
                    f"annotated_{image_path}"
                )
                if annotated_image is not None:
                    print(f"Annotated image saved as annotated_{image_path}")
            else:
                print(f"Error: {results['error']}")
        else:
            print(f"Image {image_path} not found")

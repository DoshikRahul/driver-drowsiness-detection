from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_distracted_driver_model_fast():
    """Train YOLOv8 model for distracted driver detection - FAST VERSION"""
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Load YOLOv8 model (using nano version for fastest training)
    model = YOLO('yolov8n.pt')  # Load pre-trained model
    
    # FAST Training parameters - optimized for speed
    training_args = {
        'data': 'data.yaml',           # Path to dataset YAML
        'epochs': 30,                  # Reduced from 100 to 30 epochs
        'imgsz': 416,                  # Reduced image size from 640 to 416
        'batch': 32,                   # Increased batch size for efficiency
        'lr0': 0.01,                   # Initial learning rate
        'lrf': 0.01,                   # Final learning rate
        'momentum': 0.937,             # Momentum
        'weight_decay': 0.0005,        # Weight decay
        'warmup_epochs': 1,            # Reduced warmup epochs
        'warmup_momentum': 0.8,        # Warmup momentum
        'box': 7.5,                    # Box loss gain
        'cls': 0.5,                    # Classification loss gain
        'dfl': 1.5,                    # DFL loss gain
        'val': True,                   # Validate during training
        'save': True,                  # Save checkpoints
        'save_period': 10,             # Save checkpoint every 10 epochs
        'cache': True,                 # Cache images for faster training
        'device': device,              # Training device
        'workers': 4,                  # Reduced workers for stability
        'project': 'runs/detect',      # Project directory
        'name': 'distracted_driver_fast',   # Experiment name
        'exist_ok': True,              # Overwrite existing project
        'pretrained': True,            # Use pretrained weights
        'optimizer': 'AdamW',          # AdamW for faster convergence
        'verbose': True,               # Verbose output
        'seed': 0,                     # Random seed
        'deterministic': False,        # Non-deterministic for speed
        'single_cls': False,           # Single class training
        'rect': True,                  # Rectangular training for speed
        'cos_lr': True,                # Cosine learning rate scheduler
        'close_mosaic': 5,             # Close mosaic augmentation early
        'resume': False,               # Resume training
        'amp': True,                   # Automatic Mixed Precision
        'fraction': 1.0,               # Dataset fraction to use
        'profile': False,              # Profile ONNX and TensorRT speeds
        'freeze': None,                # Freeze layers
        'patience': 10,                # Early stopping patience
    }
    
    print("Starting FAST YOLOv8 training for distracted driver detection...")
    print(f"Dataset: {training_args['data']}")
    print(f"Epochs: {training_args['epochs']} (reduced for speed)")
    print(f"Batch size: {training_args['batch']} (increased for efficiency)")
    print(f"Image size: {training_args['imgsz']} (reduced for speed)")
    print("Expected training time: 2-4 hours")
    
    # Start training
    results = model.train(**training_args)
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    # Validate the model
    print("Running validation...")
    val_results = model.val()
    
    print("Validation completed!")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    # Export the model
    print("Exporting model...")
    model.export(format='onnx')  # Export to ONNX format for deployment
    
    return model, results, val_results

if __name__ == "__main__":
    # Create runs directory if it doesn't exist
    Path("runs/detect").mkdir(parents=True, exist_ok=True)
    
    # Train the model
    model, results, val_results = train_distracted_driver_model_fast()
    
    print("\n" + "="*50)
    print("FAST TRAINING SUMMARY")
    print("="*50)
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")
    print(f"Training plots saved to: {results.save_dir}")
    print("="*50)

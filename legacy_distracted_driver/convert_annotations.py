import os
import shutil
from pathlib import Path

def convert_to_yolo_format():
    """Convert the dataset annotations to YOLO format"""
    
    # Define paths
    dataset_dir = Path("Dataset")
    yolo_dataset_dir = Path("yolo_dataset")
    
    # Create YOLO dataset structure
    for split in ['train', 'valid', 'test']:
        (yolo_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Read class names
    with open(dataset_dir / 'train' / '_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(f"Classes found: {classes}")
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")
        
        # Read annotations
        annotations_file = dataset_dir / split / '_annotations.txt'
        if not annotations_file.exists():
            print(f"Warning: {annotations_file} not found, skipping {split}")
            continue
            
        with open(annotations_file, 'r') as f:
            annotations = f.readlines()
        
        processed_count = 0
        for line in annotations:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ')
            if len(parts) != 2:
                continue
                
            image_name = parts[0]
            bbox_info = parts[1].split(',')
            
            if len(bbox_info) != 5:
                continue
                
            x1, y1, x2, y2, class_id = map(int, bbox_info)
            
            # Copy image file
            src_image = dataset_dir / split / image_name
            dst_image = yolo_dataset_dir / split / 'images' / image_name
            
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
                
                # Get image dimensions (assuming 640x480 for now, will be updated)
                img_width, img_height = 640, 480
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x1 + x2) / 2.0 / img_width
                y_center = (y1 + y2) / 2.0 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Create label file
                label_name = image_name.replace('.jpg', '.txt')
                label_file = yolo_dataset_dir / split / 'labels' / label_name
                
                with open(label_file, 'w') as lf:
                    lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                processed_count += 1
        
        print(f"Processed {processed_count} images in {split} split")
    
    # Create data.yaml file
    yaml_content = f"""
path: yolo_dataset
train: train/images
val: valid/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content.strip())
    
    print("Dataset conversion completed!")
    print(f"Created data.yaml with {len(classes)} classes")

if __name__ == "__main__":
    convert_to_yolo_format()

"""
YOLOv11 Fine-tuning on VisDrone 2019 Dataset - GPU OPTIMIZED
Maximizes GPU utilization for fastest training
"""

import os
import shutil
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
import yaml
import torch
import gc

def check_gpu_config():
    """Check and display GPU configuration"""
    print("\n" + "="*60)
    print("GPU CONFIGURATION CHECK")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training will be VERY slow on CPU.")
        print("Please install CUDA and PyTorch with GPU support.")
        return False
    
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n--- GPU {i}: {torch.cuda.get_device_name(i)} ---")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi Processors: {props.multi_processor_count}")
    
    print("\n" + "="*60)
    return True

def optimize_gpu_settings():
    """Set optimal GPU settings for maximum performance"""
    if torch.cuda.is_available():
        # Enable TF32 for Ampere GPUs (RTX 30xx, A100, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN autotuner for optimal performance
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocator settings for better performance
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        print("✓ GPU optimizations enabled:")
        print("  - TF32 enabled for faster computation")
        print("  - cuDNN benchmark mode enabled")
        print("  - Memory allocator optimized")

def calculate_optimal_batch_size():
    """Calculate optimal batch size based on GPU memory"""
    if not torch.cuda.is_available():
        return 4
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Batch size recommendations based on GPU memory
    # These are optimized for YOLOv11n with 640x640 images
    if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
        return 64
    elif gpu_memory_gb >= 16:  # RTX 4080, 3090, etc.
        return 48
    elif gpu_memory_gb >= 12:  # RTX 4070 Ti, 3080 Ti, etc.
        return 32
    elif gpu_memory_gb >= 10:  # RTX 3080, etc.
        return 24
    elif gpu_memory_gb >= 8:   # RTX 3070, 4060 Ti, etc.
        return 16
    elif gpu_memory_gb >= 6:   # RTX 3060, etc.
        return 12
    else:  # RTX 3050, etc.
        return 8

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_visdrone_dataset():
    """Download VisDrone 2019 dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING VISDRONE 2019 DATASET")
    print("="*60)
    
    base_dir = Path("visdrone_dataset")
    base_dir.mkdir(exist_ok=True)
    
    # VisDrone dataset URLs
    urls = {
        'train': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
        'val': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
    }
    
    for split, url in urls.items():
        zip_path = base_dir / f"{split}.zip"
        
        if not zip_path.exists():
            print(f"\nDownloading {split} split...")
            download_file(url, zip_path)
            
            print(f"Extracting {split} split...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            
            zip_path.unlink()
        else:
            print(f"✓ {split} split already downloaded")
    
    print("\n✓ Dataset download complete!")
    return base_dir

def convert_visdrone_to_yolo(visdrone_dir):
    """Convert VisDrone annotations to YOLO format - PEOPLE ONLY"""
    print("\n" + "="*60)
    print("CONVERTING ANNOTATIONS TO YOLO FORMAT (PEOPLE ONLY)")
    print("="*60)
    
    # Only detect people (pedestrian=1 and people=2)
    people_classes = [1, 2]
    classes = ['person']
    
    yolo_dir = Path("visdrone_yolo")
    yolo_dir.mkdir(exist_ok=True)
    
    total_annotations = 0
    total_people = 0
    
    for split in ['train', 'val']:
        img_src = visdrone_dir / f"VisDrone2019-DET-{split}" / "images"
        ann_src = visdrone_dir / f"VisDrone2019-DET-{split}" / "annotations"
        
        img_dst = yolo_dir / "images" / split
        lbl_dst = yolo_dir / "labels" / split
        
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        
        if not ann_src.exists():
            print(f"⚠ Warning: {ann_src} not found, skipping {split}")
            continue
            
        ann_files = list(ann_src.glob("*.txt"))
        print(f"\n📁 Converting {len(ann_files)} {split} annotations...")
        
        split_people = 0
        
        for ann_file in tqdm(ann_files, desc=f"Processing {split}"):
            img_file = img_src / f"{ann_file.stem}.jpg"
            
            if not img_file.exists():
                continue
            
            # Get image dimensions once
            from PIL import Image
            img = Image.open(img_file)
            img_w, img_h = img.size
            
            # Convert annotation
            yolo_lines = []
            with open(ann_file, 'r') as f:
                for line in f:
                    total_annotations += 1
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    
                    x, y, w, h = map(float, parts[:4])
                    cls = int(parts[5])
                    
                    # Only keep pedestrian (1) and people (2) classes
                    if cls not in people_classes:
                        continue
                    
                    total_people += 1
                    split_people += 1
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    # Clamp values to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    norm_w = max(0, min(1, norm_w))
                    norm_h = max(0, min(1, norm_h))
                    
                    # YOLO format: class x_center y_center width height
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
            # Only copy image if it has people annotations
            if yolo_lines:
                shutil.copy(img_file, img_dst / img_file.name)
                with open(lbl_dst / f"{ann_file.stem}.txt", 'w') as f:
                    f.writelines(yolo_lines)
        
        print(f"✓ {split}: Found {split_people} people in {len(list(lbl_dst.glob('*.txt')))} images")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Total annotations processed: {total_annotations}")
    print(f"  People annotations kept: {total_people}")
    print(f"  Filtering rate: {(total_people/total_annotations)*100:.1f}%")
    
    return yolo_dir, classes

def create_yaml_config(yolo_dir, classes):
    """Create YAML configuration file for YOLO"""
    config = {
        'path': str(yolo_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = yolo_dir / "visdrone.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✓ YAML config created: {yaml_path}")
    return yaml_path

def train_yolov11_gpu_optimized(yaml_path):
    """Train YOLOv11 with maximum GPU utilization"""
    print("\n" + "="*60)
    print("STARTING YOLOV11 FINE-TUNING - GPU OPTIMIZED")
    print("="*60)
    
    if not check_gpu_config():
        print("\n⚠ GPU not available. Training on CPU will be extremely slow!")
        response = input("Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return None
    
    try:
        from ultralytics import YOLO
        
        # Optimize GPU settings
        optimize_gpu_settings()
        
        # Calculate optimal batch size
        batch_size = calculate_optimal_batch_size()
        
        # Determine number of workers based on CPU cores
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 8)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"\n📊 TRAINING CONFIGURATION:")
        print(f"  Model: YOLOv11n (Nano)")
        print(f"  Task: Person Detection")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"  Batch Size: {batch_size} (optimized for your GPU)")
        print(f"  Image Size: 640x640")
        print(f"  Epochs: 100")
        print(f"  Workers: {num_workers}")
        print(f"  Mixed Precision: Enabled")
        print(f"  Cache: RAM (for speed)")
        print(f"  Optimizer: AdamW")
        
        # Initialize model with pretrained weights
        print("\n📥 Loading pretrained YOLOv11n model...")
        model = YOLO('yolo11n.pt')
        
        # Training hyperparameters optimized for fine-tuning
        print("\n🚀 Starting training...")
        results = model.train(
            # Data
            data=str(yaml_path),
            
            # Training parameters
            epochs=100,
            batch=batch_size,
            imgsz=640,
            
            # Device settings
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=num_workers,
            
            # Optimization
            optimizer='AdamW',  # Better for fine-tuning
            lr0=0.001,  # Initial learning rate
            lrf=0.01,   # Final learning rate (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Augmentation (reduced for fine-tuning)
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,    # HSV-Saturation augmentation
            hsv_v=0.4,    # HSV-Value augmentation
            degrees=0.0,  # Rotation
            translate=0.1,  # Translation
            scale=0.5,    # Scale
            shear=0.0,    # Shear
            perspective=0.0,  # Perspective
            flipud=0.0,   # Vertical flip
            fliplr=0.5,   # Horizontal flip
            mosaic=1.0,   # Mosaic augmentation
            mixup=0.0,    # Mixup augmentation
            copy_paste=0.0,  # Copy-paste augmentation
            
            # Performance
            amp=True,  # Automatic Mixed Precision
            cache='ram',  # Cache images in RAM for speed
            
            # Validation
            val=True,
            patience=50,  # Early stopping patience
            
            # Saving
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            
            # Output
            project='visdrone_training',
            name='yolov11n_person_detector',
            exist_ok=True,
            
            # Visualization
            plots=True,
            verbose=True,
            
            # Additional optimizations
            close_mosaic=10,  # Disable mosaic in last 10 epochs
            rect=False,  # Rectangular training (can be slower but more accurate)
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"📁 Results saved in: visdrone_training/yolov11n_person_detector")
        print(f"🎯 Model detects: person only")
        print(f"📊 Best weights: visdrone_training/yolov11n_person_detector/weights/best.pt")
        print(f"📊 Last weights: visdrone_training/yolov11n_person_detector/weights/last.pt")
        print("="*60)
        
        # Print final metrics
        if results:
            print("\n📈 FINAL METRICS:")
            print(f"  Best Epoch: {results.best_epoch if hasattr(results, 'best_epoch') else 'N/A'}")
            
        return results
        
    except ImportError:
        print("\n❌ Error: ultralytics package not installed")
        print("Please install it with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model(model_path, test_image_path=None):
    """Test the trained model"""
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        
        # Load trained model
        model = YOLO(model_path)
        
        print(f"✓ Model loaded: {model_path}")
        
        if test_image_path and Path(test_image_path).exists():
            print(f"\n🔍 Running inference on: {test_image_path}")
            results = model(test_image_path)
            
            # Display results
            for r in results:
                print(f"\n📊 Detections: {len(r.boxes)} people found")
                for i, box in enumerate(r.boxes):
                    conf = box.conf[0].item()
                    print(f"  Person {i+1}: Confidence {conf:.2f}")
            
            # Save result
            output_path = Path("test_result.jpg")
            results[0].save(str(output_path))
            print(f"\n✓ Result saved: {output_path}")
        else:
            print("\n💡 To test the model:")
            print(f"   model = YOLO('{model_path}')")
            print("   results = model('path/to/image.jpg')")
            print("   results[0].show()")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("YOLOV11 PERSON DETECTOR - VISDRONE FINE-TUNING")
    print("GPU OPTIMIZED FOR MAXIMUM PERFORMANCE")
    print("="*60)
    
    # Check GPU availability first
    if not check_gpu_config():
        print("\n⚠ WARNING: No GPU detected!")
        print("Training will be EXTREMELY slow on CPU.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled. Please install CUDA and GPU-enabled PyTorch.")
            return
    
    # Step 1: Download dataset
    print("\n[STEP 1/4] Downloading dataset...")
    visdrone_dir = download_visdrone_dataset()
    
    # Step 2: Convert to YOLO format
    print("\n[STEP 2/4] Converting annotations...")
    yolo_dir, classes = convert_visdrone_to_yolo(visdrone_dir)
    
    # Step 3: Create YAML config
    print("\n[STEP 3/4] Creating configuration...")
    yaml_path = create_yaml_config(yolo_dir, classes)
    
    # Step 4: Train model
    print("\n[STEP 4/4] Training model...")
    results = train_yolov11_gpu_optimized(yaml_path)
    
    if results:
        print("\n" + "="*60)
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n💡 NEXT STEPS:")
        print("1. Check training results in: visdrone_training/yolov11n_person_detector")
        print("2. Use best model for inference:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('visdrone_training/yolov11n_person_detector/weights/best.pt')")
        print("   results = model('your_image.jpg')")
        print("   results[0].show()")
        print("="*60)

if __name__ == "__main__":
    main()
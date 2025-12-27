"""
YOLOv11 Person Detector - Video Testing Script
Optimized for drone footage with real-time processing
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import numpy as np
video_path = r"C:\Users\jhadi\nidar"
def test_video(video_path, model_path=None, conf_threshold=0.25, 
               save_output=True, display_live=True, skip_frames=1):
    """
    Test trained model on video file
    
    Args:
        video_path: Path to your video file
        model_path: Path to trained model (None = auto-detect)
        conf_threshold: Confidence threshold (0.1-0.9)
        save_output: Save annotated video
        display_live: Show live detection window
        skip_frames: Process every N frames (1=all frames, 2=every other frame)
    """
    
    print("="*70)
    print("YOLOV11 PERSON DETECTOR - VIDEO TESTING")
    print("="*70)
    
    # Auto-detect model if not provided
    if model_path is None:
        possible_paths = [
            "visdrone_training/yolov11n_person_detector/weights/best.pt",
            "visdrone_training/yolov11n_person_detector/weights/last.pt",
            "runs/detect/train/weights/best.pt",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path is None:
            print("❌ Error: No trained model found!")
            print("Please provide model path manually.")
            return
    
    # Validate paths
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded successfully!")
    
    # Open video
    print(f"\n🎬 Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n📹 Video Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  Frame skip: {skip_frames} (processing every {skip_frames} frame(s))")
    
    # Setup output video writer
    output_path = None
    out = None
    
    if save_output:
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        output_filename = f"result_{Path(video_path).stem}_conf{conf_threshold}.mp4"
        output_path = output_dir / output_filename
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps//skip_frames, (width, height))
        print(f"✓ Output will be saved to: {output_path}")
    
    # Processing statistics
    frame_count = 0
    processed_count = 0
    detection_counts = []
    processing_times = []
    confidence_scores = []
    
    print(f"\n{'='*70}")
    print("PROCESSING VIDEO")
    print("="*70)
    print("Press 'q' to quit, 'p' to pause, SPACE to resume")
    print("-"*70)
    
    paused = False
    start_time = time.time()
    
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if needed
                if frame_count % skip_frames != 0:
                    continue
                
                processed_count += 1
                
                # Run detection
                inference_start = time.time()
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    imgsz=640,
                    device=0,  # GPU
                    verbose=False
                )[0]
                inference_time = (time.time() - inference_start) * 1000
                
                # Get detections
                boxes = results.boxes
                num_detections = len(boxes)
                detection_counts.append(num_detections)
                processing_times.append(inference_time)
                
                # Collect confidence scores
                if num_detections > 0:
                    confidences = boxes.conf.cpu().numpy()
                    confidence_scores.extend(confidences)
                
                # Get annotated frame
                annotated_frame = results.plot(
                    line_width=2,
                    labels=True,
                    conf=True
                )
                
                # Add information overlay
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                progress_pct = (frame_count / total_frames) * 100
                
                # Create info panel
                info_y = 30
                info_texts = [
                    f"Frame: {frame_count}/{total_frames} ({progress_pct:.1f}%)",
                    f"People Detected: {num_detections}",
                    f"Inference FPS: {current_fps:.1f}",
                    f"Inference Time: {inference_time:.1f}ms",
                    f"Confidence: {conf_threshold}"
                ]
                
                # Draw semi-transparent background for text
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (5, 5), (400, 170), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                # Draw text
                for text in info_texts:
                    cv2.putText(annotated_frame, text, (10, info_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    info_y += 30
                
                # Show detection boxes info
                if num_detections > 0:
                    avg_conf = confidences.mean()
                    cv2.putText(annotated_frame, f"Avg Confidence: {avg_conf:.3f}", 
                              (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                              (0, 255, 255), 2)
                
                # Save frame
                if save_output and out is not None:
                    out.write(annotated_frame)
                
                # Display frame
                if display_live:
                    # Resize for display if too large
                    display_frame = annotated_frame
                    if width > 1920:
                        scale = 1920 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                    
                    cv2.imshow('Drone Person Detection - Press Q to quit, P to pause', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⚠ User stopped processing")
                        break
                    elif key == ord('p'):
                        paused = True
                        print("\n⏸ Paused - Press SPACE to resume")
                
                # Progress update every 30 frames
                if processed_count % 30 == 0:
                    avg_fps = 1000 / np.mean(processing_times[-30:]) if processing_times else 0
                    avg_people = np.mean(detection_counts[-30:]) if detection_counts else 0
                    elapsed = time.time() - start_time
                    remaining = (total_frames - frame_count) / (frame_count / elapsed) if frame_count > 0 else 0
                    
                    print(f"Progress: {progress_pct:.1f}% | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Avg People: {avg_people:.1f} | "
                          f"ETA: {remaining:.0f}s")
            
            else:  # Paused
                if display_live:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        paused = False
                        print("▶ Resumed")
                    elif key == ord('q'):
                        print("\n⚠ User stopped processing")
                        break
    
    except KeyboardInterrupt:
        print("\n⚠ Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        if display_live:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        
        # Print final statistics
        print(f"\n{'='*70}")
        print("VIDEO PROCESSING COMPLETE")
        print("="*70)
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"  Total frames: {total_frames}")
        print(f"  Processed frames: {processed_count}")
        print(f"  Processing time: {total_time:.1f}s")
        print(f"  Average FPS: {processed_count/total_time:.1f}")
        
        if detection_counts:
            print(f"\n👥 DETECTION STATISTICS:")
            print(f"  Total people detections: {sum(detection_counts)}")
            print(f"  Average per frame: {np.mean(detection_counts):.2f}")
            print(f"  Max in single frame: {np.max(detection_counts)}")
            print(f"  Min in single frame: {np.min(detection_counts)}")
            print(f"  Frames with people: {np.sum(np.array(detection_counts) > 0)} "
                  f"({(np.sum(np.array(detection_counts) > 0)/len(detection_counts)*100):.1f}%)")
        
        if confidence_scores:
            print(f"\n🎯 CONFIDENCE STATISTICS:")
            print(f"  Average confidence: {np.mean(confidence_scores):.3f}")
            print(f"  Min confidence: {np.min(confidence_scores):.3f}")
            print(f"  Max confidence: {np.max(confidence_scores):.3f}")
            print(f"  Std deviation: {np.std(confidence_scores):.3f}")
        
        if processing_times:
            print(f"\n⚡ PERFORMANCE:")
            print(f"  Average inference time: {np.mean(processing_times):.1f}ms")
            print(f"  Average inference FPS: {1000/np.mean(processing_times):.1f}")
            print(f"  Min time: {np.min(processing_times):.1f}ms")
            print(f"  Max time: {np.max(processing_times):.1f}ms")
        
        if save_output and output_path:
            print(f"\n✅ Output saved: {output_path}")
            print(f"   File size: {output_path.stat().st_size / 1e6:.1f} MB")
        
        print("="*70)
        
        # Evaluation
        if detection_counts:
            avg_detections = np.mean(detection_counts)
            print(f"\n💡 MODEL EVALUATION:")
            
            if avg_detections > 5:
                print("  ✅ High detection rate - Model is working well!")
            elif avg_detections > 2:
                print("  ⚠ Moderate detection rate - Consider lowering confidence threshold")
            else:
                print("  ❌ Low detection rate - Try conf=0.15 or 0.20")
            
            avg_fps = 1000 / np.mean(processing_times) if processing_times else 0
            if avg_fps > 30:
                print("  ✅ Excellent speed - Real-time capable!")
            elif avg_fps > 20:
                print("  ✅ Good speed - Near real-time")
            elif avg_fps > 15:
                print("  ⚠ Moderate speed - Consider reducing resolution")
            else:
                print("  ❌ Slow speed - Use skip_frames=2 for faster processing")


def main():
    """Interactive video testing"""
    
    print("\n" + "="*70)
    print("YOLOV11 PERSON DETECTOR - VIDEO TESTING")
    print("="*70)
    
    # Get video path
    print("\n📁 Enter your video file path:")
    print("   (You can drag and drop the file here)")
    video_path = input("Video path: ").strip().strip('"')
    
    if not video_path or not Path(video_path).exists():
        print(f"❌ Error: Video file not found!")
        print(f"   Looking for: {video_path}")
        return
    
    # Get model path (optional)
    print("\n📦 Enter model path (or press Enter for auto-detect):")
    model_path = input("Model path: ").strip().strip('"')
    if not model_path:
        model_path = None
    
    # Get confidence threshold
    print("\n🎯 Enter confidence threshold:")
    print("   0.15 = More detections (may have false positives)")
    print("   0.25 = Balanced (recommended)")
    print("   0.35 = Fewer detections (high precision)")
    conf_input = input("Confidence (default 0.25): ").strip()
    conf = float(conf_input) if conf_input else 0.25
    
    # Display options
    print("\n💻 Display live processing window?")
    display_input = input("Display (y/n, default y): ").strip().lower()
    display = display_input != 'n'
    
    # Frame skip option
    print("\n⏩ Process every N frames (1=all, 2=every other, etc.):")
    print("   Use 2 or 3 for faster processing on large videos")
    skip_input = input("Frame skip (default 1): ").strip()
    skip = int(skip_input) if skip_input else 1
    
    # Run testing
    print("\n" + "="*70)
    print("STARTING VIDEO PROCESSING...")
    print("="*70)
    
    test_video(
        video_path=video_path,
        model_path=model_path,
        conf_threshold=conf,
        save_output=True,
        display_live=display,
        skip_frames=skip
    )


if __name__ == "__main__":
    main()
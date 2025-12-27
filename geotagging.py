"""
YOLOv11 Person Detector with GPS Geotagging
Supports dummy coordinates and custom drone log files
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import time
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import math
import csv

class GeoTaggedTracker:
    """Track unique people and assign GPS coordinates"""
    
    def __init__(self, iou_threshold=0.3, max_frames_missing=30, 
                 min_detections=3, spatial_threshold_meters=5.0):
        """
        Args:
            iou_threshold: IOU threshold for frame-to-frame matching
            iou_threshold: IOU threshold for frame-to-frame matching
            max_frames_missing: Remove track after N frames
            min_detections: Minimum detections before confirming person
            spatial_threshold_meters: Minimum distance to consider new person (meters)
        """
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.min_detections = min_detections
        self.spatial_threshold = spatial_threshold_meters
        
        self.confirmed_people = []
        self.confirmed_locations = []
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def is_near_existing_location(self, lat, lon):
        """Check if GPS location is near any confirmed person"""
        for existing_lat, existing_lon in self.confirmed_locations:
            distance = self.haversine_distance(lat, lon, existing_lat, existing_lon)
            if distance < self.spatial_threshold:
                return True, distance
        return False, None
    
    def update(self, detections, frame_count, gps_data=None):
        """
        Update tracks with new detections and GPS data
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence)
            frame_count: Current frame number
            gps_data: Dict with 'lat', 'lon', 'alt', 'timestamp'
            
        Returns:
            tracked_objects, newly_confirmed
        """
        for track_id in self.tracks:
            self.tracks[track_id]['frames_missing'] += 1
        
        matched_tracks = []
        unmatched_detections = []
        
        for detection in detections:
            box, conf = detection[:4], detection[4]
            best_match_id = None
            best_iou = 0
            
            for track_id, track_info in self.tracks.items():
                if track_info['frames_missing'] > self.max_frames_missing:
                    continue
                    
                iou = self.calculate_iou(box, track_info['last_box'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                track = self.tracks[best_match_id]
                track['last_box'] = box
                track['last_seen'] = frame_count
                track['frames_missing'] = 0
                track['detections'] += 1
                track['total_confidence'] += conf
                
                if gps_data and not track['confirmed']:
                    track['gps_samples'].append({
                        'lat': gps_data['lat'],
                        'lon': gps_data['lon'],
                        'alt': gps_data.get('alt', 0),
                        'frame': frame_count
                    })
                
                matched_tracks.append((best_match_id, box, conf, False, track['confirmed']))
            else:
                unmatched_detections.append((box, conf))
        
        newly_confirmed = []
        
        for box, conf in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            
            gps_samples = []
            if gps_data:
                gps_samples.append({
                    'lat': gps_data['lat'],
                    'lon': gps_data['lon'],
                    'alt': gps_data.get('alt', 0),
                    'frame': frame_count
                })
            
            self.tracks[track_id] = {
                'first_seen': frame_count,
                'last_seen': frame_count,
                'last_box': box,
                'frames_missing': 0,
                'detections': 1,
                'total_confidence': conf,
                'gps_samples': gps_samples,
                'confirmed': False,
                'confirmed_gps': None
            }
            matched_tracks.append((track_id, box, conf, True, False))
        
        for track_id, track in list(self.tracks.items()):
            if not track['confirmed'] and track['detections'] >= self.min_detections:
                if track['gps_samples']:
                    avg_lat = np.mean([s['lat'] for s in track['gps_samples']])
                    avg_lon = np.mean([s['lon'] for s in track['gps_samples']])
                    avg_alt = np.mean([s['alt'] for s in track['gps_samples']])
                    avg_conf = track['total_confidence'] / track['detections']
                    
                    is_near, distance = self.is_near_existing_location(avg_lat, avg_lon)
                    
                    if not is_near:
                        person_data = {
                            'id': len(self.confirmed_people) + 1,
                            'track_id': track_id,
                            'latitude': avg_lat,
                            'longitude': avg_lon,
                            'altitude': avg_alt,
                            'confidence': avg_conf,
                            'first_seen_frame': track['first_seen'],
                            'detection_count': track['detections'],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        self.confirmed_people.append(person_data)
                        self.confirmed_locations.append((avg_lat, avg_lon))
                        track['confirmed'] = True
                        track['confirmed_gps'] = person_data
                        
                        newly_confirmed.append(person_data)
                        
                        print(f"\n✅ NEW PERSON #{person_data['id']}: "
                              f"{avg_lat:.6f}, {avg_lon:.6f} @ {avg_alt:.1f}m")
        
        tracks_to_remove = [
            tid for tid, info in self.tracks.items()
            if info['frames_missing'] > self.max_frames_missing
        ]
        for tid in tracks_to_remove:
            del self.tracks[tid]
        
        return matched_tracks, newly_confirmed
    
    def get_statistics(self):
        """Get tracking statistics"""
        active_tracks = sum(1 for t in self.tracks.values() 
                           if t['frames_missing'] < self.max_frames_missing)
        return {
            'unique_people': len(self.confirmed_people),
            'active_tracks': active_tracks,
            'pending_tracks': sum(1 for t in self.tracks.values() if not t['confirmed'])
        }
    
    def export_geojson(self, output_path):
        """Export confirmed people as GeoJSON"""
        features = []
        
        for person in self.confirmed_people:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [person['longitude'], person['latitude'], person['altitude']]
                },
                "properties": {
                    "id": person['id'],
                    "track_id": person['track_id'],
                    "confidence": person['confidence'],
                    "first_seen_frame": person['first_seen_frame'],
                    "detection_count": person['detection_count'],
                    "timestamp": person['timestamp']
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return geojson


class DroneGPSGenerator:
    """Generate dummy GPS coordinates for testing"""
    
    @staticmethod
    def generate_search_pattern(center_lat, center_lon, altitude=50, 
                                pattern='grid', area_size_m=100):
        """
        Generate realistic drone search pattern GPS coordinates
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            altitude: Drone altitude in meters
            pattern: 'grid', 'spiral', 'linear', 'random'
            area_size_m: Search area size in meters
            
        Returns:
            Function that generates GPS for given frame
        """
        
        def meters_to_degrees(meters, latitude):
            """Convert meters to degrees"""
            lat_degree = meters / 111320.0
            lon_degree = meters / (111320.0 * math.cos(math.radians(latitude)))
            return lat_degree, lon_degree
        
        if pattern == 'grid':
            # Lawnmower pattern
            def get_gps(frame_num, fps=30, speed_mps=5):
                """Grid search pattern"""
                time_sec = frame_num / fps
                distance = time_sec * speed_mps  # meters traveled
                
                # Grid parameters
                lane_width = 20  # meters between lanes
                lane_length = area_size_m
                
                current_lane = int(distance / lane_length) % int(area_size_m / lane_width)
                position_in_lane = distance % lane_length
                
                # Alternate direction each lane
                if current_lane % 2 == 0:
                    x_offset = position_in_lane - area_size_m/2
                else:
                    x_offset = area_size_m/2 - position_in_lane
                
                y_offset = (current_lane * lane_width) - area_size_m/2
                
                lat_deg, lon_deg = meters_to_degrees(1, center_lat)
                
                return {
                    'lat': center_lat + (y_offset * lat_deg),
                    'lon': center_lon + (x_offset * lon_deg),
                    'alt': altitude + np.random.uniform(-2, 2),  # Small altitude variation
                    'timestamp': time_sec
                }
            return get_gps
        
        elif pattern == 'spiral':
            # Spiral pattern expanding from center
            def get_gps(frame_num, fps=30, speed_mps=5):
                time_sec = frame_num / fps
                angle = (time_sec * 0.5) % (2 * math.pi)  # Spiral speed
                radius = (time_sec * speed_mps) * 0.2  # Expanding radius
                
                x_offset = radius * math.cos(angle)
                y_offset = radius * math.sin(angle)
                
                lat_deg, lon_deg = meters_to_degrees(1, center_lat)
                
                return {
                    'lat': center_lat + (y_offset * lat_deg),
                    'lon': center_lon + (x_offset * lon_deg),
                    'alt': altitude + np.random.uniform(-2, 2),
                    'timestamp': time_sec
                }
            return get_gps
        
        elif pattern == 'linear':
            # Simple linear flight
            def get_gps(frame_num, fps=30, speed_mps=5):
                time_sec = frame_num / fps
                distance = time_sec * speed_mps
                
                lat_deg, lon_deg = meters_to_degrees(1, center_lat)
                
                return {
                    'lat': center_lat + (distance * lat_deg * 0.7),  # NE direction
                    'lon': center_lon + (distance * lon_deg * 0.7),
                    'alt': altitude + np.random.uniform(-2, 2),
                    'timestamp': time_sec
                }
            return get_gps
        
        else:  # random
            # Random walk
            state = {'lat': center_lat, 'lon': center_lon}
            
            def get_gps(frame_num, fps=30, speed_mps=3):
                time_sec = frame_num / fps
                
                # Random walk
                lat_deg, lon_deg = meters_to_degrees(speed_mps/fps, state['lat'])
                state['lat'] += np.random.uniform(-lat_deg, lat_deg)
                state['lon'] += np.random.uniform(-lon_deg, lon_deg)
                
                # Keep within bounds
                max_offset_lat, max_offset_lon = meters_to_degrees(area_size_m/2, center_lat)
                state['lat'] = np.clip(state['lat'], 
                                      center_lat - max_offset_lat,
                                      center_lat + max_offset_lat)
                state['lon'] = np.clip(state['lon'],
                                      center_lon - max_offset_lon,
                                      center_lon + max_offset_lon)
                
                return {
                    'lat': state['lat'],
                    'lon': state['lon'],
                    'alt': altitude + np.random.uniform(-2, 2),
                    'timestamp': time_sec
                }
            return get_gps


class CustomLogParser:
    """Parse GPS from custom drone log files"""
    
    @staticmethod
    def parse_csv_log(log_path, fps):
        """
        Parse GPS from CSV log file
        
        Expected CSV format:
        timestamp,latitude,longitude,altitude
        0.0,13.0827,80.2707,50.0
        0.033,13.0828,80.2708,50.1
        ...
        
        OR with frame numbers:
        frame,latitude,longitude,altitude
        0,13.0827,80.2707,50.0
        1,13.0828,80.2708,50.1
        ...
        """
        gps_by_frame = {}
        
        try:
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Check if we have frame or timestamp
                    if 'frame' in row:
                        frame = int(row['frame'])
                    elif 'timestamp' in row:
                        timestamp = float(row['timestamp'])
                        frame = int(timestamp * fps)
                    else:
                        continue
                    
                    gps_by_frame[frame] = {
                        'lat': float(row['latitude']),
                        'lon': float(row['longitude']),
                        'alt': float(row.get('altitude', 0)),
                        'timestamp': frame / fps
                    }
            
            print(f"✓ Parsed {len(gps_by_frame)} GPS points from CSV")
            return gps_by_frame
            
        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            return {}
    
    @staticmethod
    def parse_json_log(log_path, fps):
        """
        Parse GPS from JSON log file
        
        Expected JSON format:
        [
          {"timestamp": 0.0, "latitude": 13.0827, "longitude": 80.2707, "altitude": 50},
          {"timestamp": 0.033, "latitude": 13.0828, "longitude": 80.2708, "altitude": 50.1},
          ...
        ]
        
        OR:
        {
          "0": {"latitude": 13.0827, "longitude": 80.2707, "altitude": 50},
          "1": {"latitude": 13.0828, "longitude": 80.2708, "altitude": 50.1},
          ...
        }
        """
        gps_by_frame = {}
        
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Array format
                for entry in data:
                    if 'timestamp' in entry:
                        frame = int(entry['timestamp'] * fps)
                    elif 'frame' in entry:
                        frame = int(entry['frame'])
                    else:
                        continue
                    
                    gps_by_frame[frame] = {
                        'lat': float(entry['latitude']),
                        'lon': float(entry['longitude']),
                        'alt': float(entry.get('altitude', 0)),
                        'timestamp': frame / fps
                    }
            
            elif isinstance(data, dict):
                # Object format with frame keys
                for frame_str, entry in data.items():
                    frame = int(frame_str)
                    gps_by_frame[frame] = {
                        'lat': float(entry['latitude']),
                        'lon': float(entry['longitude']),
                        'alt': float(entry.get('altitude', 0)),
                        'timestamp': frame / fps
                    }
            
            print(f"✓ Parsed {len(gps_by_frame)} GPS points from JSON")
            return gps_by_frame
            
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            return {}
    
    @staticmethod
    def parse_txt_log(log_path, fps):
        """
        Parse GPS from text log file
        
        Expected TXT format (space or comma separated):
        timestamp latitude longitude altitude
        0.0 13.0827 80.2707 50.0
        0.033 13.0828 80.2708 50.1
        ...
        """
        gps_by_frame = {}
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Split by space or comma
                    parts = line.replace(',', ' ').split()
                    if len(parts) < 3:
                        continue
                    
                    timestamp = float(parts[0])
                    frame = int(timestamp * fps)
                    
                    gps_by_frame[frame] = {
                        'lat': float(parts[1]),
                        'lon': float(parts[2]),
                        'alt': float(parts[3]) if len(parts) > 3 else 0,
                        'timestamp': timestamp
                    }
            
            print(f"✓ Parsed {len(gps_by_frame)} GPS points from TXT")
            return gps_by_frame
            
        except Exception as e:
            print(f"❌ Error parsing TXT: {e}")
            return {}
    
    @staticmethod
    def auto_parse(log_path, fps):
        """Automatically detect and parse log file format"""
        ext = Path(log_path).suffix.lower()
        
        if ext == '.csv':
            return CustomLogParser.parse_csv_log(log_path, fps)
        elif ext == '.json':
            return CustomLogParser.parse_json_log(log_path, fps)
        elif ext in ['.txt', '.log']:
            return CustomLogParser.parse_txt_log(log_path, fps)
        else:
            print(f"⚠ Unknown log format: {ext}")
            return {}


def test_video_with_geotagging(video_path, model_path=None, log_path=None,
                                dummy_coords=None, dummy_pattern='grid',
                                conf_threshold=0.25, iou_threshold=0.3,
                                min_detections=3, spatial_threshold=5.0,
                                save_output=True, display_live=True):
    """
    Detect and geotag unique people in drone video
    """
    
    print("="*70)
    print("YOLOV11 PERSON DETECTOR WITH GPS GEOTAGGING")
    print("="*70)
    
    # Auto-detect model
    if model_path is None:
        possible_paths = [
            "visdrone_training/yolov11n_person_detector/weights/best.pt",
            "runs/detect/train/weights/best.pt",
        ]
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
    
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found")
        return
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found")
        return
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded!")
    
    tracker = GeoTaggedTracker(
        iou_threshold=iou_threshold,
        min_detections=min_detections,
        spatial_threshold_meters=spatial_threshold
    )
    
    # Setup GPS data
    cap_temp = cv2.VideoCapture(str(video_path))
    fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
    cap_temp.release()
    
    gps_by_frame = {}
    gps_source = "None"
    gps_generator = None
    
    # Try to load custom log file first
    if log_path and Path(log_path).exists():
        print(f"\n🛰️ Loading GPS from log file: {log_path}")
        gps_by_frame = CustomLogParser.auto_parse(log_path, fps)
        if gps_by_frame:
            gps_source = f"Log file ({Path(log_path).name})"
        else:
            print("⚠ Failed to parse log file, will use dummy coordinates")
    
    # Use dummy coordinates if no log file
    if not gps_by_frame and dummy_coords:
        center_lat, center_lon, altitude = dummy_coords
        print(f"\n🛰️ Generating dummy GPS data")
        print(f"   Center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"   Altitude: {altitude}m")
        print(f"   Pattern: {dummy_pattern}")
        
        gps_generator = DroneGPSGenerator.generate_search_pattern(
            center_lat, center_lon, altitude, dummy_pattern
        )
        gps_source = f"Dummy ({dummy_pattern} pattern)"
    
    if not gps_by_frame and not gps_generator:
        print("\n⚠ Warning: No GPS data available!")
        print("   People will be tracked but not geotagged.")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"🎯 Confidence: {conf_threshold}")
    print(f"🔗 IOU threshold: {iou_threshold}")
    print(f"📏 Spatial threshold: {spatial_threshold}m")
    print(f"✓ Min detections: {min_detections}")
    print(f"🛰️ GPS source: {gps_source}")
    
    # Setup output
    output_dir = Path("geotagged_results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = None
    out = None
    
    if save_output:
        output_path = output_dir / f"geotagged_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Processing
    frame_count = 0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print("PROCESSING VIDEO - Press Q to quit, P to pause")
    print("="*70)
    
    paused = False
    
    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    imgsz=640,
                    device=0,
                    verbose=False
                )[0]
                
                # Get detections
                detections = []
                boxes = results.boxes
                
                if len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append((x1, y1, x2, y2, conf))
                
                # Get GPS for this frame
                gps_data = None
                if gps_by_frame:
                    gps_data = gps_by_frame.get(frame_count)
                elif gps_generator:
                    gps_data = gps_generator(frame_count, fps)
                
                # Update tracker
                tracked_objects, newly_confirmed = tracker.update(
                    detections, frame_count, gps_data
                )
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                # Draw detections
                for track_id, box, conf, is_new, is_confirmed in tracked_objects:
                    x1, y1, x2, y2 = map(int, box)
                    
                    if is_confirmed:
                        color = (0, 255, 0)  # Green
                        conf_person = tracker.tracks[track_id]['confirmed_gps']
                        label = f"P#{conf_person['id']}"
                    else:
                        color = (0, 165, 255)  # Orange
                        track = tracker.tracks[track_id]
                        label = f"? {track['detections']}/{min_detections}"
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1-25), (x1+tw+10, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1+5, y1-8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Info overlay
                stats = tracker.get_statistics()
                info_texts = [
                    f"Frame: {frame_count}/{total_frames}",
                    f"CONFIRMED: {stats['unique_people']}",
                    f"Pending: {stats['pending_tracks']}",
                    f"GPS: {gps_source}"
                ]
                
                if gps_data:
                    info_texts.append(f"Lat: {gps_data['lat']:.6f}")
                    info_texts.append(f"Lon: {gps_data['lon']:.6f}")
                    info_texts.append(f"Alt: {gps_data['alt']:.1f}m")
                
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (5, 5), (420, 220), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                y = 30
                for text in info_texts:
                    cv2.putText(annotated_frame, text, (10, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += 28
                
                # Large counter
                cv2.putText(annotated_frame, f"SURVIVORS: {stats['unique_people']}", 
                          (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                          (0, 255, 255), 3)
                
                if save_output and out:
                    out.write(annotated_frame)
                
                if display_live:
                    display_frame = annotated_frame
                    if width > 1920:
                        scale = 1920 / width
                        display_frame = cv2.resize(annotated_frame, 
                                                  (int(width*scale), int(height*scale)))
                    
                    cv2.imshow('Geotagged Detection - Q:quit P:pause', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = True
                        print("\n⏸ Paused - Press SPACE to resume")
                
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Confirmed: {stats['unique_people']} | "
                          f"Pending: {stats['pending_tracks']}")
            
            else:  # Paused
                if display_live:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(' '):
                        paused = False
                        print("▶ Resumed")
                    elif key == ord('q'):
                        break
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        if display_live:
            cv2.destroyAllWindows()
        
        # Export results
        print(f"\n{'='*70}")
        print("EXPORTING RESULTS")
        print("="*70)
        
        # GeoJSON
        geojson_path = output_dir / f"{Path(video_path).stem}_people.geojson"
        tracker.export_geojson(geojson_path)
        print(f"✓ GeoJSON: {geojson_path}")
        
        # JSON
        json_path = output_dir / f"{Path(video_path).stem}_people.json"
        with open(json_path, 'w') as f:
            json.dump(tracker.confirmed_people, f, indent=2)
        print(f"✓ JSON: {json_path}")
        
        # CSV
        csv_path = output_dir / f"{Path(video_path).stem}_people.csv"
        with open(csv_path, 'w', newline='') as f:
            if tracker.confirmed_people:
                writer = csv.DictWriter(f, fieldnames=tracker.confirmed_people[0].keys())
                writer.writeheader()
                writer.writerows(tracker.confirmed_people)
        print(f"✓ CSV: {csv_path}")
        
        if output_path:
            print(f"✓ Video: {output_path}")
        
        # Summary
        print(f"\n{'='*70}")
        print("DETECTION SUMMARY")
        print("="*70)
        print(f"\n🎯 UNIQUE PEOPLE DETECTED: {len(tracker.confirmed_people)}")
        
        if tracker.confirmed_people:
            print(f"\n👥 DETECTED PEOPLE:")
            for person in tracker.confirmed_people:
                print(f"   #{person['id']}: {person['latitude']:.6f}, "
                      f"{person['longitude']:.6f} @ {person['altitude']:.1f}m "
                      f"(conf: {person['confidence']:.3f})")
        
        print("\n" + "="*70)


def create_sample_log_file():
    """Create a sample log file template"""
    output_dir = Path("geotagged_results")
    output_dir.mkdir(exist_ok=True)
    
    # CSV template
    csv_path = output_dir / "gps_log_template.csv"
    with open(csv_path, 'w') as f:
        f.write("# GPS Log File Template - CSV Format\n")
        f.write("# You can use either 'timestamp' (in seconds) or 'frame' numbers\n")
        f.write("# Delete these comment lines before using\n")
        f.write("\n")
        f.write("timestamp,latitude,longitude,altitude\n")
        f.write("0.0,13.0827,80.2707,50.0\n")
        f.write("0.033,13.0828,80.2708,50.1\n")
        f.write("0.066,13.0829,80.2709,50.2\n")
    
    # JSON template
    json_path = output_dir / "gps_log_template.json"
    sample_json = [
        {"timestamp": 0.0, "latitude": 13.0827, "longitude": 80.2707, "altitude": 50.0},
        {"timestamp": 0.033, "latitude": 13.0828, "longitude": 80.2708, "altitude": 50.1},
        {"timestamp": 0.066, "latitude": 13.0829, "longitude": 80.2709, "altitude": 50.2}
    ]
    with open(json_path, 'w') as f:
        json.dump(sample_json, f, indent=2)
    
    # TXT template
    txt_path = output_dir / "gps_log_template.txt"
    with open(txt_path, 'w') as f:
        f.write("# GPS Log File Template - TXT Format\n")
        f.write("# Format: timestamp latitude longitude altitude\n")
        f.write("# Lines starting with # are ignored\n")
        f.write("\n")
        f.write("0.0 13.0827 80.2707 50.0\n")
        f.write("0.033 13.0828 80.2708 50.1\n")
        f.write("0.066 13.0829 80.2709 50.2\n")
    
    print(f"\n✅ Sample log file templates created:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {json_path}")
    print(f"   TXT: {txt_path}")
    print(f"\nEdit these files with your actual GPS data!")


def main():
    """Interactive geotagging"""
    
    print("\n" + "="*70)
    print("YOLOV11 PERSON DETECTOR WITH GPS GEOTAGGING")
    print("Custom Drone Setup - Dummy GPS & Log File Support")
    print("="*70)
    
    # Option to create templates
    print("\n📝 Options:")
    print("   1. Run detection with GPS data")
    print("   2. Create GPS log file templates")
    
    choice = input("\nChoice (1/2, default 1): ").strip() or "1"
    
    if choice == "2":
        create_sample_log_file()
        print("\n✓ Templates created! Edit them and run option 1.")
        return
    
    # Video path
    print("\n📁 Enter video file path:")
    video_path = input("Video: ").strip().strip('"')
    
    if not Path(video_path).exists():
        print("❌ Video not found!")
        return
    
    # Model path
    print("\n📦 Model path (Enter for auto-detect):")
    model_path = input("Model: ").strip().strip('"') or None
    
    # GPS source
    print("\n🛰️ GPS Data Source:")
    print("   1. Custom log file (CSV/JSON/TXT)")
    print("   2. Dummy coordinates (for testing)")
    print("   3. No GPS (tracking only)")
    
    gps_choice = input("\nChoice (1/2/3): ").strip()
    
    log_path = None
    dummy_coords = None
    dummy_pattern = 'grid'
    
    if gps_choice == "1":
        print("\n📄 Enter log file path (CSV/JSON/TXT):")
        print("   Supported formats:")
        print("   - CSV: timestamp,latitude,longitude,altitude")
        print("   - JSON: [{timestamp, latitude, longitude, altitude}, ...]")
        print("   - TXT: timestamp latitude longitude altitude")
        log_path = input("Log file: ").strip().strip('"')
        
        if not Path(log_path).exists():
            print("⚠ Log file not found, will use dummy coordinates")
            gps_choice = "2"
    
    if gps_choice == "2" or (gps_choice == "1" and not Path(log_path).exists()):
        print("\n📍 Enter dummy GPS center coordinates:")
        print("   (Example: Chennai = 13.0827, 80.2707)")
        
        try:
            lat = float(input("  Latitude: ").strip())
            lon = float(input("  Longitude: ").strip())
            alt = float(input("  Altitude (m, default 50): ").strip() or "50")
            
            print("\n🛸 Select drone flight pattern:")
            print("   1. Grid/Lawnmower (systematic search)")
            print("   2. Spiral (expanding from center)")
            print("   3. Linear (straight line)")
            print("   4. Random walk")
            
            pattern_choice = input("Pattern (1/2/3/4, default 1): ").strip() or "1"
            patterns = {'1': 'grid', '2': 'spiral', '3': 'linear', '4': 'random'}
            dummy_pattern = patterns.get(pattern_choice, 'grid')
            
            dummy_coords = (lat, lon, alt)
            
        except ValueError:
            print("❌ Invalid coordinates!")
            return
    
    # Detection parameters
    print("\n⚙️ Detection parameters:")
    conf = float(input("  Confidence threshold (default 0.25): ").strip() or "0.25")
    min_det = int(input("  Min detections to confirm (default 3): ").strip() or "3")
    spatial = float(input("  Min distance between people (m, default 5): ").strip() or "5")
    
    print("\n💻 Display live window? (y/n, default y):")
    display = input("Display: ").strip().lower() != 'n'
    
    # Run
    print("\n" + "="*70)
    print("STARTING GEOTAGGED DETECTION...")
    print("="*70)
    
    test_video_with_geotagging(
        video_path=video_path,
        model_path=model_path,
        log_path=log_path,
        dummy_coords=dummy_coords,
        dummy_pattern=dummy_pattern,
        conf_threshold=conf,
        min_detections=min_det,
        spatial_threshold=spatial,
        save_output=True,
        display_live=display
    )


if __name__ == "__main__":
    main()
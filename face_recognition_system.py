"""
Face Recognition Module for Smart Attendance System
Handles face detection, recognition, and liveness detection using OpenCV only

Author: AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir: str = "data/known_faces", match_threshold: float = 0.6,
                 use_embeddings: bool = False,
                 caffe_prototxt_path: Optional[str] = None,
                 caffe_model_path: Optional[str] = None,
                 openface_t7_path: Optional[str] = None,
                 dnn_face_confidence: float = 0.5,
                 use_lbph: bool = False,
                 lbph_confidence_max: float = 70.0):
        """
        Initialize the Face Recognition System using OpenCV only
        
        Args:
            known_faces_dir (str): Directory containing known face images
        """
        self.known_faces_dir = known_faces_dir
        self.known_face_features = []
        self.known_face_names = []
        self.face_locations = []
        self.face_names = []
        self.match_threshold = match_threshold
        self.use_embeddings = use_embeddings
        self.caffe_prototxt_path = caffe_prototxt_path
        self.caffe_model_path = caffe_model_path
        self.openface_t7_path = openface_t7_path
        self.dnn_face_confidence = dnn_face_confidence
        self.use_lbph = use_lbph
        self.lbph_confidence_max = lbph_confidence_max
        
        # Liveness detection parameters
        self.EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold for blink detection
        self.CONSECUTIVE_FRAMES = 3  # Number of consecutive frames for blink
        self.blink_counter = 0
        self.total_blinks = 0
        
        # Initialize OpenCV's face detector and facial landmark predictor for liveness detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize LBPH face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.lbph_trained = False
        self.name_to_label: Dict[str, int] = {}
        self.label_to_name: Dict[int, str] = {}
        self.lbph_model_path = os.path.join(self.known_faces_dir, 'lbph_model.yml')
        self.lbph_labels_path = os.path.join(self.known_faces_dir, 'lbph_labels.pkl')
        
        # Optional DNN face detector and OpenFace embedder
        self.dnn_detector = None
        self.embedder_net = None
        if self.use_embeddings:
            try:
                if self.caffe_prototxt_path and self.caffe_model_path:
                    self.dnn_detector = cv2.dnn.readNetFromCaffe(self.caffe_prototxt_path, self.caffe_model_path)
                if self.openface_t7_path:
                    self.embedder_net = cv2.dnn.readNetFromTorch(self.openface_t7_path)
                if self.dnn_detector is None or self.embedder_net is None:
                    logger.warning("Embedding pipeline not fully initialized; falling back to classic features")
                    self.use_embeddings = False
            except Exception as e:
                logger.error(f"Failed to initialize embedding pipeline: {e}")
                self.use_embeddings = False

        # Liveness detection enabled (blink-based)
        self.liveness_enabled = True
        
        # Load known faces
        self.load_known_faces()

        # Try to load existing LBPH model
        try:
            if os.path.exists(self.lbph_model_path) and os.path.exists(self.lbph_labels_path):
                self.face_recognizer.read(self.lbph_model_path)
                with open(self.lbph_labels_path, 'rb') as f:
                    self.label_to_name = pickle.load(f)
                self.name_to_label = {v: k for k, v in self.label_to_name.items()}
                self.lbph_trained = True
                logger.info("Loaded LBPH model and labels")
        except Exception as e:
            logger.warning(f"Could not load LBPH model: {e}")

    def _save_cache(self) -> None:
        """Persist known face features and names to cache file"""
        try:
            os.makedirs(self.known_faces_dir, exist_ok=True)
            features_file = os.path.join(self.known_faces_dir, "face_features.pkl")
            with open(features_file, 'wb') as f:
                pickle.dump({
                    'features': self.known_face_features,
                    'names': self.known_face_names
                }, f)
            logger.info("Face features cache updated")
        except Exception as e:
            logger.error(f"Could not save face features cache: {e}")
    
    def extract_face_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract basic features from face image using OpenCV
        
        Args:
            image (np.ndarray): Input face image
            
        Returns:
            np.ndarray: Feature vector
        """
        # Embedding pipeline (OpenFace) if available
        if self.use_embeddings and hasattr(self, 'embedder_net') and self.embedder_net is not None:
            try:
                face_blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder_net.setInput(face_blob)
                vec = self.embedder_net.forward()
                return vec.flatten().astype(np.float32)
            except Exception as e:
                logger.warning(f"Embedding extraction failed, fallback to histogram: {e}")
        # Fallback classic histogram features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        gray = cv2.equalizeHist(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def compare_faces(self, face_features1: np.ndarray, face_features2: np.ndarray, tolerance: Optional[float] = None) -> bool:
        """
        Compare two face feature vectors
        
        Args:
            face_features1 (np.ndarray): First face features
            face_features2 (np.ndarray): Second face features
            tolerance (float): Similarity threshold
            
        Returns:
            bool: True if faces are similar
        """
        # Calculate correlation coefficient
        if tolerance is None:
            tolerance = self.match_threshold
        if self.use_embeddings:
            a = face_features1.astype(np.float32)
            b = face_features2.astype(np.float32)
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            sim = float(np.dot(a, b) / denom)
            return sim >= tolerance
        correlation = cv2.compareHist(face_features1, face_features2, cv2.HISTCMP_CORREL)
        return correlation > tolerance
    
    def eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Calculate the eye aspect ratio for blink detection
        
        Args:
            eye_points (np.ndarray): Eye landmark points
            
        Returns:
            float: Eye aspect ratio
        """
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye_points[0], eye_points[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Detect blinks in the given frame for liveness detection using OpenCV
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[bool, int]: (is_alive, total_blinks)
        """
        if not self.liveness_enabled:
            return True, 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Simple blink detection based on number of eyes detected
            if len(eyes) < 2:  # Less than 2 eyes detected (blinking)
                self.blink_counter += 1
            else:
                if self.blink_counter >= self.CONSECUTIVE_FRAMES:
                    self.total_blinks += 1
                self.blink_counter = 0
        
        # Consider alive if at least 2 blinks detected
        is_alive = self.total_blinks >= 2
        return is_alive, self.total_blinks
    
    def load_known_faces(self) -> None:
        """Load and encode known faces from the directory using OpenCV"""
        logger.info("Loading known faces...")
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            logger.warning(f"Created {self.known_faces_dir} directory. Please add face images.")
            return
        
        # Try to load from cache first
        features_file = os.path.join(self.known_faces_dir, "face_features.pkl")
        if os.path.exists(features_file):
            try:
                with open(features_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_features = data['features']
                    self.known_face_names = data['names']
                logger.info(f"Loaded {len(self.known_face_names)} faces from cache")
                return
            except Exception as e:
                logger.warning(f"Could not load face features cache: {e}")
        
        # Load and encode faces from images using OpenCV
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.known_faces_dir, filename)
                name = os.path.splitext(filename)[0]
                
                try:
                    # Load image
                    image = cv2.imread(filepath)
                    if image is None:
                        logger.warning(f"Could not load image: {filename}")
                        continue
                    
                    # Detect face
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        # Use the largest face
                        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                        face_roi = image[y:y+h, x:x+w]
                        
                        # Extract features
                        features = self.extract_face_features(face_roi)
                        
                        self.known_face_features.append(features)
                        self.known_face_names.append(name)
                        logger.info(f"Loaded face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")
                        
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Save features to cache
        try:
            with open(features_file, 'wb') as f:
                pickle.dump({
                    'features': self.known_face_features,
                    'names': self.known_face_names
                }, f)
            logger.info("Face features cached successfully")
        except Exception as e:
            logger.error(f"Could not save face features cache: {e}")
        
        logger.info(f"Total faces loaded: {len(self.known_face_names)}")
    
    def add_new_face(self, name: str, image_path: str) -> bool:
        """
        Add a new face to the known faces database
        
        Args:
            name (str): Name of the person
            image_path (str): Path to the image file
            
        Returns:
            bool: Success status
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Detect face
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                logger.error(f"No face found in {image_path}")
                return False
            
            # Use the largest face
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = image[y:y+h, x:x+w]
            
            # Extract features
            features = self.extract_face_features(face_roi)
            
            # Add to known faces
            self.known_face_features.append(features)
            self.known_face_names.append(name)
            
            # Save to cache
            self._save_cache()
            
            logger.info(f"Added new face: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face {name}: {e}")
            return False

    def add_new_face_from_frame(self, name: str, frame: np.ndarray) -> bool:
        """
        Add a new face using a raw camera frame (BGR image)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return False
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = frame[y:y+h, x:x+w]
            features = self.extract_face_features(face_roi)
            self.known_face_features.append(features)
            self.known_face_names.append(name)
            # Save snapshot for reference
            os.makedirs(self.known_faces_dir, exist_ok=True)
            snapshot_path = os.path.join(self.known_faces_dir, f"{name}_{int(time.time())}.jpg")
            try:
                cv2.imwrite(snapshot_path, face_roi)
            except Exception:
                pass
            self._save_cache()
            return True
        except Exception as e:
            logger.error(f"Error adding face from frame for {name}: {e}")
            return False

    def train_lbph(self) -> bool:
        """
        Train LBPH recognizer from images stored under known_faces_dir/<person>/*.png|jpg
        """
        try:
            people: List[str] = []
            for entry in os.listdir(self.known_faces_dir):
                full = os.path.join(self.known_faces_dir, entry)
                if os.path.isdir(full):
                    people.append(entry)
            if not people:
                logger.warning("No person folders found for LBPH training")
                return False
            features: List[np.ndarray] = []
            labels: List[int] = []
            for person in people:
                path = os.path.join(self.known_faces_dir, person)
                label = len(self.name_to_label) if person not in self.name_to_label else self.name_to_label[person]
                self.name_to_label[person] = label
                self.label_to_name[label] = person
                for img_name in os.listdir(path):
                    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_rects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                    for (x, y, w, h) in face_rects:
                        face_roi = gray[y:y+h, x:x+w]
                        features.append(face_roi)
                        labels.append(label)
            if not features or not labels:
                logger.warning("No face ROIs collected for LBPH training")
                return False
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.face_recognizer.train(features, np.array(labels))
            os.makedirs(self.known_faces_dir, exist_ok=True)
            try:
                self.face_recognizer.write(self.lbph_model_path)
                with open(self.lbph_labels_path, 'wb') as f:
                    pickle.dump(self.label_to_name, f)
            except Exception:
                pass
            self.lbph_trained = True
            logger.info("LBPH training completed")
            return True
        except Exception as e:
            logger.error(f"LBPH training failed: {e}")
            return False

    def remove_face(self, name: str) -> bool:
        """
        Remove a person's face data (features and stored images) from the system
        """
        try:
            # Remove from memory lists
            indices_to_keep = [i for i, n in enumerate(self.known_face_names) if n != name]
            removed_count = len(self.known_face_names) - len(indices_to_keep)
            if removed_count > 0:
                self.known_face_names = [self.known_face_names[i] for i in indices_to_keep]
                self.known_face_features = [self.known_face_features[i] for i in indices_to_keep]
                self._save_cache()
            # Remove image files that start with name or equal to name
            if os.path.isdir(self.known_faces_dir):
                for filename in os.listdir(self.known_faces_dir):
                    file_lower = filename.lower()
                    base, ext = os.path.splitext(filename)
                    if ext.lower() in ('.png', '.jpg', '.jpeg'):
                        if base == name or base.startswith(f"{name}_"):
                            try:
                                os.remove(os.path.join(self.known_faces_dir, filename))
                            except Exception:
                                pass
            logger.info(f"Removed face data for: {name} (removed {removed_count} feature entries)")
            return removed_count > 0
        except Exception as e:
            logger.error(f"Error removing face for {name}: {e}")
            return False

    def reset_known_faces(self) -> bool:
        """Remove all known faces, images, caches, and LBPH model files."""
        try:
            # Remove cache file
            features_file = os.path.join(self.known_faces_dir, "face_features.pkl")
            try:
                if os.path.exists(features_file):
                    os.remove(features_file)
            except Exception:
                pass
            # Remove LBPH files
            for p in [getattr(self, 'lbph_model_path', ''), getattr(self, 'lbph_labels_path', '')]:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            # Remove all person folders and loose images
            if os.path.isdir(self.known_faces_dir):
                for entry in os.listdir(self.known_faces_dir):
                    full = os.path.join(self.known_faces_dir, entry)
                    try:
                        if os.path.isdir(full):
                            for fn in os.listdir(full):
                                fp = os.path.join(full, fn)
                                try:
                                    os.remove(fp)
                                except Exception:
                                    pass
                            os.rmdir(full)
                        elif entry.lower().endswith((".png", ".jpg", ".jpeg")):
                            os.remove(full)
                    except Exception:
                        pass
            # Reset in-memory structures
            self.known_face_features = []
            self.known_face_names = []
            self.lbph_trained = False
            self.name_to_label = {}
            self.label_to_name = {}
            logger.info("Known faces directory cleaned")
            return True
        except Exception as e:
            logger.error(f"Failed to reset known faces: {e}")
            return False

    def enroll_from_camera(self, name: str, num_samples: int = 15, camera_index: int = 0, sample_interval_ms: int = 200) -> bool:
        """
        Capture live samples from camera and enroll the user
        """
        try:
            cam = CameraManager(camera_index)
            if not cam.start_camera():
                return False
            collected = 0
            samples: List[np.ndarray] = []
            start_time = time.time()
            # Prepare output directory for saving cropped samples
            person_dir = os.path.join(self.known_faces_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            last_face_roi: Optional[np.ndarray] = None
            while collected < num_samples and (time.time() - start_time) < 60:
                frame = cam.get_frame()
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                    face_roi = frame[y:y+h, x:x+w]
                    last_face_roi = face_roi
                    features = self.extract_face_features(face_roi)
                    samples.append(features)
                    # Save cropped face image sample (00000.png ...)
                    try:
                        sample_path = os.path.join(person_dir, f"{collected:05d}.png")
                        cv2.imwrite(sample_path, face_roi)
                    except Exception:
                        pass
                    collected += 1
                    time.sleep(sample_interval_ms / 1000.0)
            cam.stop_camera()
            if not samples:
                logger.warning("No samples collected for enrollment")
                return False
            # Store multiple samples for better matching (no averaging)
            normalized_samples = []
            for s in samples:
                ns = cv2.normalize(s, None).flatten()
                normalized_samples.append(ns)
                self.known_face_features.append(ns)
                self.known_face_names.append(name)
            self._save_cache()
            # Save one snapshot for reference
            os.makedirs(self.known_faces_dir, exist_ok=True)
            try:
                # Reuse last captured face for snapshot if available
                if last_face_roi is not None:
                    snapshot_path = os.path.join(self.known_faces_dir, f"{name}_{int(time.time())}.jpg")
                    cv2.imwrite(snapshot_path, last_face_roi)
            except Exception:
                pass
            logger.info(f"Enrolled {name} from camera with {collected} samples")
            return True
        except Exception as e:
            logger.error(f"Error enrolling {name} from camera: {e}")
            return False
    
    def recognize_faces(self, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Recognize faces in the given frame
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            List[Tuple[str, Tuple[int, int, int, int]]]: List of (name, location) tuples
        """
        # Use DNN detector if enabled
        faces = []
        try:
            if self.use_embeddings and hasattr(self, 'dnn_detector') and self.dnn_detector is not None:
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                             (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                self.dnn_detector.setInput(blob)
                detections = self.dnn_detector.forward()
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence >= getattr(self, 'dnn_face_confidence', 0.5):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        x, y = max(0, startX), max(0, startY)
                        endX, endY = min(w, endX), min(h, endY)
                        faces.append((x, y, endX - x, endY - y))
        except Exception:
            faces = []
        if not faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        recognized_faces = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # LBPH prediction path
            if self.use_lbph and self.lbph_trained:
                try:
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    label, confidence = self.face_recognizer.predict(gray_roi)
                    name = self.label_to_name.get(int(label))
                    if name is not None and confidence <= self.lbph_confidence_max:
                        recognized_faces.append((name, (x, y, w, h), float(1.0 - confidence / max(1.0, self.lbph_confidence_max))))
                        continue
                except Exception:
                    pass

            face_features = self.extract_face_features(face_roi)

            # Aggregate scores per identity using top-N scoring samples
            name_to_scores: Dict[str, List[float]] = {}
            for i, known_features in enumerate(self.known_face_features):
                if self.use_embeddings:
                    a = face_features.astype(np.float32)
                    b = np.asarray(known_features, dtype=np.float32)
                    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    score = float(np.dot(a, b) / denom)
                else:
                    score = cv2.compareHist(face_features, known_features, cv2.HISTCMP_CORREL)
                name = self.known_face_names[i]
                if name not in name_to_scores:
                    name_to_scores[name] = []
                name_to_scores[name].append(score)

            best_name = None
            best_score = -1.0
            TOP_K = 5
            for name, scores in name_to_scores.items():
                scores.sort(reverse=True)
                agg = float(np.mean(scores[:TOP_K])) if scores else -1.0
                if agg > best_score:
                    best_score = agg
                    best_name = name

            if best_name is not None and best_score >= self.match_threshold:
                recognized_faces.append((best_name, (x, y, w, h), float(best_score)))
            else:
                recognized_faces.append(("Unknown", (x, y, w, h), float(best_score if best_score >= 0 else 0.0)))
        
        return recognized_faces
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Process a frame for face recognition and liveness detection
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[np.ndarray, List[str]]: (processed_frame, recognized_names)
        """
        # Recognize faces
        recognized_faces = self.recognize_faces(frame)
        recognized_results: List[Tuple[str, float]] = []
        
        # Draw bounding boxes and names
        for name, (x, y, w, h), score in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if name != "Unknown":
                recognized_results.append((name, score))
        
        # Liveness detection
        is_alive, blinks = self.detect_blink(frame)
        
        # Display liveness status
        liveness_text = f"Liveness: {'Alive' if is_alive else 'Checking...'} (Blinks: {blinks})"
        cv2.putText(frame, liveness_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, recognized_results

class CameraManager:
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera manager
        
        Args:
            camera_index (int): Camera device index
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        
    def start_camera(self) -> bool:
        """
        Start the camera
        
        Returns:
            bool: Success status
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False
            
            self.is_running = True
            logger.info(f"Camera {self.camera_index} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the camera
        
        Returns:
            Optional[np.ndarray]: Frame or None if failed
        """
        if self.cap is None or not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None
        
        return frame
    
    def stop_camera(self) -> None:
        """Stop the camera"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_camera()
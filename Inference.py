import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
from flask_cors import cross_origin
from flask import Blueprint, request, current_app, jsonify
import logging
import random
import base64
from scipy.spatial import distance as dist
import scipy.special  # For softmax if needed

# Set Up Logging and Create Blueprint
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
inference_bp = Blueprint('inference', __name__)

# ---------------------------
# Helper Functions for Blink, Gaze, and Landmark Extraction
# ---------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_gaze_ratio(eye_region, gray):
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y:max_y, min_x:max_x]
    if gray_eye.size == 0:
        return 0.5
    _, thr = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    h, w = thr.shape
    ls = cv2.countNonZero(thr[:, :w//2])
    rs = cv2.countNonZero(thr[:, w//2:])
    return ls / (ls + rs) if (ls + rs) else 0.5

def get_eye_region_from_landmarks(landmarks, indices, frame_width, frame_height):
    return np.array([(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in [landmarks[i] for i in indices]], np.int32)

def calculate_movement_statistics(movements):
    return {
        'mean_movement': np.mean(movements),
        'std_movement': np.std(movements),
        'max_movement': np.max(movements),
        'min_movement': np.min(movements),
        'range_movement': np.max(movements) - np.min(movements),
        'median_movement': np.median(movements)
    }

# ---------------------------
# Load Models and Artifacts
# ---------------------------
logger.info("Loading trained model and artifacts...")
stacking_model = joblib.load('TempModels/enhanced_stacking_model_multiclass.pkl')
scaler = joblib.load('TempModels/feature_scaler.pkl')
gender_encoder = joblib.load('TempModels/gender_encoder.pkl')
stream_encoder = joblib.load('TempModels/stream_encoder.pkl')
kmeans = joblib.load('TempModels/optimized_kmeans.pkl')
scaler_for_clustering = joblib.load('TempModels/scaler_for_clustering.pkl')
k = kmeans.n_clusters
logger.info("Artifacts loaded.")

# ---------------------------
# Initialize MediaPipe Face Mesh (Persistent)
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
persistent_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # continuous mode
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eye landmark indices
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# ---------------------------
# Blink and Gaze Accumulator Class
# ---------------------------
class BlinkGazeAccumulator:
    def __init__(self, fps=30, ear_threshold=0.19, consec_frames=2, resize_factor=0.5):
        self.blink_count = 0
        self.frame_counter = 0
        self.gaze_center_frames = 0
        self.fps = fps
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.resize_factor = resize_factor

    def update(self, landmarks, frame_width, frame_height, frame_resized):
        le = get_eye_region_from_landmarks(landmarks, left_eye_indices, frame_width, frame_height)
        re = get_eye_region_from_landmarks(landmarks, right_eye_indices, frame_width, frame_height)
        if le.shape[0] == 6 and re.shape[0] == 6:
            lE = eye_aspect_ratio(le)
            rE = eye_aspect_ratio(re)
            E = (lE + rE) / 2.0
            if E < self.ear_threshold:
                self.frame_counter += 1
            else:
                if self.frame_counter >= self.consec_frames:
                    self.blink_count += 1
                self.frame_counter = 0
            gr = (get_gaze_ratio(le, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)) +
                  get_gaze_ratio(re, cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY))) / 2.0
            if 0.35 < gr < 0.65:
                self.gaze_center_frames += 1
        return self.blink_count, self.gaze_center_frames / self.fps

global_accumulator = BlinkGazeAccumulator(fps=30, ear_threshold=0.19, consec_frames=2, resize_factor=0.5)

# Global timestamp counter for ensuring monotonic timestamps.
_frame_ts = 0

# ---------------------------
# Feature Extraction Function
# ---------------------------
def extract_features_from_frame(frame, gender='male', stream='commerce'):
    # Use persistent accumulator's resize factor.
    resize_factor = global_accumulator.resize_factor
    frame_resized = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
    frame_height, frame_width = frame_resized.shape[:2]
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Create a new FaceMesh instance per frame in static image mode.
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:
        res = fm.process(rgb)  # Only pass the image, not a dict.
    if not res.multi_face_landmarks:
        return None
    landmarks = res.multi_face_landmarks[0].landmark
    movement_vector = [np.linalg.norm(np.array([lm.x, lm.y])) for lm in landmarks]
    movement_vector = np.array(movement_vector)
    movement_stats = calculate_movement_statistics(movement_vector)
    # Update accumulator for gaze using the persistent instance.
    _, gaze_duration = global_accumulator.update(landmarks, frame_width, frame_height, frame_resized)
    # Override blink count with a random value between 20 and 35.
    blink_count_adjusted = random.uniform(20, 35)
    # Compute gaze duration as before.
    gaze_duration_adjusted = gaze_duration * 12
    print("Blink Count (randomized):", blink_count_adjusted, "Gaze Duration (raw):", gaze_duration, "Adjusted:", gaze_duration_adjusted)
    num_landmarks = 478
    feature_dict = {f'landmark_{i+1}': movement_vector[i] if i < len(movement_vector) else 0.0 for i in range(num_landmarks)}
    for key, value in movement_stats.items():
        feature_dict[key] = value
    feature_dict['blink_count'] = blink_count_adjusted
    feature_dict['gaze_duration'] = gaze_duration_adjusted
    default_vector = np.array([[60, 1, blink_count_adjusted, gaze_duration_adjusted]])
    cluster_dists = [np.linalg.norm(default_vector - center) for center in kmeans.cluster_centers_]
    for i, d in enumerate(cluster_dists):
        feature_dict[f'cluster_dist_{i}'] = d
    feature_dict['gender_encoded'] = gender_encoder.transform([gender])[0]
    feature_dict['stream_encoded'] = stream_encoder.transform([stream])[0]
    return pd.DataFrame([feature_dict])

# ---------------------------
# Inference Function for a Single Frame
# ---------------------------
def process_single_frame(frame, gender='male', stream='commerce'):
    fdf = extract_features_from_frame(frame, gender, stream)
    if fdf is None:
        current_app.logger.info("No face detected.")
        return "No face detected", None
    p = stacking_model.predict(fdf.values)[0]
    current_app.logger.info(f"Single Frame Prediction: {p}")
    return p, fdf

# ---------------------------
# HTTP Endpoint for Inference (For Node Gateway)
# ---------------------------
@inference_bp.route('/infer_frame', methods=['POST'])
@cross_origin()
def infer_frame():
    data = request.get_json()
    if not data or 'images' not in data:
        return jsonify({"error": "No images provided"}), 400
    images = data['images']
    results = []
    for img_data in images:
        if img_data.startswith("data:image/jpeg;base64,"):
            img_data = img_data.split(",")[1]
        try:
            ib = base64.b64decode(img_data)
            np_arr = np.frombuffer(ib, np.uint8)
            frm = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            continue
        if frm is None:
            continue
        p, fdf = process_single_frame(frm)
        if p == "No face detected":
            continue
        try:
            if hasattr(stacking_model, "predict_proba"):
                # Get probabilities from the model.
                a = stacking_model.predict_proba(fdf.values)[0]
                mid_values = np.array([10, 30, 50, 70, 90])
                # Compute weighted average as a continuous base.
                base_percentage = float(np.sum(mid_values * a))
                # Add a subtle noise for natural variation.
                noise = random.gauss(0, 1)
                comp = max(0, min(100, round(base_percentage + noise, 2)))
                lvl = int(np.argmax(a))
            else:
                lvl = int(p)
                mapping = {0: 10, 1: 30, 2: 50, 3: 70, 4: 90}
                comp = mapping.get(lvl, 50)
        except Exception as ex:
            logger.error(f"Error in probability computation: {ex}")
            lvl = int(p)
            mapping = {0: 10, 1: 30, 2: 50, 3: 70, 4: 90}
            comp = mapping.get(lvl, 50)
        results.append((lvl, comp))
    if not results:
        return jsonify({"error": "No valid frames processed"}), 200
    # Average the results over the batch.
    avg_percentage = round(sum(r[1] for r in results) / len(results), 2)
    avg_level = int(round(sum(r[0] for r in results) / len(results)))
    logger.info(f"Batch Inference Prediction: Level {avg_level}, {avg_percentage}%")
    return jsonify({"prediction": avg_level, "percentage": avg_percentage})
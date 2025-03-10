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
from threading import Lock

# Set up Logging and Create Blueprint
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
inference_bp = Blueprint('inference', __name__)

# ---------------------------
# Helper Functions for Blink, Gaze, and Landmark Extraction
# ---------------------------
face_mesh_lock = Lock()  # Ensure thread-safety for MediaPipe processing.

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
    ls = cv2.countNonZero(thr[:, :w // 2])
    rs = cv2.countNonZero(thr[:, w // 2:])
    return ls / (ls + rs) if (ls + rs) else 0.5

def get_eye_region_from_landmarks(landmarks, indices, frame_width, frame_height):
    return np.array(
        [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in [landmarks[i] for i in indices]],
        np.int32
    )

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
stacking_model = joblib.load('PersonalizedTempModels/enhanced_stacking_model_multiclass.pkl')
scaler = joblib.load('PersonalizedTempModels/feature_scaler.pkl')
gender_encoder = joblib.load('PersonalizedTempModels/gender_encoder.pkl')
stream_encoder = joblib.load('PersonalizedTempModels/stream_encoder.pkl')
student_encoder = joblib.load('PersonalizedTempModels/student_id_encoder.pkl')
kmeans = joblib.load('PersonalizedTempModels/optimized_kmeans.pkl')
scaler_for_clustering = joblib.load('PersonalizedTempModels/scaler_for_clustering.pkl')
k = kmeans.n_clusters
logger.info("Artifacts loaded.")

# ---------------------------
# Initialize Persistent MediaPipe Face Mesh (Continuous Mode)
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
persistent_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
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
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
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
            gr = (get_gaze_ratio(le, gray) + get_gaze_ratio(re, gray)) / 2.0
            if 0.35 < gr < 0.65:
                self.gaze_center_frames += 1
        return self.blink_count, self.gaze_center_frames / self.fps

global_accumulator = BlinkGazeAccumulator(fps=30, ear_threshold=0.19, consec_frames=2, resize_factor=0.5)

# ---------------------------
# Feature Extraction Function
# ---------------------------
def extract_features_from_frame(frame, student_id, gender, stream):
    resize_factor = global_accumulator.resize_factor
    frame_resized = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
    frame_height, frame_width = frame_resized.shape[:2]
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    with face_mesh_lock:
        res = persistent_face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    landmarks = res.multi_face_landmarks[0].landmark

    # Calculate movement vector for each landmark (example: Euclidean norm of x,y coordinates)
    movement_vector = [np.linalg.norm(np.array([lm.x, lm.y])) for lm in landmarks]
    movement_vector = np.array(movement_vector)
    movement_stats = calculate_movement_statistics(movement_vector)
    _, gaze_duration = global_accumulator.update(landmarks, frame_width, frame_height, frame_resized)
    count = random.uniform(10, 80)




    num_landmarks = 478
    feature_dict = {f'landmark_{i+1}': movement_vector[i] if i < len(movement_vector) else 0.0
                    for i in range(num_landmarks)}
    feature_dict.update(movement_stats)
    feature_dict['blink_count'] = count
    feature_dict['gaze_duration'] = gaze_duration

    # Determine if the student is looking (isLooking flag)
    le = get_eye_region_from_landmarks(landmarks, left_eye_indices, frame_width, frame_height)
    re = get_eye_region_from_landmarks(landmarks, right_eye_indices, frame_width, frame_height)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gr = (get_gaze_ratio(le, gray) + get_gaze_ratio(re, gray)) / 2.0
    feature_dict['isLooking'] = 1 if (0.35 < gr < 0.65) else 0


    default_vector = np.array([[60, 1, count, gaze_duration]])
    cluster_dists = [np.linalg.norm(default_vector - center) for center in kmeans.cluster_centers_]
    for i, d in enumerate(cluster_dists):
        feature_dict[f'cluster_dist_{i}'] = d

    # Add personalized features using the provided student_id, gender, and stream
    try:
        feature_dict['gender_encoded'] = int(gender_encoder.transform([gender])[0])
    except Exception:
        feature_dict['gender_encoded'] = 0
    try:
        feature_dict['stream_encoded'] = int(stream_encoder.transform([stream])[0])
    except Exception:
        feature_dict['stream_encoded'] = 0
    # Transform student_id using the student_encoder (expecting a 2D array)
    try:
        student_feature = student_encoder.transform([[student_id]])
        student_feature_names = student_encoder.get_feature_names_out(['student_id'])
        for i, col in enumerate(student_feature_names):
            feature_dict[col] = student_feature[0, i]
    except Exception:
        # If transformation fails, fill with zeros based on number of columns
        num_cols = len(student_encoder.get_feature_names_out(['student_id']))
        for col in student_encoder.get_feature_names_out(['student_id']):
            feature_dict[col] = 0.0

    return pd.DataFrame([feature_dict])

# ---------------------------
# Inference Function for a Single Frame
# ---------------------------
def process_single_frame(frame, student_id, gender, stream):
    fdf = extract_features_from_frame(frame, student_id, gender, stream)
    if fdf is None:
        current_app.logger.info("No face detected.")
        return "No face detected", None
    # Drop the extra "isLooking" feature before prediction so that the model gets the expected features.
    fdf_model = fdf.drop("isLooking", axis=1)
    p = stacking_model.predict(fdf_model.values)[0]
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

    # Expecting each image item to be a dict with keys: image, studentID, gender, stream.
    images = data['images']
    combined_results = []  # To store results for all frames.
    per_student_results = {}  # Grouped by studentID.

    for item in images:
        if isinstance(item, dict):
            img_data = item.get("image")
            student_id = item.get("studentID", "default")
            gender = item.get("gender", "male")
            stream = item.get("stream", "commerce")
        else:
            img_data = item
            student_id = "default"
            gender = "male"
            stream = "commerce"

        if img_data is None:
            continue

        if img_data.startswith("data:image/jpeg;base64,"):
            img_data = img_data.split(",")[1]

        try:
            ib = base64.b64decode(img_data)
            np_arr = np.frombuffer(ib, np.uint8)
            frm = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            continue

        if frm is None:
            continue

        p, fdf = process_single_frame(frm, student_id, gender, stream)
        if p == "No face detected":
            continue

        try:
            if hasattr(stacking_model, "predict_proba"):
                fdf_model = fdf.drop("isLooking", axis=1)
                a = stacking_model.predict_proba(fdf_model.values)[0]
                mid_values = np.array([10, 30, 50, 70, 90])
                base_percentage = float(np.sum(mid_values * a))
                noise = random.gauss(0, 1)
                comp = max(0, min(100, round(base_percentage + noise, 2)))
                lvl = int(np.argmax(a))
            else:
                lvl = int(p)
                mapping = {0: 10, 1: 30, 2: 50, 3: 70, 4: 90}
                comp = mapping.get(lvl, 50)
        except Exception as ex:
            lvl = int(p)
            mapping = {0: 10, 1: 30, 2: 50, 3: 70, 4: 90}
            comp = mapping.get(lvl, 50)

        isLooking = bool(fdf['isLooking'].iloc[0])
        combined_results.append((lvl, comp, isLooking))
        if student_id is not None:
            if student_id not in per_student_results:
                per_student_results[student_id] = []
            per_student_results[student_id].append((lvl, comp, isLooking))

    if not combined_results:
        return jsonify({"error": "No valid frames processed"}), 200

    avg_percentage = round(sum(r[1] for r in combined_results) / len(combined_results), 2)
    avg_level = int(round(sum(r[0] for r in combined_results) / len(combined_results)))

    per_student_avg = {}
    for sid, results in per_student_results.items():
        avg_percent = round(sum(r[1] for r in results) / len(results), 2)
        avg_lvl = int(round(sum(r[0] for r in results) / len(results)))
        isLooking_flag = (sum(1 for r in results if r[2]) / len(results)) >= 0.5
        per_student_avg[sid] = {"prediction": avg_lvl, "percentage": avg_percent, "isLooking": isLooking_flag}

    return jsonify({
        "combined": {
            "prediction": avg_level,
            "percentage": avg_percentage
        },
        "perStudent": per_student_avg
    })

import json
import os
import cv2
import mediapipe as mp
from flask_cors import cross_origin
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
from flask import Blueprint, request
import tempfile
import pandas as pd
import threading
import glob
import time

# Import the continuous learning function from continuous_training.py
from continuous_training import continuous_training

# Create the blueprint
test_questions = Blueprint('test_questions', __name__)

# Blink and Gaze Detection Functions

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    h, w = threshold_eye.shape
    left_side = threshold_eye[:, 0:int(w / 2)]
    right_side = threshold_eye[:, int(w / 2):w]
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    if left_white + right_white == 0:
        return 0.5
    gaze_ratio = left_white / (left_white + right_white)
    return gaze_ratio

def get_eye_region_from_landmarks(landmarks, indices, frame_width, frame_height):
    coords = []
    for idx in indices:
        x = int(landmarks[idx].x * frame_width)
        y = int(landmarks[idx].y * frame_height)
        coords.append((x, y))
    return np.array(coords, np.int32)

# MediaPipe Face Mesh Initialization

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define eye landmark indices (for blink/gaze detection)
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Video Processing Function

def process_video_with_blink_and_gaze(video_path):
    cap = cv2.VideoCapture(video_path)
    landmark_list = deque(maxlen=1000)  # To store landmarks for averaging
    blink_count = 0
    frame_counter = 0
    gaze_center_frames = 0
    EAR_THRESHOLD = 0.21  # For blink detection
    CONSEC_FRAMES = 3

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # default fps if unavailable

    frame_count = 0  # count total frames for duration calculation
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            landmark_list.append([(lm.x, lm.y) for lm in landmarks])

            left_eye = get_eye_region_from_landmarks(landmarks, left_eye_indices, frame_width, frame_height)
            right_eye = get_eye_region_from_landmarks(landmarks, right_eye_indices, frame_width, frame_height)

            if left_eye.shape[0] == 6 and right_eye.shape[0] == 6:
                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSEC_FRAMES:
                        blink_count += 1
                    frame_counter = 0

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gaze_ratio_left = get_gaze_ratio(left_eye, gray)
                gaze_ratio_right = get_gaze_ratio(right_eye, gray)
                gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2.0
                if 0.35 < gaze_ratio < 0.65:
                    gaze_center_frames += 1

    cap.release()

    avg_landmarks = np.mean(landmark_list, axis=0).tolist() if len(landmark_list) > 0 else None
    gaze_duration = gaze_center_frames / fps
    video_duration = frame_count / fps  # total duration in seconds

    return avg_landmarks, blink_count, gaze_duration, video_duration

# Movement Statistics Functions

def calculate_landmark_movement(neutral_landmarks, question_landmarks):
    movements = []
    for n, q in zip(neutral_landmarks, question_landmarks):
        dx = q[0] - n[0]
        dy = q[1] - n[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        movements.append(distance)
    return movements

def calculate_movement_statistics(movements):
    return {
        'mean_movement': np.mean(movements),
        'std_movement': np.std(movements),
        'max_movement': np.max(movements),
        'min_movement': np.min(movements),
        'range_movement': np.max(movements) - np.min(movements),
        'median_movement': np.median(movements)
    }

# Upload Endpoint

@test_questions.route('/upload', methods=['POST'])
@cross_origin()
def upload_video():
    # Check for required video files
    if 'neutral_video' not in request.files or 'question_video' not in request.files:
        return "Both 'neutral_video' and 'question_video' files must be provided.", 400

    neutral_file = request.files['neutral_video']
    question_files = request.files.getlist('question_video')
    if neutral_file.filename == '' or len(question_files) == 0:
        return "Neutral video or question videos not selected.", 400

    # Retrieve and parse the metadata JSON (if provided)
    metadata_json = request.form.get('question_metadata')
    if metadata_json:
        try:
            metadata_list = json.loads(metadata_json)
        except Exception as e:
            return f"Error parsing question metadata: {str(e)}", 400
    else:
        metadata_list = []

    # Save and process the neutral video
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as neutral_tmp:
        neutral_file.save(neutral_tmp.name)
        neutral_tmp_path = neutral_tmp.name

    try:
        neutral_landmarks, neutral_blinks, neutral_gaze, neutral_duration = process_video_with_blink_and_gaze(neutral_tmp_path)
    except Exception as e:
        os.remove(neutral_tmp_path)
        return f"Error processing neutral video: {str(e)}", 400
    os.remove(neutral_tmp_path)

    # Process each question video
    question_results = []
    for i, question_file in enumerate(question_files):
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as question_tmp:
            question_file.save(question_tmp.name)
            question_tmp_path = question_tmp.name

        try:
            q_landmarks, blink_count, gaze_duration, _ = process_video_with_blink_and_gaze(question_tmp_path)
            os.remove(question_tmp_path)

            # Compute flattened landmarks: for each of the 478 landmarks, compute (x+y)/2
            if q_landmarks is not None:
                flattened_landmarks = [(lm[0] + lm[1]) / 2 for lm in q_landmarks]
            else:
                flattened_landmarks = [None] * 478

            # Compute movement statistics between neutral and question landmarks
            if neutral_landmarks is not None and q_landmarks is not None:
                movements = calculate_landmark_movement(neutral_landmarks, q_landmarks)
                stats = calculate_movement_statistics(movements)
            else:
                stats = {
                    'mean_movement': None,
                    'std_movement': None,
                    'max_movement': None,
                    'min_movement': None,
                    'range_movement': None,
                    'median_movement': None
                }

            result = {
                "blink_count": blink_count,
                "gaze_duration": gaze_duration,
            }
            # Add flattened landmarks as separate columns landmark_1 ... landmark_478
            for j, val in enumerate(flattened_landmarks, start=1):
                result[f"landmark_{j}"] = val

            # Add movement statistics
            result.update(stats)

            # Merge metadata for this question if available
            if i < len(metadata_list):
                result.update(metadata_list[i])
            question_results.append(result)
        except Exception as e:
            if os.path.exists(question_tmp_path):
                os.remove(question_tmp_path)
            return f"Error processing question video {i+1}: {str(e)}", 400

    # Combine results (neutral_duration is available if needed)
    overall_results = {"neutral_duration": neutral_duration, "questions": question_results}
    df = pd.DataFrame(overall_results["questions"])
    metadata_cols = ["student_id", "gender", "question_id", "stream", "is_correct", "timestamp"]
    landmark_cols = [f"landmark_{i}" for i in range(1, 479)]
    movement_cols = ["mean_movement", "std_movement", "max_movement", "min_movement", "range_movement", "median_movement"]
    extra_cols = ["blink_count", "gaze_duration"]
    desired_order = metadata_cols + landmark_cols + movement_cols + extra_cols

    df = df.reindex(columns=desired_order)

    # Save CSV to GeneralCsv folder with filename as student_id
    output_folder = "GeneralCsv"
    os.makedirs(output_folder, exist_ok=True)
    if "student_id" in df.columns and not df["student_id"].empty and pd.notnull(df["student_id"].iloc[0]):
        student_id = df["student_id"].iloc[0]
    else:
        student_id = "default"
    csv_filename = os.path.join(output_folder, f"{student_id}.csv")
    df.to_csv(csv_filename, index=False)
    print("Results saved to CSV:")
    print(df)

    # Trigger continuous training if threshold is met
    THRESHOLD = 10
    csv_files = glob.glob(os.path.join(output_folder, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in '{output_folder}'.")
    if len(csv_files) >= THRESHOLD:
        print("Threshold reached. Triggering continuous training process...")
        thread = threading.Thread(target=continuous_learning, args=(output_folder,))
        thread.start()
    else:
        print(f"Waiting for more CSV files. Threshold: {THRESHOLD}")

    return "Results saved to " + csv_filename, 200

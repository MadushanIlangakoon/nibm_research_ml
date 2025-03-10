import os
import glob
import pandas as pd
import numpy as np
import time
import joblib
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------- Utility Functions --------------

def combine_new_csv_files(new_data_dir, threshold=10):
    """Check for at least `threshold` CSV files in new_data_dir, and combine them into one DataFrame."""
    csv_files = glob.glob(os.path.join(new_data_dir, '*.csv'))
    if len(csv_files) < threshold:
        logger.info(f"Only {len(csv_files)} CSV files found. Need at least {threshold} to trigger continuous training.")
        return None, csv_files
    logger.info(f"Found {len(csv_files)} new CSV files. Combining them...")
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_new_data = pd.concat(df_list, ignore_index=True)
    return combined_new_data, csv_files


def move_files_to_archive(file_list, new_data_dir):
    """Move files in file_list to the archive folder within new_data_dir."""
    archive_dir = os.path.join(new_data_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    for file in file_list:
        base = os.path.basename(file)
        os.rename(file, os.path.join(archive_dir, base))
    logger.info("Processed CSV files moved to archive.")


# -------------- Continuous Training Script --------------

def continuous_training():
    # Folder containing new CSV files.
    new_data_dir = "GeneralCsv"

    # Load old training dataset.
    historical_data_path = "../final_student_comprehension_data.csv"
    logger.info("Loading historical dataset...")
    historical_data = pd.read_csv(historical_data_path)
    historical_data['is_correct'] = historical_data['is_correct'].astype(int)
    historical_data['timestamp'] = historical_data['timestamp'].fillna(historical_data['timestamp'].median())

    # Combine new CSV files.
    new_data, csv_files = combine_new_csv_files(new_data_dir, threshold=10)
    if new_data is None:
        logger.info("Not enough new data to retrain. Exiting continuous training process.")
        return

    # Optionally, save the combined new data for record (optional step)
    combined_new_data_path = os.path.join(new_data_dir, 'combined_new_data.csv')
    new_data.to_csv(combined_new_data_path, index=False)
    logger.info(f"Combined new data saved to {combined_new_data_path}")

    # Merge historical data with new data.
    merged_data = pd.concat([historical_data, new_data], ignore_index=True)
    logger.info(f"Total merged dataset shape: {merged_data.shape}")
    # Save merged dataset back to the historical dataset file.
    merged_data.to_csv(historical_data_path, index=False)
    logger.info(f"Historical dataset updated and saved to {historical_data_path}")

    # Move new CSV files to archive.
    move_files_to_archive(csv_files, new_data_dir)

    # ---------------------------
    # Now run the personalized training pipeline (without modifying model parameters)
    # ---------------------------

    # 1. Derive comprehension levels via clustering
    clust_feats = ['timestamp', 'is_correct', 'blink_count', 'gaze_duration']
    data_for_clust = merged_data[clust_feats]
    scaler_clust = StandardScaler()
    data_for_clust_scaled = scaler_clust.fit_transform(data_for_clust)

    n_levels = 5
    km = KMeans(n_clusters=n_levels, random_state=42)
    clust_labels = km.fit_predict(data_for_clust_scaled)
    y_orig = pd.Series(clust_labels)

    # 2. Compute cluster distance features
    centers = km.cluster_centers_
    dists = np.array([[np.linalg.norm(sample - center) for center in centers]
                      for sample in data_for_clust_scaled])
    for i in range(n_levels):
        merged_data[f'dist_feat_{i}'] = dists[:, i]

    # 3. Prepare features for training with personalized student_id
    # Keep student_id for personalization.
    cols_drop = ['question_id', 'is_correct', 'timestamp']
    X_df = merged_data.drop(columns=cols_drop)

    # Encode gender and stream using LabelEncoder
    gen_enc = LabelEncoder()
    str_enc = LabelEncoder()
    X_df['gender_enc'] = gen_enc.fit_transform(X_df['gender'])
    X_df['stream_enc'] = str_enc.fit_transform(X_df['stream'])
    # One-hot encode student_id
    student_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    student_encoded = student_encoder.fit_transform(X_df[['student_id']])
    student_encoded_df = pd.DataFrame(student_encoded,
                                      columns=student_encoder.get_feature_names_out(['student_id']),
                                      index=X_df.index)
    # Drop original categorical columns for student_id, gender, stream
    X_df = X_df.drop(columns=['student_id', 'gender', 'stream'])
    # Concatenate one-hot encoded student_id with the rest of the features
    X_processed = pd.concat([student_encoded_df, X_df], axis=1)

    # 4. Scale features
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_processed)

    # 5. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_orig, test_size=0.3, random_state=42, stratify=y_orig
    )

    # 6. Apply SMOTE to balance training data
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # 7. Define and tune base models (using current parameters, unchanged)
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=1, random_state=42,
                                      class_weight='balanced', n_jobs=-1)
    xgb_model = XGBClassifier(n_estimators=30, max_depth=1, learning_rate=0.01,
                              random_state=42, n_jobs=-1, eval_metric='mlogloss')
    etc_model = ExtraTreesClassifier(n_estimators=30, max_depth=1, random_state=42,
                                     class_weight='balanced', n_jobs=-1)
    lr_model = LogisticRegression(max_iter=1000, C=0.01, random_state=42,
                                  multi_class='multinomial', class_weight='balanced')
    svc_model = SVC(probability=True, kernel='rbf', C=0.5, gamma='scale', random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(25,), max_iter=300, alpha=0.1, random_state=42)
    gbc_model = GradientBoostingClassifier(random_state=42)

    # GridSearch for RandomForest (simpler grid)
    param_grid_rf = {
        'n_estimators': [20, 30],
        'max_depth': [1],
        'min_samples_split': [5],
        'min_samples_leaf': [2]
    }
    gs_rf = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1, verbose=1)
    start = time.time()
    gs_rf.fit(X_train_bal, y_train_bal)
    logger.info(f"RF GridSearch completed in {time.time() - start:.2f} seconds.")
    best_rf = gs_rf.best_estimator_

    # GridSearch for XGBoost (simpler grid)
    xgb_grid = XGBClassifier(random_state=42, n_jobs=1, eval_metric='mlogloss')
    param_grid_xgb = {
        'n_estimators': [20, 30],
        'learning_rate': [0.01],
        'max_depth': [1],
        'subsample': [0.6],
        'colsample_bytree': [0.7]
    }
    gs_xgb = GridSearchCV(xgb_grid, param_grid_xgb, cv=3, n_jobs=-1, verbose=1)
    start = time.time()
    gs_xgb.fit(X_train_bal, y_train_bal)
    logger.info(f"XGB GridSearch completed in {time.time() - start:.2f} seconds.")
    best_xgb = gs_xgb.best_estimator_

    # 8. Build a stacking ensemble with LightGBM as meta-learner
    base_estimators = [
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lr', lr_model)
    ]
    final_est = lgb.LGBMClassifier(n_estimators=20, max_depth=1, learning_rate=0.05, random_state=42)
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_est,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=True
    )

    # 9. Train and evaluate the stacking ensemble
    stacking_clf.fit(X_train_bal, y_train_bal)
    y_train_pred = stacking_clf.predict(X_train_bal)
    train_acc = accuracy_score(y_train_bal, y_train_pred)
    y_pred = stacking_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(stacking_clf, X_train_bal, y_train_bal, cv=3)

    logger.info(f"Training Accuracy: {train_acc}")
    logger.info(f"Test Accuracy: {test_acc}")
    logger.info(f"Cross-validation scores: {cv_scores}")
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("Classification Report:")
    logger.info(report)

    # 10. Save models and preprocessing artifacts in 'personalized' folder
    save_dir = "personalized"
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(stacking_clf, os.path.join(save_dir, 'enhanced_stacking_model_multiclass.pkl'))
    joblib.dump(scaler_full, os.path.join(save_dir, 'feature_scaler.pkl'))
    joblib.dump(gen_enc, os.path.join(save_dir, 'gender_encoder.pkl'))
    joblib.dump(str_enc, os.path.join(save_dir, 'stream_encoder.pkl'))
    joblib.dump(student_encoder, os.path.join(save_dir, 'student_id_encoder.pkl'))
    joblib.dump(km, os.path.join(save_dir, 'optimized_kmeans.pkl'))
    joblib.dump(scaler_clust, os.path.join(save_dir, 'scaler_for_clustering.pkl'))
    logger.info("Updated personalized models and artifacts saved in the 'personalized' folder.")

    # 11. Save detailed performance metrics to a txt file (no graphs)
    performance_file = os.path.join(save_dir, "model_performance.txt")
    with open(performance_file, "w") as f:
        f.write("Training Accuracy: {:.4f}\n".format(train_acc))
        f.write("Test Accuracy: {:.4f}\n".format(test_acc))
        f.write("Cross-validation scores: {}\n".format(cv_scores))
        f.write("\nClassification Report:\n")
        f.write(report)
    logger.info(f"Detailed performance metrics saved to {performance_file}")


if __name__ == "__main__":
    continuous_training()

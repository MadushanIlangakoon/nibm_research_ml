# server/blueprints/archive.py
from flask import Blueprint, request, jsonify
import os
import pandas as pd

archive_bp = Blueprint('archive', __name__)

# Set these paths as appropriate for your environment.
GENERAL_CSV_FOLDER = '/GeneralCsv'
ARCHIVE_FOLDER = '/GeneralCsv/archive'
FINAL_CSV_FILE = '../final_student_comprehension_data.csv'

@archive_bp.route('/delete_student_csvs', methods=['POST'])
def delete_student_csvs():
    data = request.get_json()
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'error': 'student_id is required'}), 400

    # Delete matching CSV file in the GeneralCsv folder.
    general_csv_path = os.path.join(GENERAL_CSV_FOLDER, f"{student_id}.csv")
    if os.path.exists(general_csv_path):
        try:
            os.remove(general_csv_path)
        except Exception as e:
            return jsonify({'error': f'Error deleting file {general_csv_path}: {str(e)}'}), 500

    # Delete matching CSV file in the archive folder.
    archive_csv_path = os.path.join(ARCHIVE_FOLDER, f"{student_id}.csv")
    if os.path.exists(archive_csv_path):
        try:
            os.remove(archive_csv_path)
        except Exception as e:
            return jsonify({'error': f'Error deleting file {archive_csv_path}: {str(e)}'}), 500

    # Update final_student_comprehension_data.csv: remove rows with matching student_id.
    if os.path.exists(FINAL_CSV_FILE):
        try:
            df = pd.read_csv(FINAL_CSV_FILE)
            # Assuming the column name is 'student_id'
            df_filtered = df[df['student_id'] != student_id]
            df_filtered.to_csv(FINAL_CSV_FILE, index=False)
        except Exception as e:
            return jsonify({'error': f'Error updating final CSV file: {str(e)}'}), 500

    return jsonify({'message': 'CSV files for student deleted successfully.'}), 200

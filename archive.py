from flask import Blueprint, request, jsonify
import os
import pandas as pd
from flask_cors import cross_origin

archive_bp = Blueprint('archive', __name__)

# Set these paths as appropriate for your environment.
GENERAL_CSV_FOLDER = '/GeneralCsv'
ARCHIVE_FOLDER = '/GeneralCsv/archive'
FINAL_CSV_FILE = '../final_student_comprehension_data.csv'

@archive_bp.route('/delete_student_csvs', methods=['POST', 'OPTIONS'])
@cross_origin(origin='http://localhost:3000', methods=['POST', 'OPTIONS'])
def delete_student_csvs():
    # Handle preflight (OPTIONS) request.
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'error': 'student_id is required'}), 400

    # Delete matching CSV file in the GeneralCsv folder.
    general_csv_path = os.path.join(GENERAL_CSV_FOLDER, f"{student_id}.csv")
    try:
        os.remove(general_csv_path)
    except FileNotFoundError:
        # File does not exist, which is acceptable.
        pass
    except Exception as e:
        return jsonify({'error': f'Error deleting file {general_csv_path}: {str(e)}'}), 500

    # Delete matching CSV file in the archive folder.
    archive_csv_path = os.path.join(ARCHIVE_FOLDER, f"{student_id}.csv")
    try:
        os.remove(archive_csv_path)
    except FileNotFoundError:
        pass
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

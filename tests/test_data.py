import os
import json
import tempfile
import shutil
import pytest
from src.data.loader import load_data
from src.data.preprocessor import preprocess_data

def test_load_data_file_json():
    # Create a temporary JSON file with a single medical report.
    data = {"patient_id": "123", "text": "Test medical report", "date": "2025-03-15"}
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_file:
        json.dump(data, tmp_file)
        tmp_file_path = tmp_file.name

    # Load data from the file and then clean up.
    loaded_data = load_data(tmp_file_path)
    os.unlink(tmp_file_path)  # Cleanup the temporary file

    assert isinstance(loaded_data, list)
    assert len(loaded_data) == 1
    assert loaded_data[0]["patient_id"] == "123"

def test_load_data_dir_json():
    path = 'data/medical_reports'
    loaded_data = load_data(path)
    assert isinstance(loaded_data, list)

def test_load_data_directory_json():
    # Create a temporary directory and write two JSON report files.
    temp_dir = tempfile.mkdtemp()
    data1 = {"patient_id": "123", "text": "Report one", "date": "2025-03-15"}
    data2 = {"patient_id": "456", "text": "Report two", "date": "2025-03-16"}

    file1 = os.path.join(temp_dir, "report1.json")
    file2 = os.path.join(temp_dir, "report2.json")
    with open(file1, "w") as f:
        json.dump(data1, f)
    with open(file2, "w") as f:
        json.dump(data2, f)

    loaded_data = load_data(temp_dir)
    shutil.rmtree(temp_dir)  # Cleanup the temporary directory

    assert isinstance(loaded_data, list)
    patient_ids = {report["patient_id"] for report in loaded_data}
    assert patient_ids == {"123", "456"}

def test_preprocess_data():
    # Create dummy reports with and without the 'text' field.
    reports = [
        {"patient_id": "001", "text": "   SAMPLE REPORT   ", "date": "2025-03-15"},
        {"patient_id": "002", "date": "2025-03-16"},  # Missing text field.
    ]
    processed = preprocess_data(reports)
    # Check that the text is normalized and missing text gets a default empty string.
    assert processed[0]["text"] == "sample report"
    assert processed[1]["text"] == ""

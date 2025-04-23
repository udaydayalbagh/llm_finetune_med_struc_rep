import os
import json
import tempfile
import shutil
import pytest
from src.data.loader import load_data
from src.data.preprocessor import preprocess_data

def test_load_data_file_json():
    data = {"patient_id": "123", "text": "Test medical report", "date": "2025-03-15"}
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_file:
        json.dump(data, tmp_file)
        tmp_file_path = tmp_file.name

    loaded_data = load_data(tmp_file_path)
    os.unlink(tmp_file_path)

    assert isinstance(loaded_data, list)
    assert len(loaded_data) == 1
    assert loaded_data[0]["patient_id"] == "123"

def test_load_data_dir_json():
    path = 'data/medical_reports'
    loaded_data = load_data(path)
    assert isinstance(loaded_data, list)

def test_load_data_directory_json():
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
    shutil.rmtree(temp_dir)  

    assert isinstance(loaded_data, list)
    patient_ids = {report["patient_id"] for report in loaded_data}
    assert patient_ids == {"123", "456"}

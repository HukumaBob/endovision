from ultralytics import YOLO
import os
import json

def load_classes(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def load_model(model_path):
    return YOLO(model_path)

def find_json_for_model(model_path):
    model_dir, model_name = os.path.split(model_path)
    json_path = os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}.json")
    return json_path if os.path.exists(json_path) else None

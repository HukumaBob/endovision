import os
import json
import cv2


def init_video_processing(input_path: str, output_path: str):
    """
    Initializes VideoCapture and VideoWriter objects for video processing.

    :param input_path: Path to the input video file.
    :param output_path: Path to save the processed video.
    :return: Tuple (VideoCapture, VideoWriter).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return None, None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for writing video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Unable to create output file {output_path}")
        cap.release()
        return None, None

    return cap, writer

import cv2

def process_frame(cap, model, writer, class_names):
    """
    Processes a single video frame with category-based bounding box colors.

    :param cap: VideoCapture object for reading the video.
    :param model: Loaded YOLO model.
    :param writer: VideoWriter object for saving processed frames.
    :param class_names: Dictionary mapping categories to class indices and names.
    :return: Tuple (processed frame, completion flag).
    """
    # Define category colors
    category_colors = {
        "anatomy": (0, 255, 0),   # Green
        "findings": (0, 0, 255), # Red
        "quality": (0, 255, 255), # Yellow
        "artifacts": (255, 0, 0), # Blue
    }

    # Flatten class names into a lookup table {class_id: (category, class_name)}
    class_lookup = {}
    for category, classes in class_names.items():
        for class_id, class_name in classes.items():
            class_lookup[int(class_id)] = (category, class_name)

    ret, frame = cap.read()
    if not ret:
        return None, True  # Return None and completion flag

    # Run the frame through the model
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            confidence = box.conf[0] * 100

            # Lookup class name and category
            category, class_name = class_lookup.get(cls, ("Unknown", "Unknown"))

            # Determine the box color based on category
            box_color = category_colors.get(category, (255, 255, 255))  # Default white for unknown category

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Add class name and confidence to the frame
            cv2.putText(frame, f"{class_name} {confidence:.2f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Write the processed frame to the output
    writer.write(frame)

    return frame, False




def finalize_processing(cap, writer):
    """
    Releases resources associated with video processing.

    :param cap: VideoCapture object.
    :param writer: VideoWriter object.
    """
    if cap:
        cap.release()
    if writer:
        writer.release()
    print("Processing complete and resources released.")


def load_classes(classes_path: str):
    """
    Loads class names from a JSON file.

    :param classes_path: Path to the JSON file.
    :return: List of class names or None if the file is not provided or invalid.
    """
    if not classes_path or not os.path.exists(classes_path):
        print("Class names JSON file not provided or does not exist.")
        return None

    with open(classes_path, "r") as f:
        return json.load(f)

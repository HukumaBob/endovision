import cv2

from box_style import DashedBox, Ellipse, RoundedBox
from logo import overlay_logo

def init_video_processing(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    return cap, writer

def process_frame(cap, model, writer, class_names, logo_path=None):
    """
    Processes a single video frame with category-based bounding box styles.

    :param cap: VideoCapture object for reading the video.
    :param model: Loaded YOLO model.
    :param writer: VideoWriter object for saving processed frames.
    :param class_names: Dictionary mapping categories to class indices and names.
    :return: Tuple (processed frame, completion flag).
    """
    # Define styles for categories
    category_styles = {
        "anatomy": RoundedBox(color=(0, 255, 0)),      # Green rounded box
        "findings": Ellipse(color=(0, 0, 255)),     # Red dashed box
        "quality": DashedBox(color=(0, 255, 255)),      # Yellow ellipse
        "artifacts": Ellipse(color=(255, 0, 0))    # Blue rounded box
    }

    # Flatten class names into a lookup table
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

            # Determine the style and draw
            style = category_styles.get(category, RoundedBox(color=(255, 255, 255)))
            style.draw(frame, x1, y1, x2, y2, f"{class_name} {int(confidence)}%")


    # Write the processed frame to the output
    writer.write(frame)

    return frame, False

def finalize_processing(cap, writer):
    cap.release()
    writer.release()
import cv2
import os
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QGroupBox, QVBoxLayout

from video_processor import load_classes

def create_group_box(title, rows):
    """
    Creates a group box with a given title and rows of widgets.

    :param title: Title of the group box.
    :param rows: List of layouts (rows) to add to the group box.
    :return: QGroupBox with the specified rows.
    """
    group_box = QGroupBox(title)
    layout = QVBoxLayout()
    for row in rows:
        layout.addLayout(row)
    group_box.setLayout(layout)
    return group_box

def create_file_row(placeholder, attr_name, btn_text, file_filter, parent):
    """
    Creates a horizontal layout with a text input and a button for file selection.

    :param placeholder: Placeholder text for the QLineEdit.
    :param attr_name: Attribute name for storing the QLineEdit.
    :param btn_text: Button text.
    :param file_filter: File type filter for the file dialog.
    :param parent: Parent QWidget that calls this function.
    :return: QHBoxLayout containing the QLineEdit and QPushButton.
    """
    layout = QHBoxLayout()

    # Create a text input field
    line_edit = QLineEdit()
    line_edit.setPlaceholderText(placeholder)
    setattr(parent, attr_name, line_edit)

    # Create a button for file selection
    button = QPushButton(btn_text)
    button.clicked.connect(lambda: select_file(line_edit, placeholder, file_filter, parent))
    layout.addWidget(line_edit)
    layout.addWidget(button)

    return layout

def select_file(line_edit, placeholder, attr_name, file_filter, parent):
    """
    Opens a file dialog to select a file or specify a save path. If a model file is selected,
    automatically loads the corresponding JSON file with class names.

    :param line_edit: QLineEdit to update with the selected file path.
    :param placeholder: Dialog title.
    :param attr_name: Attribute name for the associated input field.
    :param file_filter: File type filter for the file dialog.
    :param parent: Parent QWidget that calls this function.
    """
    if attr_name == "output_path":
        # Allow specifying a new file for saving
        path, _ = QFileDialog.getSaveFileName(parent, placeholder, "", file_filter)
    else:
        # Allow selecting existing files
        path, _ = QFileDialog.getOpenFileName(parent, placeholder, "", file_filter)

    if path:
        line_edit.setText(path)

        # If the selected file is a YOLO model, attempt to load the corresponding JSON file
        if attr_name == "model_path":
            model_dir, model_name = os.path.split(path)
            json_path = os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}.json")
            if os.path.exists(json_path):
                class_names = load_classes(json_path)  # Assume load_classes is a utility function
                parent.class_names = class_names
                parent.status_label.setText("Class names loaded successfully!")
            else:
                parent.class_names = None
                parent.status_label.setText("Warning: Corresponding JSON file not found!")



def calculate_sharpness(image):
    """
    Calculates the sharpness of an image using the variance of the Laplacian.

    :param image: Input image.
    :return: Sharpness value (variance of Laplacian).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def find_sharpest_frame(frames):
    """
    Finds the sharpest frame in a list of frames.

    :param frames: List of frames to analyze.
    :return: The sharpest frame.
    """
    max_sharpness = 0
    sharpest_frame = None
    for frame in frames:
        sharpness = calculate_sharpness(frame)
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            sharpest_frame = frame
    return sharpest_frame

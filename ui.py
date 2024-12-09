import cv2
import os
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QWidget, QMainWindow, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from video_processor import init_video_processing, process_frame, finalize_processing, load_classes
from collections import deque
from ultralytics import YOLO


class VideoProcessorUI(QMainWindow):
    """
    Main application class for video processing with YOLO model.
    Provides a user interface for selecting files, starting video processing, and previewing the output.
    """
    FRAME_BUFFER_SIZE = 50

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing with YOLO")
        self.setGeometry(200, 200, 800, 600)

        # Initialize state variables
        self.timer = QTimer()
        self.cap = None
        self.model = None
        self.writer = None
        self.output_file_path = None
        self.class_names = None
        self.frame_buffer = deque(maxlen=self.FRAME_BUFFER_SIZE)

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initializes the user interface."""
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        # Input and output sections
        input_group = self.create_group_box("Input Files", [
            self.create_file_row("Select a video file...", "input_path", "Choose Video", "*.mp4 *.mpg *.avi *.webm"),
            self.create_file_row("Select a YOLO model file...", "model_path", "Choose Model", "*.pt"),
        ])
        output_group = self.create_group_box("Output", [
            self.create_file_row("Specify the output file path...", "output_path", "Save As...", "*.mp4"),
        ])
        logo_group = self.create_group_box("Logo", [
            self.create_file_row("Select a logo file...", "logo_path", "Choose Logo", "*.png"),
        ])

        # Buttons
        self.process_btn = self.create_button("Start Processing", self.start_processing, bold=True)
        self.freeze_btn = self.create_button("Freeze", self.freeze_frame)

        # Video preview labels
        self.video_label = self.create_label("Video Preview", 640, 480)
        self.sharpest_label = self.create_label("Sharpest Frame", 320, 240)

        # Status label
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-size: 12px;")

        # Layout arrangement
        main_layout.addWidget(logo_group)
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(self.process_btn)
        main_layout.addWidget(self.freeze_btn)
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.sharpest_label)
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(main_widget)

    def create_button(self, text, callback, bold=False):
        """Helper to create a styled button."""
        button = QPushButton(text)
        style = "font-size: 14px;"
        if bold:
            style += "font-weight: bold;"
        button.setStyleSheet(style)
        button.clicked.connect(callback)
        return button

    def create_label(self, text, width, height):
        """Helper to create a styled label."""
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background-color: black;")
        label.setFixedSize(width, height)
        return label

    def create_group_box(self, title, rows):
        """Creates a group box with a given title and rows of widgets."""
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        for row in rows:
            layout.addLayout(row)
        group_box.setLayout(layout)
        return group_box

    def create_file_row(self, placeholder, attr_name, btn_text, file_filter="*"):
        """Creates a row with a QLineEdit and a file selection button."""
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        setattr(self, attr_name, line_edit)

        button = QPushButton(btn_text)
        button.clicked.connect(lambda: self.select_file(line_edit, attr_name, file_filter))
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return layout

    def select_file(self, line_edit, attr_name, file_filter):
        """Opens a file dialog and sets the selected path to the line edit."""
        dialog_method = QFileDialog.getSaveFileName if attr_name == "output_path" else QFileDialog.getOpenFileName
        path, _ = dialog_method(self, "Select File", "", file_filter)
        if path:
            line_edit.setText(path)
            if attr_name == "model_path":
                self.load_class_names(path)

    def load_class_names(self, model_path):
        """Attempts to load class names from a JSON file associated with the model."""
        json_path = os.path.splitext(model_path)[0] + ".json"
        if os.path.exists(json_path):
            self.class_names = load_classes(json_path)

    def start_processing(self):
        """Initializes and starts the video processing."""
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_path = self.output_path.text()
        logo_path = self.logo_path.text()

        if not (input_path and model_path and output_path):
            self.update_status("Error: Please fill in all required fields!", "red")
            return
        if not self.class_names:
            self.update_status("Error: Corresponding JSON file not found!", "red")
            return
        if logo_path and not os.path.exists(logo_path):
            self.update_status("Error: Logo file not found!", "red")
            return

        self.model = YOLO(model_path)
        self.cap, self.writer = init_video_processing(input_path, output_path)
        if not self.cap or not self.writer:
            self.update_status("Error: Failed to open video or create output file!", "red")
            return

        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)))
        self.update_status("Video processing started...", "green")

    def process_frame(self):
        """Processes a single frame and updates the UI."""
        frame, finished = process_frame(self.cap, self.model, self.writer, self.class_names, logo_path=self.logo_path.text())
        if finished:
            self.timer.stop()
            finalize_processing(self.cap, self.writer)
            self.update_status(f"Processing complete! File saved at: {self.output_path.text()}", "green")
            return

        self.frame_buffer.append(frame)
        self.update_preview(self.video_label, frame)

    def freeze_frame(self):
        """Displays the sharpest frame from the buffer."""
        if not self.frame_buffer:
            self.update_status("Buffer is empty! No frames to analyze.", "red")
            return

        sharpest_frame = self.find_sharpest_frame(self.frame_buffer)
        if sharpest_frame is not None:
            self.update_preview(self.sharpest_label, sharpest_frame)
            self.update_status("Sharpest frame displayed!", "green")
        else:
            self.update_status("Unable to find the sharpest frame.", "red")

    def find_sharpest_frame(self, frames):
        """Finds the sharpest frame using Laplacian variance."""
        return max(frames, key=self.calculate_sharpness, default=None)

    @staticmethod
    def calculate_sharpness(image):
        """Calculates sharpness using the variance of the Laplacian."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def update_preview(self, label, frame):
        """Updates a QLabel with a frame preview."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_image))

    def update_status(self, message, color):
        """Updates the status label with a message."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")

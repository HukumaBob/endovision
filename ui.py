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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing with YOLO")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()

        # Initialize processing components
        self.timer = QTimer()
        self.cap = None
        self.model = None
        self.output_file_path = None
        self.class_names = None
        self.frame_buffer = deque(maxlen=50)  # Буфер для последних 50 кадров

    def init_ui(self):
        """Initializes the user interface."""
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        # Input files section
        input_group = self.create_group_box("Input Files", [
            self.create_file_row("Select a video file...", "input_path", "Choose Video", "*.mp4 *.mpg *.avi *.webm"),
            self.create_file_row("Select a YOLO model file...", "model_path", "Choose Model", "*.pt"),
        ])

        # Output files section
        output_group = self.create_group_box("Output", [
            self.create_file_row("Specify the output file path...", "output_path", "Save As...", "*.mp4"),
        ])
        # Добавление секции для выбора логотипа
        logo_group = self.create_group_box("Logo", [
            self.create_file_row("Select a logo file...", "logo_path", "Choose Logo", "*.png")
        ])            

        # Start processing button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.process_btn.clicked.connect(self.start_processing)

        # Freeze button
        self.freeze_btn = QPushButton("Freeze")
        self.freeze_btn.setStyleSheet("font-size: 14px;")
        self.freeze_btn.clicked.connect(self.freeze_frame)

        # Video preview label
        self.video_label = QLabel("Video Preview", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)

        # Sharpest frame preview label
        self.sharpest_label = QLabel("Sharpest Frame", self)
        self.sharpest_label.setAlignment(Qt.AlignCenter)
        self.sharpest_label.setStyleSheet("background-color: black;")
        self.sharpest_label.setFixedSize(320, 240)

        # Status label
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-size: 12px;")

        # Add components to the main layout
        main_layout.addWidget(logo_group)   
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(self.process_btn)
        main_layout.addWidget(self.freeze_btn)
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.sharpest_label)
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(main_widget)

    def freeze_frame(self):
        """
        Computes the sharpest frame from the buffer and displays it.
        """
        if not self.frame_buffer:
            self.status_label.setText("Buffer is empty! No frames to analyze.")
            return

        sharpest_frame = self.find_sharpest_frame(self.frame_buffer)
        if sharpest_frame is not None:
            # Convert and display the sharpest frame
            rgb_frame = cv2.cvtColor(sharpest_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.sharpest_label.setPixmap(pixmap)
            self.status_label.setText("Sharpest frame displayed!")
        else:
            self.status_label.setText("Unable to find the sharpest frame.")

    @staticmethod
    def calculate_sharpness(image):
        """
        Calculates the sharpness of an image using the variance of the Laplacian.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def find_sharpest_frame(self, frames):
        """
        Finds the sharpest frame in a list of frames.
        """
        max_sharpness = 0
        sharpest_frame = None
        for frame in frames:
            sharpness = self.calculate_sharpness(frame)
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                sharpest_frame = frame
        return sharpest_frame
    def create_group_box(self, title, rows):
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

    def create_file_row(self, placeholder, attr_name, btn_text, file_filter="*"):
        """
        Creates a horizontal layout with a text input and a button for file selection.

        :param placeholder: Placeholder text for the QLineEdit.
        :param attr_name: Attribute name for storing the QLineEdit.
        :param btn_text: Button text.
        :param file_filter: File type filter for the file dialog.
        :return: QHBoxLayout containing the QLineEdit and QPushButton.
        """
        layout = QHBoxLayout()

        # Create a text input field
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        setattr(self, attr_name, line_edit)

        # Create a button for file selection
        button = QPushButton(btn_text)
        button.clicked.connect(lambda: self.select_file(line_edit, placeholder, attr_name, file_filter))
        layout.addWidget(line_edit)
        layout.addWidget(button)

        return layout

    def select_file(self, line_edit, placeholder, attr_name, file_filter):
        """
        Opens a file dialog to select a file or specify a save path.

        :param line_edit: QLineEdit to update with the selected file path.
        :param placeholder: Dialog title.
        :param attr_name: Attribute name for the associated input field.
        :param file_filter: File type filter for the file dialog.
        """
        if attr_name == "output_path":
            # Allow specifying a new file for saving
            path, _ = QFileDialog.getSaveFileName(self, placeholder, "", file_filter)
        else:
            # Allow selecting existing files
            path, _ = QFileDialog.getOpenFileName(self, placeholder, "", file_filter)
        
        if path:
            line_edit.setText(path)
            if attr_name == "model_path":
                # Automatically find the corresponding JSON file if it exists
                model_dir, model_name = os.path.split(path)
                json_path = os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}.json")
                if os.path.exists(json_path):
                    self.class_names = load_classes(json_path)

    def start_processing(self):
        """
        Initializes video processing using the selected files and starts processing frames.
        """
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_path = self.output_path.text()
        logo_path = self.logo_path.text() if self.logo_path else None

        if not input_path or not model_path or not output_path:
            self.status_label.setText("Error: Please fill in all required fields!")
            return

        if not self.class_names:
            self.status_label.setText("Error: Corresponding JSON file not found!")
            return
        
        if not logo_path or not os.path.exists(logo_path):
            self.status_label.setText("Error: Logo file not found or not selected!")
            return        

        self.model = YOLO(model_path)

        # Initialize video processing
        self.cap, self.writer = init_video_processing(input_path, output_path)
        if not self.cap or not self.writer:
            self.status_label.setText("Error: Failed to open video or create file!")
            return

        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)))
        self.status_label.setText("Video processing started...")

    def process_frame(self):
        """
        Processes a single video frame, updates the UI, and handles video completion.
        """
        logo_path = self.logo_path.text() if self.logo_path else None
        frame, finished = process_frame(self.cap, self.model, self.writer, self.class_names, logo_path=logo_path)
        if finished:
            self.timer.stop()
            finalize_processing(self.cap, self.writer)
            self.status_label.setText(f"Processing complete! File saved at: {self.output_path.text()}")
            return

        # Add frame to buffer
        self.frame_buffer.append(frame)

        # Run frame processing (if needed)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

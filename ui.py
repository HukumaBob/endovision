import os
from collections import deque
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QWidget, QMainWindow, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2

from video_processing import init_video_processing, process_frame, finalize_processing
from frame_analysis import find_sharpest_frame
from model_handler import load_model, find_json_for_model, load_classes


class VideoProcessorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing with YOLO")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()

        self.timer = QTimer()
        self.cap = None
        self.model = None
        self.writer = None
        self.frame_buffer = deque(maxlen=50)
        self.class_names = None

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

        # Logo files section
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

    def freeze_frame(self):
        if not self.frame_buffer:
            self.status_label.setText("Buffer is empty!")
            return

        sharpest_frame = find_sharpest_frame(self.frame_buffer)
        if sharpest_frame is not None:
            self.display_frame(self.sharpest_label, sharpest_frame)
            self.status_label.setText("Sharpest frame displayed!")
        else:
            self.status_label.setText("No sharp frame found.")

    def display_frame(self, label, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)

    def start_processing(self):
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_path = self.output_path.text()
        logo_path = self.logo_path.text()

        if not input_path or not model_path or not output_path:
            self.status_label.setText("Fill in all required fields!")
            return

        json_path = find_json_for_model(model_path)
        if json_path:
            self.class_names = load_classes(json_path)

        self.model = load_model(model_path)
        self.cap, self.writer = init_video_processing(input_path, output_path)

        self.timer.timeout.connect(lambda: self.process_video_frame(logo_path))
        self.timer.start(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)))

    def process_video_frame(self, logo_path):
        frame, finished = process_frame(self.cap, self.model, self.writer, self.class_names, logo_path)
        if finished:
            self.timer.stop()
            finalize_processing(self.cap, self.writer)
            self.status_label.setText(f"Processing complete! File saved at: {self.output_path.text()}")
            return

        self.frame_buffer.append(frame)
        self.display_frame(self.video_label, frame)

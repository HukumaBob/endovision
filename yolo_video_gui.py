import os
import sys
import cv2
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QLineEdit, QWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO

class VideoProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Видеообработка с YOLO")
        self.setGeometry(200, 200, 800, 600)  # Увеличиваем размеры окна
        self.init_ui()
        self.timer = QTimer()
        self.cap = None  # Объект OpenCV для видео
        self.model = None  # YOLO модель
        self.output_file_path = None  # Переименованное поле для пути файла

    def init_ui(self):
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        self.input_path = QLineEdit(self)
        self.input_path.setPlaceholderText("Выберите видеофайл...")
        input_btn = QPushButton("Выбрать видео", self)
        input_btn.clicked.connect(self.select_input_file)

        self.model_path = QLineEdit(self)
        self.model_path.setPlaceholderText("Выберите модель YOLO...")
        model_btn = QPushButton("Выбрать модель", self)
        model_btn.clicked.connect(self.select_model_file)

        self.output_path = QLineEdit(self)  # Поле QLineEdit для ввода пути файла
        self.output_path.setPlaceholderText("Укажите путь для сохранения результата...")
        output_btn = QPushButton("Сохранить в...", self)
        output_btn.clicked.connect(self.select_output_file)

        self.classes_path = QLineEdit(self)
        self.classes_path.setPlaceholderText("Выберите JSON с именами классов (необязательно)...")
        classes_btn = QPushButton("Выбрать JSON", self)
        classes_btn.clicked.connect(self.select_classes_file)

        process_btn = QPushButton("Запустить обработку", self)
        process_btn.clicked.connect(self.start_processing)

        self.video_label = QLabel("Предпросмотр видео", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)

        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(QLabel("Входной файл:"))
        main_layout.addWidget(self.input_path)
        main_layout.addWidget(input_btn)

        main_layout.addWidget(QLabel("Модель YOLO:"))
        main_layout.addWidget(self.model_path)
        main_layout.addWidget(model_btn)

        main_layout.addWidget(QLabel("Результат сохранить как:"))
        main_layout.addWidget(self.output_path)
        main_layout.addWidget(output_btn)

        main_layout.addWidget(QLabel("JSON с именами классов (необязательно):"))
        main_layout.addWidget(self.classes_path)
        main_layout.addWidget(classes_btn)

        main_layout.addWidget(process_btn)
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(main_widget)

    def select_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", "", "Видео файлы (*.mp4 *.mpg *.avi *.webm)")
        if path:
            self.input_path.setText(path)

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель YOLO", "", "Модель YOLO (*.pt)")
        if path:
            self.model_path.setText(path)

    def select_output_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", "", "Видео файлы (*.mp4 *.mpg *.avi *.webm)")
        if path:
            self.output_file_path = path  # Сохранение пути
            self.output_path.setText(path)  # Отображение в поле ввода


    def select_classes_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите JSON файл", "", "JSON файлы (*.json)")
        if path:
            self.classes_path.setText(path)

    def start_processing(self):
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_path = self.output_path.text()  # Получаем путь из текстового поля
        classes_path = self.classes_path.text()

        if not input_path or not model_path or not output_path:
            self.status_label.setText("Ошибка: Заполните все обязательные поля!")
            return

        self.output_file_path = output_path  # Сохраняем путь в поле, чтобы использовать его позже

        self.class_names = None
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                self.class_names = json.load(f)

        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            self.status_label.setText("Ошибка: не удалось открыть видеофайл!")
            return

        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)))

        self.status_label.setText("Обработка видео началась...")


    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.status_label.setText("Обработка завершена!")
            return

        results = self.model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cls = int(box.cls[0])
                confidence = box.conf[0] * 100
                class_name = self.class_names[cls] if self.class_names and cls < len(self.class_names) else "Unknown"
                cv2.putText(frame, f"{class_name} {confidence:.2f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.video_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessorApp()
    window.show()
    sys.exit(app.exec_())

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QWidget, QMainWindow, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from video_processor import init_video_processing, process_frame, finalize_processing, load_classes
from ultralytics import YOLO
import cv2


class VideoProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Видеообработка с YOLO")
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()

        self.timer = QTimer()
        self.cap = None
        self.model = None
        self.output_file_path = None
        self.class_names = None

    def init_ui(self):
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)

        # Создаем секцию для выбора файлов
        input_group = self.create_group_box("Входные данные", [
            self.create_file_row("Выберите видеофайл...", "input_path", "Выбрать видео"),
            self.create_file_row("Выберите модель YOLO...", "model_path", "Выбрать модель"),
            self.create_file_row("JSON с именами классов (необязательно)...", "classes_path", "Выбрать JSON"),
        ])

        # Создаем секцию для вывода
        output_group = self.create_group_box("Результат", [
            self.create_file_row("Укажите путь для сохранения результата...", "output_path", "Сохранить в..."),
        ])

        # Кнопка запуска
        self.process_btn = QPushButton("Запустить обработку")
        self.process_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.process_btn.clicked.connect(self.start_processing)

        # Предпросмотр видео
        self.video_label = QLabel("Предпросмотр видео", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)

        # Статус
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-size: 12px;")

        # Добавляем элементы на главный компоновщик
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(self.process_btn)
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(main_widget)

    def create_group_box(self, title, rows):
        """Создает QGroupBox с набором строк (компонентов)."""
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        for row in rows:
            layout.addLayout(row)
        group_box.setLayout(layout)
        return group_box

    def create_file_row(self, placeholder, attr_name, btn_text):
        """Создает горизонтальную строку с текстовым полем и кнопкой."""
        layout = QHBoxLayout()

        # Создаем текстовое поле
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        setattr(self, attr_name, line_edit)

        # Создаем кнопку
        button = QPushButton(btn_text)
        button.clicked.connect(lambda: self.select_file(line_edit, placeholder, attr_name))
        layout.addWidget(line_edit)
        layout.addWidget(button)

        return layout

    def select_file(self, line_edit, placeholder, attr_name):
        """Открывает диалог выбора файла."""
        if attr_name == "output_path":
            path, _ = QFileDialog.getSaveFileName(self, placeholder, "", "*.mp4 *.avi *.webm")
        else:
            path, _ = QFileDialog.getOpenFileName(self, placeholder, "", "*.mp4 *.avi *.webm *.pt *.json")
        if path:
            line_edit.setText(path)

    def start_processing(self):
        """Инициализация обработки видео."""
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_path = self.output_path.text()
        classes_path = self.classes_path.text()

        if not input_path or not model_path or not output_path:
            self.status_label.setText("Ошибка: Заполните все обязательные поля!")
            return

        self.class_names = load_classes(classes_path)
        self.model = YOLO(model_path)

        # Инициализация обработки видео
        self.cap, self.writer = init_video_processing(input_path, output_path)
        if not self.cap or not self.writer:
            self.status_label.setText("Ошибка: не удалось открыть видео или создать файл!")
            return

        self.timer.timeout.connect(self.process_frame)
        self.timer.start(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)))
        self.status_label.setText("Обработка видео началась...")

    def process_frame(self):
        """Обработка кадра и обновление интерфейса."""
        frame, finished = process_frame(self.cap, self.model, self.writer, self.class_names)
        if finished:
            self.timer.stop()
            finalize_processing(self.cap, self.writer)
            self.status_label.setText(f"Обработка завершена! Файл сохранён: {self.output_path.text()}")
            return

        # Отображение обработанного кадра в QLabel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

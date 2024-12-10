from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QWidget, QMainWindow, QFileDialog, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import os
from collections import deque
import cv2

from utils import ConfigHandler, generate_output_filename
from video_processing import init_video_processing, process_frame, finalize_processing
from frame_analysis import find_sharpest_frame
from model_handler import load_model, find_json_for_model, load_classes



class VideoProcessorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing with YOLO")
        self.setGeometry(200, 200, 1000, 600)
        
        # Инициализация текущей темы
        self.current_theme = "light"  # По умолчанию светлая тема
        settings = ConfigHandler.load_settings()
        self.current_theme = settings.get("theme", "light")  # Загрузка сохраненной темы
        
        self.init_ui()
        self.apply_theme()  # Применить тему при старте приложения

        self.timer = QTimer()
        self.cap = None
        self.model = None
        self.writer = None
        self.frame_buffer = deque(maxlen=5)
        self.class_names = None
        self.frozen_frames = []  # Список замороженных кадров
        self.default_logo_path = "assets/default_logo.png"  # Укажите ваш путь

    def init_ui(self):
        """Initializes the user interface."""
        # Создание основного виджета и главного макета
        main_widget = QWidget(self)
        main_layout = QHBoxLayout(main_widget)

        # Левая секция (управление и видео)
        left_layout = QVBoxLayout()

        # Загрузка настроек
        settings = ConfigHandler.load_settings()

        # Input Files group
        input_group = self.create_group_box("Input Files", [
            self.create_file_row(
                placeholder="Select a video file...",
                attr_name="input_path",
                btn_text="Choose Video",
                file_filter="*.mp4 *.mpg *.avi *.webm"
            ),
            self.create_file_row(
                placeholder="Select a YOLO model file...",
                attr_name="model_path",
                btn_text="Choose Model",
                file_filter="*.pt"
            )
        ])

        # Output group
        output_group = self.create_group_box("Output", [
            self.create_folder_row(
                placeholder="Specify the output folder...",
                attr_name="output_path",
                btn_text="Choose Folder"
            )
        ])

        # Logo group
        logo_group = self.create_group_box("Logo", [
            self.create_file_row(
                placeholder="Select a logo file...",
                attr_name="logo_path",
                btn_text="Choose Logo",
                file_filter="*.png"
            )
        ])

        # Кнопки управления
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.process_btn.clicked.connect(self.start_processing)

        self.freeze_btn = QPushButton("Freeze")
        self.freeze_btn.setStyleSheet("font-size: 14px;")
        self.freeze_btn.clicked.connect(self.freeze_frame)

        # Кнопка переключения темы
        self.theme_btn = QPushButton("Switch to Dark Mode")
        self.theme_btn.setStyleSheet("font-size: 14px;")
        self.theme_btn.clicked.connect(self.toggle_theme)

        # Превью видео
        self.video_label = QLabel("Video Preview", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setFixedSize(640, 480)
        self.video_label.mouseDoubleClickEvent = self.expand_video_to_fullscreen

        # Статусная строка
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-size: 12px;")

        # Заполнение путей из настроек
        self.input_path.setText(settings.get("last_video_path", ""))
        self.model_path.setText(settings.get("last_model_path", ""))
        self.logo_path.setText(settings.get("last_logo_path", ""))
        self.output_path.setText(settings.get("output_folder", ""))

        # Компоновка левой секции
        left_layout.addWidget(input_group)
        left_layout.addWidget(output_group)
        left_layout.addWidget(logo_group)
        left_layout.addWidget(self.process_btn)
        left_layout.addWidget(self.freeze_btn)
        left_layout.addWidget(self.theme_btn)
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.status_label)

        # Правая секция (замороженные кадры)
        right_layout = QVBoxLayout()
        frozen_group = QGroupBox("Frozen Frames")
        frozen_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)
        self.scroll_area.setWidget(self.scroll_area_content)

        frozen_layout.addWidget(self.scroll_area)
        frozen_group.setLayout(frozen_layout)
        right_layout.addWidget(frozen_group)

        # Добавление секций в основной макет
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setCentralWidget(main_widget)

    def toggle_theme(self):
        """Switches between Light Mode and Dark Mode."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme()
        self.theme_btn.setText("Switch to Light Mode" if self.current_theme == "dark" else "Switch to Dark Mode")

        # Сохранить текущую тему в настройках
        settings = ConfigHandler.load_settings()
        settings["theme"] = self.current_theme
        ConfigHandler.save_settings(settings)

    def apply_theme(self):
        """Applies the current theme to the application."""
        if self.current_theme == "dark":
            self.setStyleSheet("""
                QWidget {
                    background-color: #2E2E2E;
                    color: #FFFFFF;
                }
                QLineEdit, QPushButton {
                    background-color: #3C3C3C;
                    border: 1px solid #5A5A5A;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #FFFFFF;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #FFFFFF;
                    color: #000000;
                }
                QLineEdit, QPushButton {
                    background-color: #F0F0F0;
                    border: 1px solid #D0D0D0;
                    color: #000000;
                }
                QLabel {
                    color: #000000;
                }
            """)


    def expand_video_to_fullscreen(self, event):
        """Expands the video preview to fullscreen."""
        fullscreen_window = QLabel()
        fullscreen_window.setPixmap(self.video_label.pixmap())
        fullscreen_window.setAlignment(Qt.AlignCenter)
        fullscreen_window.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        fullscreen_window.setStyleSheet("background-color: black;")
        fullscreen_window.showFullScreen()
        fullscreen_window.mouseDoubleClickEvent = lambda _: fullscreen_window.close()

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

    def choose_output_folder(self):
        """Позволяет выбрать папку для сохранения и сохраняет ее в настройки."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path.setText(folder)
            settings = ConfigHandler.load_settings()
            settings["output_folder"] = folder
            ConfigHandler.save_settings(settings)

    def create_folder_row(self, placeholder, attr_name, btn_text):
        layout = QHBoxLayout()

        # Поле для ввода
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        setattr(self, attr_name, line_edit)

        # Кнопка выбора папки
        button = QPushButton(btn_text)
        button.clicked.connect(self.choose_output_folder)
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
        settings = ConfigHandler.load_settings()
        start_path = ""
        
        # Определяем стартовую папку для каждого типа файла
        if attr_name == "input_path":
            start_path = settings.get("last_video_path", settings.get("input_folder", ""))
        elif attr_name == "model_path":
            start_path = settings.get("last_model_path", settings.get("models_folder", ""))
        elif attr_name == "logo_path":
            start_path = settings.get("last_logo_path", "")

        # Вызываем диалог выбора файла
        path, _ = QFileDialog.getOpenFileName(self, placeholder, start_path, file_filter)
        
        if path:
            line_edit.setText(path)
            
            # Сохраняем выбранный путь в настройки
            if attr_name == "input_path":
                settings["last_video_path"] = path
            elif attr_name == "model_path":
                settings["last_model_path"] = path
            elif attr_name == "logo_path":
                settings["last_logo_path"] = path
            
            ConfigHandler.save_settings(settings)
            
            # Автоматически искать JSON для модели
            if attr_name == "model_path":
                model_dir, model_name = os.path.split(path)
                json_path = os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}.json")
                if os.path.exists(json_path):
                    self.class_names = load_classes(json_path)
  

    def freeze_frame(self):
        """Freezes the current frame and adds it to the list of frozen frames."""
        if not self.frame_buffer:
            self.status_label.setText("Buffer is empty!")
            return

        sharpest_frame = find_sharpest_frame(self.frame_buffer)
        if sharpest_frame is not None:
            self.add_frozen_frame(sharpest_frame)
            self.status_label.setText("Sharpest frame added!")
        else:
            self.status_label.setText("No sharp frame found.")

    def add_frozen_frame(self, frame):
        """Добавляет замороженный кадр в прокручиваемый список."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Получаем 90% фактической ширины видимой области скролла
        scroll_width = self.scroll_area.viewport().width()
        frame_width = int(0.95 * scroll_width)

        # Создаем QLabel для кадра
        frame_label = QLabel()
        frame_label.setPixmap(pixmap.scaled(frame_width, frame_width * h // w, Qt.KeepAspectRatio))
        frame_label.setAlignment(Qt.AlignCenter)
        frame_label.setFrameShape(QFrame.Box)

        # Добавляем кадр в компоновку и список
        self.scroll_area_layout.addWidget(frame_label)
        self.frozen_frames.append(frame)

        # Прокручиваем скролл вниз после обновления интерфейса
        QTimer.singleShot(5, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

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
        logo_path = self.logo_path.text()

        if not logo_path and os.path.exists(self.default_logo_path):
            logo_path = self.default_logo_path            

        if not input_path or not model_path:
            self.status_label.setText("Fill in all required fields!")
            return

        settings = ConfigHandler.load_settings()
        output_folder = settings.get("output_folder", "")
        if not output_folder or not os.path.isdir(output_folder):
            self.status_label.setText("Output folder not configured or invalid!")
            return

        output_path = generate_output_filename(input_path, output_folder)
        self.output_path.setText(output_path)

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

import json
import os
import uuid

class ConfigHandler:
    CONFIG_FILE = "settings.json"

    @staticmethod
    def load_settings():
        """Загружает настройки из JSON-файла."""
        if os.path.exists(ConfigHandler.CONFIG_FILE):
            with open(ConfigHandler.CONFIG_FILE, "r") as file:
                return json.load(file)
        # Возвращаем настройки по умолчанию
        return {
            "output_folder": "",
            "last_video_path": "",
            "last_model_path": "",
            "last_logo_path": ""
        }

    @staticmethod
    def save_settings(settings):
        """Сохраняет настройки в JSON-файл."""
        with open(ConfigHandler.CONFIG_FILE, "w") as file:
            json.dump(settings, file, indent=4)


def generate_output_filename(input_path, output_folder):
    """Генерирует имя выходного файла с UUID."""
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    unique_name = f"{base_name}_{uuid.uuid4().hex[:8]}{ext}"
    return os.path.join(output_folder, unique_name)

import os
import subprocess
import sys

def ensure_package_installed(package_name: str) -> None:
    """Убедитесь, что пакет установлен, иначе установите его."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"Пакет {package_name} не установлен. Устанавливаем...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Пакет {package_name} успешно установлен.")

def clean_environment() -> None:
    """Очистить переменные окружения, чтобы избежать конфликтов."""
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    os.environ.pop("QT_PLUGIN_PATH", None)

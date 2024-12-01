import sys
from PyQt5.QtWidgets import QApplication
from environment import ensure_package_installed, clean_environment
from ui import VideoProcessorApp

# Убедимся, что необходимые пакеты установлены
ensure_package_installed("PyQt5")
ensure_package_installed("ultralytics")

# Очистим переменные окружения, чтобы избежать конфликтов
clean_environment()

# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessorApp()
    window.show()
    sys.exit(app.exec_())

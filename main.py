import sys
from PyQt5.QtWidgets import QApplication
from environment import ensure_package_installed, clean_environment
from ui import VideoProcessorUI

# Ensure necessary packages are installed
ensure_package_installed("PyQt5")
ensure_package_installed("ultralytics")

# Clean up environment variables to avoid conflicts
clean_environment()

# Application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessorUI()
    window.show()
    sys.exit(app.exec_())

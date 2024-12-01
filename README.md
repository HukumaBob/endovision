# YOLO Video Processor

A PyQt5 application for processing videos using a YOLO model. This tool allows you to load a video file, apply a YOLO model for object detection, and save the processed video with detected objects annotated.

---

## Features

- **Video Input**: Load videos in formats like `.mp4`, `.avi`, `.webm`, etc.
- **YOLO Model Support**: Use any YOLO `.pt` model file for processing.
- **Class Names**: Optionally load a JSON file with class names for better labeling.
- **Video Preview**: Preview the processing in real time within the app.
- **Save Processed Video**: Export the processed video with bounding boxes and class labels.
- **User-Friendly Interface**: Intuitive and structured UI for easy navigation.

---

## Requirements

- Python 3.8+
- PyQt5
- OpenCV
- Ultralytics YOLO

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hukumabob/endovision.git
   cd yolo-video-processor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Use the interface to:
   - Select a video file.
   - Select a YOLO `.pt` model file.
   - Optionally, select a JSON file with class names.
   - Specify the output path for the processed video.

3. Click **"Start Processing"** to begin.

---

## Directory Structure

```
endovision/
│
├── main.py               # Application entry point
├── ui.py                 # UI components and layout
├── video_processor.py    # Core video processing logic
├── requirements.txt      # Required Python libraries
├── README.md             # Project documentation
└── assets/               # Optional: Icons or images for the app
```

---

## Screenshots

### Main Interface

![Main Interface](/assets/main.png)  
*Example of the application's main window.*

---

## Dependencies

The following Python libraries are required:
- [PyQt5](https://pypi.org/project/PyQt5/) - For building the user interface.
- [OpenCV](https://pypi.org/project/opencv-python/) - For video handling and frame processing.
- [Ultralytics YOLO](https://pypi.org/project/ultralytics/) - For object detection.

---

## Troubleshooting

- If the YOLO model is not loading, ensure the `.pt` file is compatible with the Ultralytics library.
- If you encounter video-related issues, verify that your OpenCV installation supports the required video codecs.
- If you have problems installing cv2 and PyQt5 together, you might want to try this:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO library.
- The Python and OpenCV communities for making such amazing tools.

---

### Contact

For questions or feedback, please contact **Nikita Bogatkov** at `ldc.endoscopy@gmail.com`.
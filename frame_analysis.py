import cv2
from collections import deque

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def find_sharpest_frame(frames):
    max_sharpness = 0
    sharpest_frame = None
    for frame in frames:
        sharpness = calculate_sharpness(frame)
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            sharpest_frame = frame
    return sharpest_frame

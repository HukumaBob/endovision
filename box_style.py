import cv2
import numpy as np

THICKNESS = 1

class RoundedBox:
    """Draws a rectangle with rounded corners."""
    def __init__(self, color, thickness=THICKNESS, radius=10):
        self.color = color
        self.thickness = thickness
        self.radius = radius

    def draw(self, frame, x1, y1, x2, y2, label):
        """Draws the rounded rectangle and label on the frame."""
        radius = self.radius

        # Draw straight lines
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), self.color, self.thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), self.color, self.thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), self.color, self.thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), self.color, self.thickness)

        # Draw rounded corners
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, self.color, self.thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, self.color, self.thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, self.color, self.thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, self.color, self.thickness, lineType=cv2.LINE_AA)

        # Add label above the box
        if label:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_x = x1 + 5  # Offset from the top-left corner
            label_y = y1 - 10  # Offset above the box
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, self.thickness, lineType=cv2.LINE_AA)


class DashedBox(RoundedBox):
    """Draws a dashed rectangle with rounded corners."""
    def __init__(self, color, thickness=THICKNESS, radius=10, dash_length=10):
        super().__init__(color, thickness, radius)
        self.dash_length = dash_length

    def draw(self, frame, x1, y1, x2, y2, label):
        """Draws the dashed rounded rectangle and label on the frame."""
        radius = self.radius

        # Define points for dashed straight lines
        points = [
            ((x1 + radius, y1), (x2 - radius, y1)),  # Top
            ((x2, y1 + radius), (x2, y2 - radius)),  # Right
            ((x2 - radius, y2), (x1 + radius, y2)),  # Bottom
            ((x1, y2 - radius), (x1, y1 + radius))   # Left
        ]

        # Draw dashed straight lines
        for pt1, pt2 in points:
            dist = int(np.linalg.norm(np.array(pt2) - np.array(pt1)))
            for i in range(0, dist, self.dash_length * 2):
                start = (int(pt1[0] + i / dist * (pt2[0] - pt1[0])), int(pt1[1] + i / dist * (pt2[1] - pt1[1])))
                end = (int(pt1[0] + (i + self.dash_length) / dist * (pt2[0] - pt1[0])), 
                       int(pt1[1] + (i + self.dash_length) / dist * (pt2[1] - pt1[1])))
                cv2.line(frame, start, end, self.color, self.thickness, lineType=cv2.LINE_AA)

        # Draw rounded corners
        super().draw(frame, x1, y1, x2, y2, label=None)

        # Add label above the box
        if label:
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_x = x1 + 5  # Offset from the top-left corner
            label_y = y1 - 10  # Offset above the box
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, self.thickness, lineType=cv2.LINE_AA)



class Ellipse:
    """Draws an ellipse around the bounding box."""
    def __init__(self, color, thickness=THICKNESS):
        self.color = color
        self.thickness = thickness

    def draw(self, frame, x1, y1, x2, y2, label):
        """Draws the ellipse and label on the frame."""
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (int((x2 - x1) / 2), int((y2 - y1) / 2))
        cv2.ellipse(frame, center, axes, 0, 0, 360, self.color, self.thickness, lineType=cv2.LINE_AA)

        # Add label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, THICKNESS, lineType=cv2.LINE_AA)

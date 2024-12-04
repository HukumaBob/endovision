import cv2


def overlay_logo(frame, logo_path, x=0, y=0):
    """
    Overlays a logo without transparency on the frame at the specified position.

    :param frame: The original video frame (BGR).
    :param logo_path: Path to the logo image file.
    :param x: X-coordinate of the top-left corner where the logo will be placed.
    :param y: Y-coordinate of the top-left corner where the logo will be placed.
    """
    # Load the logo
    logo = cv2.imread(logo_path)
    if logo is None:
        print("Error: Unable to load logo image.")
        return frame

    # Extract dimensions
    h_logo, w_logo, _ = logo.shape
    h_frame, w_frame, _ = frame.shape

    # Ensure the logo fits within the frame
    if x + w_logo > w_frame or y + h_logo > h_frame:
        print("Error: Logo does not fit in the frame at the given position.")
        return frame

    # Overlay the logo directly
    frame[y:y + h_logo, x:x + w_logo] = logo

    return frame

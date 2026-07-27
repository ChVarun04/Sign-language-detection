import cv2
import numpy as np

from backend.config import IMAGE_SIZE, PADDING


def prepare_hand(image_rgb, hand_landmarks):
    """
    Detects the hand using MediaPipe landmarks and prepares it
    exactly like the training notebook.

    Returns
    -------
    hand_crop : RGB cropped hand image
    processed : 64x64 grayscale normalized image
    features  : Flattened feature vector (1 x 4096)
    bbox      : Bounding box coordinates
    """

    h, w = image_rgb.shape[:2]

    # ---------------------------------------
    # Landmark coordinates
    # ---------------------------------------
    x_points = [int(lm.x * w) for lm in hand_landmarks]
    y_points = [int(lm.y * h) for lm in hand_landmarks]

    xmin = min(x_points)
    xmax = max(x_points)

    ymin = min(y_points)
    ymax = max(y_points)

    # ---------------------------------------
    # Add padding
    # ---------------------------------------
    xmin -= PADDING
    ymin -= PADDING

    xmax += PADDING
    ymax += PADDING

    xmin = max(0, xmin)
    ymin = max(0, ymin)

    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # ---------------------------------------
    # Make square crop
    # ---------------------------------------
    box_width = xmax - xmin
    box_height = ymax - ymin

    side = max(box_width, box_height)

    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    xmin = max(0, center_x - side // 2)
    xmax = min(w, center_x + side // 2)

    ymin = max(0, center_y - side // 2)
    ymax = min(h, center_y + side // 2)

    # ---------------------------------------
    # Crop hand
    # ---------------------------------------
    hand_crop = image_rgb[ymin:ymax, xmin:xmax]

    if hand_crop.size == 0:
        raise ValueError("Empty hand crop.")

    # ---------------------------------------
    # Resize
    # ---------------------------------------
    # Keep aspect ratio using padding
    h_crop, w_crop = hand_crop.shape[:2]

    side = max(h_crop, w_crop)

    square = np.full((side, side, 3), 255, dtype=np.uint8)

    y_offset = (side - h_crop) // 2
    x_offset = (side - w_crop) // 2

    square[
        y_offset:y_offset + h_crop,
        x_offset:x_offset + w_crop
    ] = hand_crop

    hand_crop = cv2.resize(
        square,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------------
    # Convert to grayscale
    # (exactly like notebook)
    # ---------------------------------------
    gray = cv2.cvtColor(
        hand_crop,
        cv2.COLOR_RGB2GRAY
    )

    # ---------------------------------------
    # Normalize
    # ---------------------------------------
    processed = gray.astype(np.float32) / 255.0

    # ---------------------------------------
    # Flatten
    # ---------------------------------------
    features = processed.flatten().reshape(1, -1)

    bbox = (xmin, ymin, xmax, ymax)

    return hand_crop, processed, features, bbox
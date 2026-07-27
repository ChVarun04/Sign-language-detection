from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from backend.config import (
    TASK_MODEL_PATH,
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


def create_hand_landmarker():
    """
    Create a MediaPipe Hand Landmarker in IMAGE mode.
    """

    options = vision.HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(TASK_MODEL_PATH)
        ),

        running_mode=vision.RunningMode.IMAGE,

        num_hands=MAX_HANDS,

        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,

        min_tracking_confidence=MIN_TRACKING_CONFIDENCE

    )

    return vision.HandLandmarker.create_from_options(options)
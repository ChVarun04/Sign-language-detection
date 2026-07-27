from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "sign_language_LR_pipeline.pkl"

TASK_MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"

IMAGE_SIZE = (64,64)

PADDING = 45

MAX_HANDS = 1

MIN_DETECTION_CONFIDENCE = 0.6

MIN_TRACKING_CONFIDENCE = 0.6
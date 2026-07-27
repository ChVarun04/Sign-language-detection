import joblib
import numpy as np

from backend.config import MODEL_PATH


class GesturePredictor:
    """
    Loads the trained ML pipeline and predicts
    the sign language gesture.
    """

    def __init__(self):

        self.pipeline = joblib.load(MODEL_PATH)

    def predict(self, features):
        """
        Parameters
        ----------
        features : ndarray
            Shape (1,4096)

        Returns
        -------
        prediction : str
        confidence : float or None
        """

        prediction = self.pipeline.predict(features)[0]

        confidence = None

        # Some models (Logistic Regression, Random Forest, etc.)
        # support predict_proba(). KNN also supports it.
        if hasattr(self.pipeline, "predict_proba"):

            probabilities = self.pipeline.predict_proba(features)[0]

            confidence = float(np.max(probabilities) * 100)

        return prediction, confidence


# -------------------------------------------------
# Load predictor only once
# -------------------------------------------------

_predictor = GesturePredictor()


def predict_gesture(features):
    """
    Wrapper function used by app.py
    """

    return _predictor.predict(features)
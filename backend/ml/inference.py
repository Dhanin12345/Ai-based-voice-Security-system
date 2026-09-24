from typing import Dict, Any, Tuple
import numpy as np
from backend.ml.predictor import predictor
from backend.utils.logger import logger

class InferenceEngine:
    """
    Inference Engine using pre-trained ML classifier and feature scaler.
    """
    def predict(self, feature_vector: np.ndarray, feature_dict: Dict[str, Any] = None) -> Tuple[float, float]:
        """
        Runs model prediction on feature vector.
        Returns tuple: (human_probability, synthetic_probability)
        """
        try:
            if predictor.scaler is not None and predictor.model is not None:
                scaled_vector = predictor.scaler.transform([feature_vector])
                probs = predictor.model.predict_proba(scaled_vector)[0]
                human_prob = float(probs[0])
                synthetic_prob = float(probs[1])
            else:
                probs = predictor.model.predict_proba(feature_vector, feature_dict=feature_dict)[0]
                human_prob = float(probs[0])
                synthetic_prob = float(probs[1])

            total = human_prob + synthetic_prob
            if total > 0:
                human_prob /= total
                synthetic_prob /= total

            return round(human_prob, 4), round(synthetic_prob, 4)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return 0.5, 0.5

inference_engine = InferenceEngine()

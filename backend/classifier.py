"""
classifier.py — BiLSTM fault classification wrapper.
"""
import numpy as np
from models_loader import store
from config import FAULT_NAMES, WINDOW_SIZE


def classify_window(window: np.ndarray) -> tuple[int, float, dict]:
    """
    Classify a (WINDOW_SIZE, n_features) sensor window.

    Returns:
        fault_class  (int):  0=Normal, 1=Rich, 2=Lean, 3=Ignition
        confidence   (float): probability of the predicted class (0-1)
        probabilities (dict): class_name -> probability
    """
    if window.shape[0] != WINDOW_SIZE:
        raise ValueError(f"Expected window of {WINDOW_SIZE} steps, got {window.shape[0]}")

    x = window[np.newaxis, ...].astype(np.float32)      # (1, 30, n_features)
    probs = store.classifier.predict(x, verbose=0)[0]    # (4,)

    fault_class = int(np.argmax(probs))
    confidence  = float(probs[fault_class])
    prob_dict   = {FAULT_NAMES[i]: float(probs[i]) for i in range(4)}

    return fault_class, confidence, prob_dict

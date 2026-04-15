"""
models_loader.py — Load all ML models and metadata once at startup.
Exposes a single ModelStore instance imported by all other modules.
"""
import json
import numpy as np
import os

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from config import (BILSTM_PATH, TWIN_PATH, THRESHOLDS_PATH,
                    TWIN_META_PATH, SHAP_CACHE_PATH, DATASET_META_PATH)


class ModelStore:
    """Holds all loaded models and metadata. Loaded once at startup."""

    def __init__(self):
        self.classifier   = None
        self.twin         = None
        self.thresholds   = None
        self.twin_meta    = None
        self.shap_cache   = None
        self.dataset_meta = None
        self._loaded      = False

    def load(self):
        if self._loaded:
            return
        print("[ModelStore] Loading BiLSTM classifier...")
        self.classifier = tf.keras.models.load_model(BILSTM_PATH, compile=False)

        print("[ModelStore] Loading LSTM digital twin...")
        self.twin = tf.keras.models.load_model(TWIN_PATH, compile=False)

        print("[ModelStore] Loading metadata...")
        with open(THRESHOLDS_PATH)   as f: self.thresholds   = json.load(f)
        with open(TWIN_META_PATH)    as f: self.twin_meta     = json.load(f)
        with open(SHAP_CACHE_PATH)   as f: self.shap_cache    = json.load(f)
        with open(DATASET_META_PATH) as f: self.dataset_meta  = json.load(f)

        # Warm up models (first inference is slow due to TF graph compilation)
        dummy_clf  = np.zeros((1, 30, 13), dtype=np.float32)
        dummy_twin = np.zeros((1, 30, 15), dtype=np.float32)
        self.classifier.predict(dummy_clf,  verbose=0)
        self.twin.predict(dummy_twin, verbose=0)

        self._loaded = True
        print("[ModelStore] All models loaded and warmed up.")

    @property
    def is_loaded(self):
        return self._loaded

    def get_feature_cols(self, engine_id: str) -> list:
        return self.dataset_meta[engine_id]["feature_cols"]

    def get_shap_for_class(self, fault_class: int) -> list:
        """Return top SHAP features for a fault class (pre-computed cache)."""
        return self.shap_cache[str(fault_class)]["top_features"]

    def get_bilstm_f1(self) -> float:
        return float(self.thresholds.get("macro_f1", 0.0))

    def get_twin_rmse(self) -> float:
        return float(self.twin_meta.get("overall_rmse", 0.0))


# Module-level singleton
store = ModelStore()

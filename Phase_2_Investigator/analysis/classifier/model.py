"""Mechanism classifier: gradient boosting on interpretable group features.

Trained on synthetic ground truth (training/train_mechanism_classifier.py).
If scikit-learn or the trained model file is missing, a transparent
rule-based fallback is used so the rest of the pipeline still runs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from sim_env.spatter.constants import MECHANISMS

from .features import FEATURE_NAMES, group_features

DEFAULT_MODEL = Path(__file__).parent / "models" / "mechanism_gb.pkl"


class MechanismClassifier:
    def __init__(self, model=None, classes=MECHANISMS):
        self.model = model
        self.classes = list(classes)

    # -------------------------------------------------- training / io
    @classmethod
    def train(cls, X, y, seed=0, **kw):
        from sklearn.ensemble import GradientBoostingClassifier

        params = dict(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9, random_state=seed)
        params.update(kw)
        m = GradientBoostingClassifier(**params)
        m.fit(np.asarray(X), np.asarray(y))
        return cls(m, m.classes_)

    def save(self, path=DEFAULT_MODEL):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "classes": self.classes, "features": FEATURE_NAMES}, f)
        return path

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        """Trained model if available, otherwise the rule-based fallback."""
        path = Path(path)
        if path.exists():
            with open(path, "rb") as f:
                d = pickle.load(f)
            if d.get("features") == FEATURE_NAMES:
                return cls(d["model"], d["classes"])
        return cls(None)

    @property
    def is_trained(self):
        return self.model is not None

    # -------------------------------------------------- prediction
    def predict_proba_features(self, feats):
        if self.model is not None:
            x = np.array([[feats[k] for k in FEATURE_NAMES]])
            p = self.model.predict_proba(x)[0]
            return {c: float(pi) for c, pi in zip(self.classes, p)}
        return rule_based(feats)

    def predict_group(self, group):
        feats = group_features(group)
        probs = self.predict_proba_features(feats)
        return max(probs, key=probs.get), probs, feats

    def feature_importance(self):
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(sorted(zip(FEATURE_NAMES, map(float, self.model.feature_importances_)),
                           key=lambda kv: -kv[1]))


def rule_based(f):
    """Hand-written scores from textbook descriptions (fallback only)."""
    s = {m: 0.1 for m in MECHANISMS}
    if f["max_area_cm2"] > 15 and f["log_n"] < 1.5:
        s["transfer"] += 3
    if f["frac_round"] > 0.6 and f["width_med_mm"] > 6:
        s["drip"] += 2
    if f["frac_ceiling"] > 0.2 or (f["along_axis"] > 0.8 and f["linearity"] > 0.6):
        s["castoff"] += 1.5
    if f["convergence_frac"] > 0.4 and f["width_med_mm"] < 6:
        s["impact"] += 1.5
    if f["width_med_mm"] > 7 and f["linearity"] > 0.4:
        s["projected"] += 1.2
    tot = sum(s.values())
    return {k: v / tot for k, v in s.items()}

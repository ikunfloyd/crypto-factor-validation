#!/usr/bin/env python3
"""ModelInference / MoE 测试"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_core.model_inference import ModelInference

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "data" / "models"
HAVE_MODELS = all((MODEL_DIR / m / f"{m}_model.txt").exists()
                  for m in ("hunter", "sniper", "defender"))


def _engine() -> ModelInference:
    # point at a nonexistent dir so no models load -> exercises the pipeline
    # plumbing (feature prep, gating, signal generation) without artifacts.
    # fallback_mode=False keeps it from trying to mkdir the model directory.
    return ModelInference(model_dir="/nonexistent-model-dir", fallback_mode=False)


def test_feature_contract_matches_training():
    """The single most important invariant: the inference feature list must be
    byte-for-byte identical (name + order) to what train_models.py trained on.
    A mismatch is what silently broke MoE prediction historically."""
    from quant_core.training.train_models import ModelTrainer

    inf = _engine().feature_columns
    trn = ModelTrainer(str(MODEL_DIR)).feature_columns
    assert inf == trn, "feature list drift between inference and training"
    assert len(inf) == 17


def test_prepare_features_dimension():
    eng = _engine()
    factors = pl.DataFrame({"rsi_14": [55.0], "macd_hist": [0.01],
                            "bollinger_pctb": [0.5]})
    feats = eng.prepare_features(factors)
    assert feats is not None
    assert feats.shape == (1, len(eng.feature_columns))


def test_predict_structure_with_fallback():
    eng = _engine()
    feats = np.zeros((1, len(eng.feature_columns)))
    result = eng.predict_with_moe(feats, "trending")
    assert "signal" in result
    assert result["signal"]["direction"] in {"LONG", "SHORT", "NEUTRAL"}
    assert 0.0 <= result["signal"]["position_size"] <= 1.0


@pytest.mark.skipif(not HAVE_MODELS,
                    reason="trained models absent (run scripts/demo.py first)")
def test_real_models_load_and_predict():
    eng = ModelInference(model_dir=str(MODEL_DIR))
    assert set(eng.models.keys()) == {"hunter", "sniper", "defender"}
    feats = np.random.default_rng(0).normal(size=(1, len(eng.feature_columns)))
    result = eng.predict_with_moe(feats, "trending")
    assert "error" not in result
    assert np.isfinite(result["final_score"])

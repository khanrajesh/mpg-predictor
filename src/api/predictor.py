"""Runtime loader and inference helper for the exported XGBoost MPG model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBRegressor


def _normalize_origin(value: object) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "1": "usa",
        "2": "europe",
        "3": "asia",
        "usa": "usa",
        "united states": "usa",
        "europe": "europe",
        "eu": "europe",
        "asia": "asia",
        "japan": "asia",
    }
    return mapping.get(raw, raw if raw else "unknown")


class MPGPredictor:
    """Loads model artifacts and serves single-record predictions."""

    MPG_TO_KM_PER_L = 0.425143707

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Run `python -m src.train.export_xgboost_json` first."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}. "
                "Run `python -m src.train.export_xgboost_json` first."
            )

        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.numeric_columns: list[str] = list(self.metadata["numeric_columns"])
        self.categorical_columns: list[str] = list(self.metadata["categorical_columns"])
        self.feature_columns: list[str] = list(self.metadata["feature_columns"])
        self.numeric_medians: dict[str, float] = {
            k: float(v) for k, v in self.metadata["numeric_medians"].items()
        }
        self.category_options: dict[str, list[str]] = {
            k: list(v) for k, v in self.metadata.get("category_options", {}).items()
        }

        self.model = XGBRegressor()
        self.model.load_model(str(self.model_path))

    def _default_category(self, col: str) -> str:
        options = self.category_options.get(col, [])
        return options[0] if options else "unknown"

    def _payload_to_features(self, payload: dict[str, Any]) -> pd.DataFrame:
        row: dict[str, Any] = {}

        for col in self.numeric_columns:
            raw = payload.get(col, self.numeric_medians.get(col, 0.0))
            if raw in (None, ""):
                raw = self.numeric_medians.get(col, 0.0)
            try:
                row[col] = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Field '{col}' must be numeric.") from exc

        for col in self.categorical_columns:
            raw = payload.get(col, self._default_category(col))
            text = str(raw).strip().lower()
            if not text:
                text = self._default_category(col)
            if col == "origin":
                text = _normalize_origin(text)
            row[col] = text

        frame = pd.DataFrame([row], columns=self.numeric_columns + self.categorical_columns)
        encoded = pd.get_dummies(frame, columns=self.categorical_columns, dtype=float)
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
        return encoded.astype(float)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = self._payload_to_features(payload)
        predicted_mpg = float(self.model.predict(features)[0])
        predicted_km_per_l = predicted_mpg * self.MPG_TO_KM_PER_L
        return {
            "predicted_mpg": round(predicted_mpg, 3),
            "predicted_km_per_l": round(predicted_km_per_l, 3),
            "model_name": self.metadata.get("model_name"),
            "model_version": self.metadata.get("model_version"),
            "distance_unit": "miles",
            "volume_unit": "US_gallon",
        }

    def schema(self) -> dict[str, Any]:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_medians": self.numeric_medians,
            "category_options": self.category_options,
            "default_payload": self.metadata.get("default_payload", {}),
            "model_name": self.metadata.get("model_name"),
            "model_version": self.metadata.get("model_version"),
        }


@lru_cache(maxsize=1)
def get_predictor() -> MPGPredictor:
    base_dir = Path(__file__).resolve().parents[2]
    return MPGPredictor(
        model_path=base_dir / "models" / "xgboost_mpg.json",
        metadata_path=base_dir / "models" / "model_metadata.json",
    )

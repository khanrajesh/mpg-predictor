"""Train an XGBoost MPG model and export it as JSON + metadata.

Usage:
    python -m src.train.export_xgboost_json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

NUMERIC_COLUMNS = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "length",
    "width",
    "height",
    "wheelbase",
    "num_gears",
    "drag_coefficient",
    "frontal_area",
    "compression_ratio",
    "top_speed",
]

CATEGORICAL_COLUMNS = [
    "origin",
    "engine_type",
    "fuel_system",
    "transmission_type",
    "drive_type",
]


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


def _prepare_dataframe(
    frame: pd.DataFrame,
    numeric_medians: dict[str, float] | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, list[str]]]:
    work = frame.copy()

    for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        if col not in work.columns:
            work[col] = np.nan

    for col in NUMERIC_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["origin"] = work["origin"].map(_normalize_origin)

    for col in CATEGORICAL_COLUMNS:
        work[col] = (
            work[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
        )

    if fit or numeric_medians is None:
        numeric_medians = {
            col: float(work[col].median(skipna=True))
            for col in NUMERIC_COLUMNS
        }

    for col in NUMERIC_COLUMNS:
        default_val = numeric_medians.get(col, 0.0)
        if pd.isna(default_val):
            default_val = 0.0
        work[col] = work[col].fillna(float(default_val))

    category_options: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLUMNS:
        options = sorted(set(work[col].dropna().astype(str).tolist()))
        if "unknown" not in options:
            options.append("unknown")
        category_options[col] = options

    encoded = pd.get_dummies(
        work[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS],
        columns=CATEGORICAL_COLUMNS,
        dtype=float,
    )
    encoded = encoded.astype(float)

    return encoded, numeric_medians, category_options


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_np = y_true.to_numpy(dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    r2 = float(r2_score(y_true_np, y_pred_np))

    nonzero = y_true_np != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs((y_true_np[nonzero] - y_pred_np[nonzero]) / y_true_np[nonzero])) * 100.0)
    else:
        mape = float("nan")

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train_and_export(data_path: Path, model_dir: Path, model_version: str) -> tuple[Path, Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    raw_df = pd.read_csv(data_path)
    if "mpg" not in raw_df.columns:
        raise ValueError("Column 'mpg' is required in the training dataset.")

    y_all = pd.to_numeric(raw_df["mpg"], errors="coerce")
    valid_mask = y_all.notna()
    if int(valid_mask.sum()) < 100:
        raise ValueError("Not enough valid MPG rows to train a model.")

    features_df = raw_df.loc[valid_mask].copy()
    y_all = y_all.loc[valid_mask]

    x_all, numeric_medians, category_options = _prepare_dataframe(features_df, fit=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.20,
        random_state=42,
    )

    model_params = {
        "objective": "reg:squarederror",
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
    }

    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)

    test_pred = model.predict(x_test)
    metrics = _compute_metrics(y_test, test_pred)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_mpg.json"
    metadata_path = model_dir / "model_metadata.json"

    model.save_model(model_path)

    default_payload = {
        col: round(float(numeric_medians[col]), 4)
        for col in NUMERIC_COLUMNS
    }
    for col in CATEGORICAL_COLUMNS:
        default_payload[col] = (
            category_options[col][0] if category_options[col] else "unknown"
        )

    metadata = {
        "model_name": "xgboost_mpg_regressor",
        "model_version": model_version,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_column": "mpg",
        "data_path": str(data_path.as_posix()),
        "model_file": str(model_path.as_posix()),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "feature_columns": list(x_all.columns),
        "numeric_medians": numeric_medians,
        "category_options": category_options,
        "default_payload": default_payload,
        "rows": {
            "total": int(len(x_all)),
            "train": int(len(x_train)),
            "test": int(len(x_test)),
        },
        "metrics_test": metrics,
        "xgboost_params": model_params,
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    return model_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export XGBoost MPG model to JSON.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to input CSV (defaults to data/raw/automobile_data.csv).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory to save model artifacts (defaults to models/).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0.0",
        help="Version label stored in metadata.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    data_path = args.data_path or (base_dir / "data" / "raw" / "automobile_data.csv")
    model_dir = args.model_dir or (base_dir / "models")

    model_path, metadata_path = train_and_export(
        data_path=data_path,
        model_dir=model_dir,
        model_version=args.model_version,
    )

    print(f"Model exported: {model_path}")
    print(f"Metadata exported: {metadata_path}")


if __name__ == "__main__":
    main()

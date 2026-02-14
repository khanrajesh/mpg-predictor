"""HTTP routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request, url_for

from .predictor import get_predictor

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    admin_url = url_for("admin.index")
    model_ready = True
    model_error = ""

    try:
        schema = get_predictor().schema()
    except Exception as exc:  # pragma: no cover - rendered for runtime setup issues
        model_ready = False
        model_error = str(exc)
        schema = {
            "numeric_columns": [],
            "categorical_columns": [],
            "default_payload": {},
            "category_options": {},
            "model_name": "not_loaded",
            "model_version": "not_loaded",
        }

    return render_template(
        "index.html",
        admin_url=admin_url,
        model_ready=model_ready,
        model_error=model_error,
        numeric_columns=schema["numeric_columns"],
        categorical_columns=schema["categorical_columns"],
        default_payload=schema["default_payload"],
        category_options=schema["category_options"],
        model_name=schema["model_name"],
        model_version=schema["model_version"],
    )


@bp.route("/api/health", methods=["GET"])
def health():
    try:
        schema = get_predictor().schema()
        return jsonify(
            {
                "status": "ok",
                "model_ready": True,
                "model_name": schema["model_name"],
                "model_version": schema["model_version"],
            }
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostic path
        return (
            jsonify(
                {
                    "status": "ok",
                    "model_ready": False,
                    "error": str(exc),
                }
            ),
            503,
        )


@bp.route("/api/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object."}), 400

    try:
        predictor = get_predictor()
        result = predictor.predict(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - runtime setup issues
        return jsonify({"error": str(exc)}), 503

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.feature_extract import FEATURE_COLUMNS


MODEL_COUNT = 9

EXCIPIENT_CLASSES = [
    "Buffer / pH modifier",
    "Sugar stabilizer",
    "Polyol stabilizer",
    "Surfactant",
    "Amino acid stabilizer",
    "Antioxidant",
    "Chelating agent",
    "Salt / ionic strength modifier",
    "Preservative",
]

MODEL_FILENAMES = [
    "00_buffer_ph_modifier.joblib",
    "01_sugar_stabilizer.joblib",
    "02_polyol_stabilizer.joblib",
    "03_surfactant.joblib",
    "04_amino_acid_stabilizer.joblib",
    "05_antioxidant.joblib",
    "06_chelating_agent.joblib",
    "07_salt_ionic_strength_modifier.joblib",
    "08_preservative.joblib",
]

FEATURE_SCHEMA_FILENAME = "feature_columns.json"


@dataclass(frozen=True)
class PredictionResult:
    probabilities: dict[str, float]
    used_mock: bool
    model_files: list[Path]


def _find_model_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    model_files = [model_dir / filename for filename in MODEL_FILENAMES]
    return model_files if all(path.exists() for path in model_files) else []


def load_feature_columns(model_dir: Path) -> list[str] | None:
    schema_path = model_dir / FEATURE_SCHEMA_FILENAME
    if not schema_path.exists():
        return None

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    feature_columns = schema.get("feature_columns")
    if not isinstance(feature_columns, list):
        return None
    if not all(isinstance(name, str) for name in feature_columns):
        return None
    if set(feature_columns) != set(FEATURE_COLUMNS):
        return None

    return feature_columns


def _feature_matrix(
    features: dict[str, float],
    feature_columns: list[str],
) -> list[list[float]]:
    return [[features[name] for name in feature_columns]]


def _clip_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _as_first_value(value: object) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list):
        value = value[0]
    return float(value)


def _probability_from_model(model: object, feature_matrix: list[list[float]]) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_matrix)
        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()
        return _clip_probability(float(probabilities[0][-1]))

    prediction = model.predict(feature_matrix)
    return _clip_probability(_as_first_value(prediction))


def _mock_probabilities(features: dict[str, float], class_names: list[str]) -> dict[str, float]:
    length = features["length"]
    hydrophobic = features["hydrophobic_ratio"]
    charged = features["positive_ratio"] + features["negative_ratio"]
    aromatic = features["aromatic_ratio"]

    raw_scores = [
        0.32 + charged * 0.55,
        0.30 + max(0.0, 1.0 - hydrophobic) * 0.35,
        0.26 + hydrophobic * 0.45,
        0.20 + hydrophobic * 0.50 + aromatic * 0.20,
        0.24 + features["positive_ratio"] * 0.40,
        0.18 + aromatic * 0.65,
        0.16 + features["negative_ratio"] * 0.55,
        0.20 + min(length / 600.0, 1.0) * 0.30,
        0.12 + max(0.0, 1.0 - min(length / 300.0, 1.0)) * 0.20,
    ]
    max_score = max(raw_scores)
    scaled = [score / max_score * 0.92 for score in raw_scores]
    return {
        name: _clip_probability(max(score, 0.03))
        for name, score in zip(class_names, scaled, strict=True)
    }


def predict_probabilities(
    features: dict[str, float],
    model_dir: Path,
    class_names: list[str] | None = None,
) -> PredictionResult:
    names = class_names or EXCIPIENT_CLASSES
    model_files = _find_model_files(model_dir)
    feature_columns = load_feature_columns(model_dir)

    if len(model_files) != MODEL_COUNT or feature_columns is None:
        return PredictionResult(
            probabilities=_mock_probabilities(features, names),
            used_mock=True,
            model_files=model_files,
        )

    try:
        import joblib
    except ImportError:
        return PredictionResult(
            probabilities=_mock_probabilities(features, names),
            used_mock=True,
            model_files=model_files,
        )

    matrix = _feature_matrix(features, feature_columns)
    probabilities: dict[str, float] = {}
    try:
        for class_name, model_file in zip(names, model_files, strict=True):
            model = joblib.load(model_file)
            probabilities[class_name] = _probability_from_model(model, matrix)
    except Exception:
        return PredictionResult(
            probabilities=_mock_probabilities(features, names),
            used_mock=True,
            model_files=model_files,
        )

    return PredictionResult(
        probabilities=probabilities,
        used_mock=False,
        model_files=model_files,
    )

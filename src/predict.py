from __future__ import annotations

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


@dataclass(frozen=True)
class PredictionResult:
    probabilities: dict[str, float]
    used_mock: bool
    model_files: list[Path]


def _find_model_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("*.joblib"))


def _feature_matrix(features: dict[str, float]) -> list[list[float]]:
    return [[features[name] for name in FEATURE_COLUMNS]]


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

    if len(model_files) != MODEL_COUNT:
        return PredictionResult(
            probabilities=_mock_probabilities(features, names),
            used_mock=True,
            model_files=model_files,
        )

    import joblib

    matrix = _feature_matrix(features)
    probabilities: dict[str, float] = {}
    for class_name, model_file in zip(names, model_files, strict=True):
        model = joblib.load(model_file)
        probabilities[class_name] = _probability_from_model(model, matrix)

    return PredictionResult(
        probabilities=probabilities,
        used_mock=False,
        model_files=model_files,
    )

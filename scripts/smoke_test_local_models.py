from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extract import clean_sequence_input, extract_basic_features
from src.predict import (
    EXCIPIENT_CLASSES,
    MODEL_FILENAMES,
    load_feature_columns,
    predict_probabilities,
)


MODEL_DIR = PROJECT_ROOT / "models"

# Non-training smoke-test input. The script reports length only and never prints it.
SMOKE_TEST_SEQUENCE = (
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYWMHWVRQAPGQGLEWMG"
    "YINPSSGYTNYNQKFKDRVTITADKSTSTAYMELSSLRSEDTAVYYCARDY"
    "WGQGTLVTVSSDIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKP"
    "GKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQY"
    "NSYPYTFGQGTKLEIK"
)


def _expected_model_paths() -> list[Path]:
    return [MODEL_DIR / filename for filename in MODEL_FILENAMES]


def _check_model_files() -> list[Path]:
    model_paths = _expected_model_paths()
    missing = [path.name for path in model_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing model files: " + ", ".join(missing))
    return model_paths


def _load_joblib_models(model_paths: list[Path]) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit(
            "joblib is not installed. Install training dependencies with: "
            "pip install -r requirements-train.txt"
        ) from exc

    for model_path in model_paths:
        try:
            joblib.load(model_path)
        except Exception as exc:
            raise SystemExit(f"Could not load {model_path.name}: {exc}") from exc


def main() -> int:
    feature_columns = load_feature_columns(MODEL_DIR)
    if feature_columns is None:
        raise SystemExit("Could not load a valid models/feature_columns.json schema.")

    model_paths = _check_model_files()
    _load_joblib_models(model_paths)

    cleaned_sequence = clean_sequence_input(SMOKE_TEST_SEQUENCE)
    features = extract_basic_features(cleaned_sequence)
    result = predict_probabilities(
        features=features,
        model_dir=MODEL_DIR,
        class_names=EXCIPIENT_CLASSES,
    )

    print(f"Loaded feature columns: {len(feature_columns)}")
    print(f"Loaded joblib models: {len(model_paths)}")
    print(f"Smoke-test sequence length: {len(cleaned_sequence)} aa")
    print(f"used_mock={result.used_mock}")
    print("Probabilities:")
    for class_name in EXCIPIENT_CLASSES:
        print(f"- {class_name}: {result.probabilities[class_name]:.6f}")

    if result.used_mock:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

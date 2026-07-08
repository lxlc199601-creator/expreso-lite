from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extract import FEATURE_COLUMNS, clean_sequence_input, extract_basic_features
from src.predict import EXCIPIENT_CLASSES


DATA_PATH = PROJECT_ROOT / "data" / "training_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORT_DIR / "model_metrics.csv"
FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_columns.json"
RANDOM_STATE = 42

LABEL_TO_CLASS = [
    ("label_buffer_ph_modifier", "Buffer / pH modifier"),
    ("label_sugar_stabilizer", "Sugar stabilizer"),
    ("label_polyol_stabilizer", "Polyol stabilizer"),
    ("label_surfactant", "Surfactant"),
    ("label_amino_acid_stabilizer", "Amino acid stabilizer"),
    ("label_antioxidant", "Antioxidant"),
    ("label_chelating_agent", "Chelating agent"),
    ("label_salt_ionic_strength_modifier", "Salt / ionic strength modifier"),
    ("label_preservative", "Preservative"),
]

REQUIRED_COLUMNS = [
    "protein_id",
    "protein_name",
    "sequence",
    *[label for label, _class_name in LABEL_TO_CLASS],
]


def _check_training_data_exists() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Missing {DATA_PATH}. Please provide real labeled training data at "
            "data/training_data.csv. Do not use generated or placeholder labels."
        )


def _import_training_dependencies():
    try:
        import joblib
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with: "
            "pip install -r requirements-train.txt"
        ) from exc

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "XGBoost is required for training but is not installed. "
            "Install training dependencies with: pip install -r requirements-train.txt"
        ) from exc

    return {
        "accuracy_score": accuracy_score,
        "joblib": joblib,
        "pd": pd,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "XGBClassifier": XGBClassifier,
    }


def _load_training_data(pd):
    data = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise SystemExit(
            "Training data is missing required columns: "
            + ", ".join(missing_columns)
        )

    if data.empty:
        raise SystemExit("Training data is empty. Please provide real labeled rows.")

    return data


def _validate_labels(data) -> None:
    for label_column, class_name in LABEL_TO_CLASS:
        if data[label_column].isna().any():
            raise SystemExit(f"{label_column} for {class_name} contains missing labels.")

        values = data[label_column].dropna()
        invalid_values = sorted(set(values) - {0, 1})
        if invalid_values:
            raise SystemExit(
                f"{label_column} for {class_name} must contain only 0 or 1 values. "
                f"Invalid values: {invalid_values}"
            )

        counts = values.value_counts()
        if counts.get(0, 0) < 2 or counts.get(1, 0) < 2:
            raise SystemExit(
                f"{label_column} for {class_name} needs at least two 0 labels and "
                "two 1 labels so a stratified train/test split can be evaluated."
            )


def _build_feature_frame(data, pd):
    cleaned_sequences = data["sequence"].fillna("").map(clean_sequence_input)
    empty_sequence_rows = data.index[cleaned_sequences == ""].tolist()
    if empty_sequence_rows:
        row_numbers = [row_index + 2 for row_index in empty_sequence_rows]
        raise SystemExit(
            "Some rows have no valid canonical amino-acid sequence after cleaning. "
            f"CSV row numbers: {row_numbers}"
        )

    feature_rows = [extract_basic_features(sequence) for sequence in cleaned_sequences]
    return pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)


def _make_model(XGBClassifier):
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _model_filename(class_name: str, index: int) -> str:
    normalized = (
        class_name.lower()
        .replace(" / ", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{index:02d}_{normalized}.joblib"


def _train_one_model(
    deps,
    features,
    labels,
    class_name: str,
    index: int,
) -> dict[str, float | int | str]:
    x_train, x_test, y_train, y_test = deps["train_test_split"](
        features,
        labels.astype(int),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels.astype(int),
    )

    model = _make_model(deps["XGBClassifier"])
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    model_path = MODEL_DIR / _model_filename(class_name, index)
    deps["joblib"].dump(model, model_path)

    return {
        "class_name": class_name,
        "label_column": LABEL_TO_CLASS[index][0],
        "model_file": model_path.name,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "positive_train_rows": int(y_train.sum()),
        "positive_test_rows": int(y_test.sum()),
        "accuracy": deps["accuracy_score"](y_test, predictions),
        "precision": deps["precision_score"](y_test, predictions, zero_division=0),
        "recall": deps["recall_score"](y_test, predictions, zero_division=0),
        "roc_auc": deps["roc_auc_score"](y_test, probabilities),
    }


def main() -> None:
    _check_training_data_exists()
    deps = _import_training_dependencies()
    pd = deps["pd"]

    data = _load_training_data(pd)
    _validate_labels(data)
    features = _build_feature_frame(data, pd)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    schema = {
        "feature_columns": FEATURE_COLUMNS,
        "label_columns": [label for label, _class_name in LABEL_TO_CLASS],
        "class_names": EXCIPIENT_CLASSES,
    }
    FEATURE_SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    metrics = []
    for index, (label_column, class_name) in enumerate(LABEL_TO_CLASS):
        metrics.append(
            _train_one_model(
                deps=deps,
                features=features,
                labels=data[label_column],
                class_name=class_name,
                index=index,
            )
        )

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)

    print(f"Saved {len(metrics)} models to {MODEL_DIR}")
    print(f"Saved feature schema to {FEATURE_SCHEMA_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

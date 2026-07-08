from __future__ import annotations


def get_training_notice() -> str:
    return (
        "No complete set of 9 trained joblib models was found in models/. "
        "The app is currently showing mock probabilities generated from basic "
        "sequence features. These values are for workflow demonstration only."
    )


def build_recommendation_text(top3: list[dict[str, float | str]], used_mock: bool) -> str:
    names = ", ".join(str(row["Excipient class"]) for row in top3)
    if used_mock:
        return (
            f"Current top 3: {names}. These results are mock outputs for the "
            "demo interface and prediction pipeline; they are not validated "
            "model conclusions."
        )

    return (
        f"Current top 3 from the loaded models: {names}. These probabilities "
        "come from the 9 local joblib models in the models directory."
    )

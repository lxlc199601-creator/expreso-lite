from pathlib import Path

import streamlit as st

from src.explain import build_recommendation_text, get_training_notice
from src.feature_extract import FEATURE_COLUMNS, clean_sequence_input, extract_basic_features
from src.predict import EXCIPIENT_CLASSES, MODEL_COUNT, predict_probabilities


st.set_page_config(
    page_title="ExPreSo-Lite",
    page_icon="E",
    layout="wide",
)


DEFAULT_SEQUENCE = """>example_protein
MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"""


def build_feature_rows(features: dict[str, float]) -> list[dict[str, float | str]]:
    return [{"feature": name, "value": features[name]} for name in FEATURE_COLUMNS]


def build_prediction_rows(predictions: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {"Excipient class": name, "Probability": prob}
        for name, prob in sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    ]


def main() -> None:
    st.title("ExPreSo-Lite")
    st.caption("Fast mode MVP for protein excipient recommendation")

    with st.sidebar:
        st.header("Mode")
        st.info(
            "This first version uses basic sequence features with either 9 local "
            "joblib models or mock probabilities when models are unavailable."
        )
        st.markdown(f"Expected model files: `{MODEL_COUNT}` `.joblib` files")
        st.markdown("Model directory: `models/`")

    sequence_input = st.text_area(
        "Protein FASTA or plain sequence",
        value=DEFAULT_SEQUENCE,
        height=220,
        placeholder="Paste a FASTA record or a plain amino-acid sequence.",
    )

    if not st.button("Run recommendation", type="primary"):
        st.stop()

    cleaned_sequence = clean_sequence_input(sequence_input)
    if not cleaned_sequence:
        st.error(
            "No valid standard amino-acid sequence was detected. "
            "Please enter letters from the 20 canonical amino acids."
        )
        st.stop()

    features = extract_basic_features(cleaned_sequence)
    prediction_result = predict_probabilities(
        features=features,
        model_dir=Path(__file__).parent / "models",
        class_names=EXCIPIENT_CLASSES,
    )

    if prediction_result.used_mock:
        st.warning(get_training_notice())
    else:
        st.success("Loaded local joblib models and generated model predictions.")

    st.subheader("Cleaned sequence")
    st.caption(f"Valid length: {len(cleaned_sequence)} aa")

    prediction_rows = build_prediction_rows(prediction_result.probabilities)
    feature_rows = build_feature_rows(features)
    top3_rows = prediction_rows[:3]

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Ranked probabilities")
        st.dataframe(
            [
                {**row, "Probability": f"{row['Probability']:.2%}"}
                for row in prediction_rows
            ],
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Top 3 recommended excipients")
        for rank, row in enumerate(top3_rows, start=1):
            st.metric(
                label=f"Top {rank}: {row['Excipient class']}",
                value=f"{row['Probability']:.1%}",
            )

    st.subheader("Probability bar chart")
    st.bar_chart(prediction_rows, x="Excipient class", y="Probability")

    with st.expander("View extracted sequence features"):
        st.dataframe(
            [
                {**row, "value": f"{row['value']:.6f}"}
                for row in feature_rows
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Recommendation note")
    st.write(build_recommendation_text(top3_rows, prediction_result.used_mock))


if __name__ == "__main__":
    main()

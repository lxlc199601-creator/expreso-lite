# AGENTS.md

## Project scope

ExPreSo-Lite is a lightweight Streamlit demo for protein excipient recommendation.

The first version is strictly a Fast mode MVP:
- Accept one protein FASTA record or one plain protein sequence.
- Clean the input to the 20 canonical amino acids.
- Extract basic sequence features only.
- Produce probabilities for 9 excipient classes.
- Use mock probabilities when trained models are missing.

## Do not add in this MVP

Do not implement or add runtime dependencies for:
- Protein language models
- 3D structure prediction or parsing
- SHAP explanations
- Database storage
- Authentication
- Industrial deployment features
- Medical or pharmaceutical claims

## Dependency rules

`requirements.txt` is for running the Streamlit demo and must stay lightweight.

Keep large or training-only packages out of `requirements.txt`. If training tools are needed, put them in `requirements-train.txt`.

In particular:
- Do not require `xgboost` to preview the demo.
- Do not require `scikit-learn`, `pandas`, or `numpy` unless the running demo truly needs them.
- The demo must run with mock probabilities when no models are present.

## Code organization

Keep `app.py` thin. UI orchestration belongs there, while core logic belongs in:
- `src/feature_extract.py` for sequence cleaning and feature extraction
- `src/predict.py` for model discovery, loading, and mock probabilities
- `src/explain.py` for user-facing notices and recommendation text

Do not expand the feature scope without an explicit user request.

## Expected project files

```text
expreso_lite/
  app.py
  requirements.txt
  requirements-train.txt
  README.md
  AGENTS.md
  src/
    feature_extract.py
    predict.py
    explain.py
  models/
    README.md
```

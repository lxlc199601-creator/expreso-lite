# ExPreSo-Lite

ExPreSo-Lite is a lightweight Streamlit MVP for protein excipient recommendation.
The current training-data template is oriented toward antibody formulations,
including monoclonal antibodies, bispecific antibodies, heavy chains, light
chains, and related chain formats. The app accepts a protein FASTA record or
plain amino-acid sequence, extracts basic sequence features, and ranks
probabilities for 9 excipient classes.

This repository is currently a mock demo. It does not include trained model
files, so the deployed app will show mock probabilities for workflow preview
only, not real model predictions.

This first version only implements Fast mode. It does not run large sequence
models, structure prediction, structure parsing, or local large model inference.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` only contains lightweight packages needed to preview the
Streamlit demo. Training-only packages are listed separately in
`requirements-train.txt`.

## Model behavior

If `models/` contains exactly 9 `.joblib` files, the app loads them in sorted
filename order and maps them to the 9 excipient classes in `src/predict.py`.

If trained models are missing, the app uses mock probabilities and displays a
clear warning in the interface. Mock outputs are only for demonstrating the
workflow and should not be treated as scientific or pharmaceutical claims.

## Feature extraction

The MVP extracts:

- Cleaned sequence length
- Amino-acid composition ratios for the 20 canonical residues
- Positive, negative, hydrophobic, and aromatic residue ratios

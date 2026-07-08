# Models

Place 9 trained `.joblib` models in this directory.

The demo expects one binary model per excipient class. Files are loaded in sorted
filename order and mapped to:

1. Buffer / pH modifier
2. Sugar stabilizer
3. Polyol stabilizer
4. Surfactant
5. Amino acid stabilizer
6. Antioxidant
7. Chelating agent
8. Salt / ionic strength modifier
9. Preservative

Expected feature columns are defined in `src/feature_extract.py` as
`FEATURE_COLUMNS`.

If this directory does not contain exactly 9 `.joblib` files, the app will use
mock probabilities and show a clear training notice.

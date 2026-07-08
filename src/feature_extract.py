from __future__ import annotations

from collections import Counter


STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
HYDROPHOBIC_AA = set("AILMFWVY")
AROMATIC_AA = set("FWY")

AA_FEATURE_COLUMNS = [f"aa_{aa}_ratio" for aa in STANDARD_AMINO_ACIDS]
FEATURE_COLUMNS = (
    ["length"]
    + AA_FEATURE_COLUMNS
    + [
        "positive_ratio",
        "negative_ratio",
        "hydrophobic_ratio",
        "aromatic_ratio",
    ]
)


def clean_sequence_input(raw_text: str) -> str:
    """Remove FASTA headers and non-standard residues, preserving 20 canonical amino acids."""
    sequence_parts: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">"):
            continue
        sequence_parts.append(stripped)

    merged = "".join(sequence_parts).upper()
    allowed = set(STANDARD_AMINO_ACIDS)
    return "".join(residue for residue in merged if residue in allowed)


def _group_ratio(sequence: str, group: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(1 for residue in sequence if residue in group) / len(sequence)


def extract_basic_features(sequence: str) -> dict[str, float]:
    """Extract Fast-mode sequence descriptors for model inference."""
    cleaned = clean_sequence_input(sequence)
    length = len(cleaned)
    counts = Counter(cleaned)

    features: dict[str, float] = {"length": float(length)}
    for aa in STANDARD_AMINO_ACIDS:
        features[f"aa_{aa}_ratio"] = counts.get(aa, 0) / length if length else 0.0

    features["positive_ratio"] = _group_ratio(cleaned, POSITIVE_AA)
    features["negative_ratio"] = _group_ratio(cleaned, NEGATIVE_AA)
    features["hydrophobic_ratio"] = _group_ratio(cleaned, HYDROPHOBIC_AA)
    features["aromatic_ratio"] = _group_ratio(cleaned, AROMATIC_AA)

    return {name: features[name] for name in FEATURE_COLUMNS}


clean_fasta = clean_sequence_input

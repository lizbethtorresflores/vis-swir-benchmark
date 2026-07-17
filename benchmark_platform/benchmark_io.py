"""Lecture et validation des homographies envoyees au benchmark.

Accepte des matrices depuis du texte, des fichiers .txt ou .npy, et rejette les
matrices dont la forme, les valeurs ou la stabilite numerique sont invalides.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np


def _check_homography_stability(H: np.ndarray) -> None:
    """Rejette les matrices singulieres, non inversibles ou numeriquement instables."""
    det = float(np.linalg.det(H))
    if abs(det) < 1e-10:
        raise ValueError(
            "Invalid homography: determinant is too close to zero; the matrix is singular or nearly singular."
        )
    try:
        condition = float(np.linalg.cond(H))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Invalid homography: condition number could not be computed.") from exc
    if not np.isfinite(condition):
        raise ValueError("Invalid homography: condition number is not finite; the matrix is unstable.")
    if condition > 1e12:
        raise ValueError(
            f"Invalid homography: matrix is extremely unstable (condition number {condition:.3e})."
        )
    try:
        np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Invalid homography: matrix is not invertible.") from exc


def validate_homography(H: np.ndarray) -> np.ndarray:
    """Valide une matrice 3x3 et la normalise pour que H[2,2] vaille 1 si possible."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Invalid homography: expected a 3x3 matrix, got shape {H.shape}.")
    if not np.isfinite(H).all():
        bad = np.argwhere(~np.isfinite(H))
        raise ValueError(
            "Invalid homography: matrix contains NaN or Inf values "
            f"at indices {bad.tolist()}."
        )
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    _check_homography_stability(H)
    return H


def parse_homography_text(text: str) -> np.ndarray:
    """Convertit le texte colle par l'utilisateur en homographie validee."""
    rows = []
    if not text or not text.strip():
        raise ValueError("Invalid homography text: paste a 3x3 matrix before submitting.")
    for line_number, raw_line in enumerate(text.strip().splitlines(), start=1):
        line = raw_line.replace(",", " ").strip()
        if not line:
            continue
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as exc:
            raise ValueError(f"Invalid homography text: non-numeric value on line {line_number}.") from exc
    return validate_homography(np.array(rows, dtype=np.float64))


def read_homography_file(uploaded_file) -> np.ndarray:
    """Lit une homographie envoyee depuis l'interface Streamlit."""
    if uploaded_file is None:
        raise ValueError("Invalid submission: upload H_global.txt or H_global.npy.")
    name = Path(uploaded_file.name).name.lower()
    data = uploaded_file.read()
    if name.endswith(".npy"):
        return validate_homography(np.load(BytesIO(data), allow_pickle=False))
    if name.endswith(".txt"):
        return parse_homography_text(data.decode("utf-8"))
    raise ValueError("Unsupported homography format. Use H_global.txt or H_global.npy.")


def read_homography_path(path: Path) -> np.ndarray:
    """Lit une homographie depuis le disque pour methodes existantes ou tests locaux."""
    if path.suffix.lower() == ".npy":
        return validate_homography(np.load(path, allow_pickle=False))
    if path.suffix.lower() == ".txt":
        return parse_homography_text(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported homography format: {path.suffix}")

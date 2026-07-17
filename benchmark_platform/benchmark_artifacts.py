"""Helpers pour empaqueter les artefacts prives d'une submission.

Ce module ne calcule pas de metriques ; il cree seulement des fichiers
telechargeables pour que le submitter recupere ses images alignees et artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_zip_from_dirs(output_path: Path, roots: list[Path], base_dir: Path) -> Path:
    """Cree un ZIP prive avec les dossiers d'artefacts d'une submission."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        for root in roots:
            if not root.exists():
                continue
            for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
                zf.write(file_path, file_path.relative_to(base_dir))
    return output_path


def create_private_aligned_outputs_zip(method_path: Path) -> Path:
    """Archive les images alignees generees pour la submission courante."""
    output_path = method_path / "downloads" / "own_aligned_outputs.zip"
    roots = [
        method_path / "split_images",
        method_path / "concatenate_images",
    ]
    return create_zip_from_dirs(output_path, roots, method_path)


def write_private_artifacts_manifest(method_path: Path, artifacts: dict) -> None:
    """Enregistre les artefacts telechargeables uniquement par le submitter."""
    path = method_path / "private_artifacts.json"
    path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

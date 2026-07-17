"""Runner separe pour executer une evaluation sans bloquer Streamlit.

La page Streamlit cree un job JSON, lance ce script dans un processus Python
separe, puis lit le meme JSON pour afficher le progres.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_submit import submit_aligned_image_based, submit_homography_based


class LocalUploadedFile:
    """Petit wrapper compatible avec les validateurs de fichiers Streamlit."""

    def __init__(self, path: Path, name: str | None = None) -> None:
        self.path = path
        self.name = name or path.name
        self._data = path.read_bytes()

    def read(self) -> bytes:
        return self._data

    def getbuffer(self):
        return memoryview(self._data)


def now_iso() -> str:
    """Retourne l'heure UTC courante en format stable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_job(job_file: Path) -> dict:
    """Lit l'etat courant du job."""
    return json.loads(job_file.read_text(encoding="utf-8"))


def write_job(job_file: Path, job: dict) -> None:
    """Ecrit l'etat courant du job."""
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")


def update_job(job_file: Path, **updates) -> None:
    """Met a jour quelques champs du job sans perdre le reste."""
    job = read_job(job_file)
    job.update(updates)
    job["updated_at"] = now_iso()
    write_job(job_file, job)


def main() -> int:
    """Point d'entree CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True, type=Path)
    args = parser.parse_args()
    job_file = args.job_file
    job = read_job(job_file)

    def progress_callback(current: int, total: int, label: str) -> None:
        update_job(
            job_file,
            status="running",
            phase="Step 4: Validation and official evaluation running",
            progress_current=int(current),
            progress_total=int(total),
            progress_label=label,
        )

    try:
        update_job(
            job_file,
            status="running",
            phase="Step 4: Validation and official evaluation running",
            started_at=job.get("started_at") or now_iso(),
        )
        metadata = job["metadata"]
        runtime = job.get("runtime", {})
        device = job.get("device", "auto")
        if job["submission_kind"] == "homography":
            H = np.load(job["payload_h_path"], allow_pickle=False)
            method_path, metrics = submit_homography_based(
                metadata,
                H,
                runtime,
                device=device,
                progress_callback=progress_callback,
            )
        else:
            uploaded_zip = LocalUploadedFile(Path(job["payload_zip_path"]), job.get("payload_zip_name"))
            method_path, metrics = submit_aligned_image_based(
                metadata,
                uploaded_zip,
                runtime,
                device=device,
                progress_callback=progress_callback,
            )
        update_job(
            job_file,
            status="completed",
            phase="Step 5: Official Evaluation completed",
            method_path=str(method_path),
            num_rows=int(len(metrics)),
            finished_at=now_iso(),
        )
        return 0
    except Exception as exc:
        update_job(
            job_file,
            status="failed",
            phase="Evaluation failed",
            error=str(exc),
            traceback=traceback.format_exc(),
            finished_at=now_iso(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

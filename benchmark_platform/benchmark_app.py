"""Interface Streamlit de la plateforme VIS-SWIR Benchmark.

Contient les pages de protocole, leaderboard, comparaison de methodes,
soumission de resultats et detail par methode. La logique lourde vit dans les
modules benchmark_eval, benchmark_submit, benchmark_storage et benchmark_plots.
"""

from __future__ import annotations

import sys
import socket
import threading
import re
import json
import subprocess
import uuid
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_compare import build_ranking, load_method_bundle, mean_metric_table, metrics_with_method
from benchmark_config import DATASETS, DIRECTIONS, EVALUATION_PROTOCOL
from benchmark_config import RESULTS_ROOT
from benchmark_generate import generate_comparison_artifacts
from benchmark_io import parse_homography_text, read_homography_file
from benchmark_plots import (
    render_comparison_artifact_dir,
    render_centered_table,
    render_single_method_artifact_dir,
    single_metric_panel,
)
from benchmark_storage import discover_methods, load_benchmark_leaderboard
from benchmark_subsets import (
    build_subset_manifest,
    create_raw_full_dataset_zip_file,
    ensure_official_subset_manifests,
    full_raw_dataset_zip_path,
    official_subset_id,
)


st.set_page_config(page_title="VIS-SWIR Benchmark", layout="wide")

ONLINE_SUBMISSIONS_ENABLED = False
EVALUATION_CONTACT_EMAIL = "vis-swir-benchmark@proton.me"

# Styles globaux de l'app : topbar, cartes de protocole, tableaux et boutons.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        max-width: 1500px;
    }
    div[data-testid="stSidebar"] {
        display: none;
    }
    .benchmark-topbar {
        border-bottom: 1px solid #d7dce2;
        padding: 0.35rem 0 0.9rem 0;
        margin-bottom: 0.65rem;
    }
    .benchmark-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: 0;
        margin: 0;
    }
    .benchmark-subtitle {
        color: #55606d;
        margin-top: 0.2rem;
        font-size: 0.98rem;
    }
    .protocol-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid #d7dce2;
        border-radius: 6px;
        overflow: hidden;
        margin: 0.8rem 0 1.1rem 0;
    }
    .protocol-cell {
        padding: 0.7rem 0.9rem;
        border-right: 1px solid #d7dce2;
        background: #f7f8fa;
    }
    .protocol-cell:last-child {
        border-right: 0;
    }
    .protocol-label {
        color: #697386;
        font-size: 0.74rem;
        text-transform: uppercase;
        font-weight: 700;
    }
    .protocol-value {
        color: #1f2937;
        font-size: 0.98rem;
        font-weight: 700;
        margin-top: 0.15rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.2rem 0 1.15rem 0;
        padding: 0.35rem 0;
    }
    div[role="radiogroup"] label {
        border: 1px solid #cfd6df;
        border-radius: 6px;
        padding: 0.45rem 0.75rem;
        background: #ffffff;
        min-height: 2.25rem;
    }
    div[role="radiogroup"] label p {
        color: #1f2937;
    }
    div[role="radiogroup"] label:has(input:checked) {
        border-color: #2f6f9f;
        background: #e8f2ff;
        color: #12344d;
        box-shadow: inset 0 0 0 1px #2f6f9f;
    }
    div[role="radiogroup"] label:has(input:checked) p {
        color: #12344d;
    }
    div[role="radiogroup"] label p {
        font-weight: 700;
        font-size: 0.9rem;
    }
    .benchmark-workflow {
        max-width: 720px;
        margin: 1rem auto 1.4rem auto;
    }
    .workflow-step {
        border: 1px solid #d7dce2;
        border-radius: 6px;
        background: #f7f8fa;
        padding: 0.72rem 0.9rem;
        text-align: center;
        font-weight: 800;
        color: #1f2937;
    }
    .workflow-step span {
        display: block;
        margin-top: 0.12rem;
        font-weight: 500;
        color: #55606d;
        font-size: 0.88rem;
    }
    .workflow-arrow {
        text-align: center;
        color: #2f6f9f;
        font-size: 1.35rem;
        line-height: 1.55rem;
        font-weight: 800;
    }
    .protocol-callout {
        border-left: 4px solid #2f6f9f;
        background: #eef6ff;
        padding: 0.9rem 1rem;
        border-radius: 4px;
        margin: 1rem auto;
        max-width: 720px;
        font-weight: 800;
        color: #12344d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DISPLAY_COLUMNS = {
    "rank": "Rank",
    "method_name": "Method name",
    "dataset": "Dataset",
    "direction": "Direction",
    "edge_f1_8px": "Edge Alignment 8 px",
    "training_pairs_used": "Training pairs",
    "goa": "GOA",
    "mind_ssd": "MIND-SSD",
    "reprojection_error": "Reprojection Error",
    "runtime_seconds": "Runtime",
    "fps": "FPS",
    "common_frames": "Common frames",
    "submission_type": "Submission",
    "num_evaluated_pairs": "# Pairs",
    "subset_id": "Official Subset",
}
PLOT_METRICS = ["goa", "edge_f1_8px", "mind_ssd", "reprojection_error"]
LEADERBOARD_RANK_METRICS = ["GOA", "Edge Alignment", "MIND-SSD", "Reprojection Error", "Runtime", "FPS"]
COMPARE_RANK_METRICS = ["GOA", "Edge Alignment", "MIND-SSD", "Reprojection Error"]


def render_step(title: str, body: str | None = None) -> None:
    """Affiche un titre d'etape dans le workflow de soumission."""
    st.markdown(f"### {title}")
    if body:
        st.caption(body)


def frame_ids_preview(frame_ids: list[int]) -> str:
    """Resume une liste de frames officielle sous forme compacte."""
    if len(frame_ids) <= 10:
        return ", ".join(str(value) for value in frame_ids)
    return f"{frame_ids[0]}, {frame_ids[1]}, {frame_ids[2]}, ..., {frame_ids[-1]}"


def display_dataset(dataset: str) -> str:
    """Affiche un nom de dataset lisible sans modifier la valeur interne."""
    return str(dataset).replace("_test", "").title()


def pretty_direction(direction: str) -> str:
    """Affiche la direction avec une fleche lisible."""
    return str(direction).replace("_to_", " → ").replace("_", " ")


def display_submission_type(submission_type: str) -> str:
    """Affiche le type de submission avec une casse lisible."""
    return str(submission_type).replace("_", " ").title()


def official_subset_display_name(manifest: dict) -> str:
    """Construit le nom public court du subset officiel."""
    dataset = display_dataset(str(manifest.get("dataset", "")))
    direction = pretty_direction(str(manifest.get("direction", "")))
    return f"{dataset} {direction} official leaderboard set"


def render_subset_card(manifest: dict) -> None:
    """Affiche le subset officiel du leaderboard sous forme de carte lisible."""
    frame_ids = [int(value) for value in manifest.get("official_subset_frame_ids", [])]
    subset_name = official_subset_display_name(manifest)
    subset_id = manifest.get("subset_id", "")
    st.markdown(
        f"""
        <div class="protocol-row">
          <div class="protocol-cell">
            <div class="protocol-label">Dataset</div>
            <div class="protocol-value">{manifest.get("dataset", "")}</div>
          </div>
          <div class="protocol-cell">
            <div class="protocol-label">Direction</div>
            <div class="protocol-value">{manifest.get("direction", "")}</div>
          </div>
          <div class="protocol-cell">
            <div class="protocol-label">Number of official frames</div>
            <div class="protocol-value">{manifest.get("num_subset_pairs", "")}</div>
          </div>
          <div class="protocol-cell">
            <div class="protocol-label">Official Subset</div>
            <div class="protocol-value">{subset_name}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"ID: {subset_id}")
    st.caption(f"Frame IDs: {frame_ids_preview(frame_ids)}")
    st.caption(f"Sampling strategy: {manifest.get('sampling_strategy', '')}")


def file_size_label(path: Path) -> str:
    """Formate la taille d'un fichier pour l'interface."""
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class QuietDownloadHandler(SimpleHTTPRequestHandler):
    """Serveur HTTP local silencieux pour gros ZIPs."""

    def log_message(self, format: str, *args: object) -> None:
        return


def _port_is_open(host: str, port: int) -> bool:
    """Verifie si le serveur local de telechargement est deja lance."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def ensure_download_server(port: int = 8765) -> str:
    """Demarre un petit serveur HTTP local pour les gros fichiers de download."""
    downloads_dir = RESULTS_ROOT / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    if not _port_is_open("127.0.0.1", port):
        handler = partial(QuietDownloadHandler, directory=str(downloads_dir))
        server = ThreadingHTTPServer(("0.0.0.0", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    return f"http://localhost:{port}"


def render_prepared_zip_download(file_path: Path) -> None:
    """Affiche un vrai lien de telechargement sans charger le ZIP dans Streamlit."""
    base_url = ensure_download_server()
    url = f"{base_url}/{file_path.name}"
    st.markdown(
        f"""
        <a href="{url}" target="_blank" download
           style="display:inline-block;padding:0.48rem 0.78rem;border-radius:6px;
                  background:#ff4b4b;color:white;text-decoration:none;font-weight:600;">
           Download Full RAW Unaligned Dataset
        </a>
        <span style="margin-left:0.6rem;color:#697386;">{file_size_label(file_path)}</span>
        """,
        unsafe_allow_html=True,
    )


def now_iso() -> str:
    """Retourne l'heure UTC courante en format stable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float | None) -> str:
    """Formate une duree courte pour l'interface."""
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def read_submission_job() -> dict | None:
    """Lit le job de soumission courant depuis le JSON de session."""
    job_file_raw = st.session_state.get("submission_job_file")
    if not job_file_raw:
        return None
    job_file = Path(job_file_raw)
    if not job_file.exists():
        return None
    try:
        job = json.loads(job_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    job["job_file"] = str(job_file)
    return job


def submission_job_is_running() -> bool:
    """Indique si un processus d'evaluation est encore en cours."""
    job = read_submission_job()
    return bool(job and job.get("status") in {"queued", "running"})


def submission_timing(job: dict) -> tuple[float | None, float | None]:
    """Calcule temps ecoule et ETA approximatif depuis le progres courant."""
    try:
        started_at = datetime.fromisoformat(str(job.get("started_at", "")))
    except ValueError:
        return None, None
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    current = int(job.get("progress_current") or 0)
    total = int(job.get("progress_total") or 0)
    if current <= 0 or total <= 0:
        return elapsed, None
    eta = elapsed * max(total - current, 0) / max(current, 1)
    return elapsed, eta


def safe_method_slug(method_name: str) -> str:
    """Cree un fragment de dossier lisible pour un job."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_name.strip()).strip("_")
    return slug or "method"


def write_submission_job(job: dict) -> Path:
    """Cree le dossier de job et ecrit son fichier JSON."""
    job_root = RESULTS_ROOT / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{safe_method_slug(job['metadata']['method_name'])}_{uuid.uuid4().hex[:8]}"
    job_dir = job_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / "job.json"
    job["job_id"] = job_id
    job["job_dir"] = str(job_dir)
    job["job_file"] = str(job_file)
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job_file


def launch_submission_job(job_file: Path) -> None:
    """Lance le runner d'evaluation dans un processus Python separe."""
    log_path = job_file.parent / "runner.log"
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("benchmark_job_runner.py")),
            "--job-file",
            str(job_file),
        ],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    log_file.close()
    job = json.loads(job_file.read_text(encoding="utf-8"))
    job["pid"] = process.pid
    job["log_path"] = str(log_path)
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")


def start_submission_job(
    submission_kind: str,
    metadata: dict,
    runtime: dict,
    device: str,
    H: np.ndarray | None = None,
    uploaded_zip=None,
) -> Path:
    """Prepare les payloads, ecrit le job JSON et lance le processus runner."""
    job = {
        "status": "queued",
        "phase": "Step 4: Validation queued",
        "submission_kind": submission_kind,
        "metadata": metadata,
        "runtime": runtime,
        "device": device,
        "method_name": metadata.get("method_name", ""),
        "dataset": metadata.get("dataset", ""),
        "direction": metadata.get("direction", ""),
        "started_at": now_iso(),
        "progress_current": 0,
        "progress_total": 0,
        "progress_label": "Waiting for validation",
    }
    job_file = write_submission_job(job)
    job_dir = job_file.parent
    if submission_kind == "homography":
        h_path = job_dir / "H_global.npy"
        np.save(h_path, H)
        job["payload_h_path"] = str(h_path)
    else:
        zip_path = job_dir / "aligned_images.zip"
        zip_path.write_bytes(uploaded_zip.read())
        job["payload_zip_path"] = str(zip_path)
        job["payload_zip_name"] = uploaded_zip.name
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
    launch_submission_job(job_file)
    st.session_state["submission_job_file"] = str(job_file)
    return job_file


def render_submission_job_status() -> bool:
    """Affiche le statut de l'evaluation lancee par processus separe."""
    job = read_submission_job()
    if not job:
        return False
    status = job.get("status")
    if status in {"queued", "running"}:
        st.info(
            f"Evaluation running for {job.get('method_name', 'method')} "
            f"({job.get('dataset', '')}, {job.get('direction', '')})."
        )
        render_step(job.get("phase", "Step 4: Validation running"))
        current = int(job.get("progress_current") or 0)
        total = int(job.get("progress_total") or 0)
        if total > 0:
            st.progress(min(current / total, 1.0))
            st.caption(f"Progress: {current}/{total} - {job.get('progress_label', '')}")
        elapsed, eta = submission_timing(job)
        st.caption(f"Elapsed time: {format_duration(elapsed)}")
        st.caption(f"Estimated time remaining: {format_duration(eta)}")
        st.caption(f"Started at: {job.get('started_at', '')}")
        if st.button("Refresh evaluation status"):
            st.rerun()
        return True
    if status == "completed":
        method_path = job.get("method_path")
        if method_path:
            st.session_state["last_own_submission_path"] = method_path
        st.success(f"Evaluation completed for {job.get('method_name', 'method')}.")
        st.caption(f"Finished at: {job.get('finished_at', '')}")
        return False
    if status == "failed":
        st.error(f"Evaluation failed: {job.get('error', 'Unknown error')}")
        st.caption(f"Finished at: {job.get('finished_at', '')}")
        return False
    return False


def render_last_submission_result() -> None:
    """Reaffiche le dernier resultat de soumission apres un changement de page."""
    render_submission_job_status()
    method_path_raw = st.session_state.get("last_own_submission_path")
    if not method_path_raw:
        return
    method_path = Path(method_path_raw)
    if not method_path.exists():
        return
    try:
        bundle = load_method_bundle(method_path)
    except Exception:
        return
    metrics = bundle.get("metrics", pd.DataFrame())
    st.markdown("### Last evaluation result")
    st.success(f"Last method evaluated and saved in {method_path}")
    render_step("Step 5: Official Evaluation")
    st.caption("Official metrics were computed on the fixed Official Evaluation frame IDs.")
    if not metrics.empty:
        visible_metrics = [
            column
            for column in [
                "frame_id",
                "goa",
                "edge_f1_8px",
                "mind_ssd",
                "reprojection_error",
            ]
            if column in metrics.columns
        ]
        if visible_metrics:
            render_centered_table(metrics[visible_metrics].head(20), max_height=360)


def render_header() -> None:
    """Affiche l'en-tete principal de l'application Streamlit."""
    st.markdown(
        """
        <div class="benchmark-topbar">
          <div class="benchmark-title">VIS-SWIR Benchmark</div>
          <div class="benchmark-subtitle">
            Public benchmark for global homography-based VIS-SWIR image registration.<br>
            All methods are evaluated on the same fixed test set, metrics, and protocol.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_protocol_strip(df: pd.DataFrame) -> None:
    """Affiche le resume global du protocole officiel du benchmark."""
    methods = 0 if df.empty else df["method_path"].nunique()
    protocol_label = EVALUATION_PROTOCOL.replace("_", " ").title()
    frames = "Official leaderboard subset"
    frame_col = "num_evaluated_pairs" if "num_evaluated_pairs" in df else "num_test_frames"
    if not df.empty and frame_col in df:
        valid = pd.to_numeric(df[frame_col], errors="coerce").dropna()
        if not valid.empty:
            min_frames = int(valid.min())
            max_frames = int(valid.max())
            if min_frames == max_frames:
                frames = f"{min_frames} fixed evaluation pairs"
            else:
                frames = f"{min_frames}-{max_frames} evaluated pairs"
    st.markdown(
        f"""
        <div class="protocol-row">
          <div class="protocol-cell">
            <div class="protocol-label">Methods indexed</div>
            <div class="protocol-value">{methods}</div>
          </div>
          <div class="protocol-cell">
            <div class="protocol-label">Protocol</div>
            <div class="protocol-value">{protocol_label}</div>
          </div>
          <div class="protocol-cell">
            <div class="protocol-label">Official subset</div>
            <div class="protocol-value">{frames}</div>
          </div>
          <div class="protocol-cell">
            <div class="protocol-label">Metrics</div>
            <div class="protocol-value">same for all methods</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def official_leaderboard_view(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les anciennes variantes cropped du leaderboard."""
    if df.empty or "method_path" not in df:
        return df
    table = df.copy()
    method_path = table["method_path"].fillna("").astype(str)
    keep = ~method_path.str.contains("_cropped_like_rasmd", case=False, na=False)
    if "evaluation_protocol" in table:
        keep &= table["evaluation_protocol"].fillna("").eq(EVALUATION_PROTOCOL)
    return table.loc[keep].copy()


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare une table lisible pour l'interface avec noms de colonnes et arrondis."""
    table = df.copy()
    numeric_cols = ["goa", "edge_f1_8px", "mind_ssd", "reprojection_error", "runtime_seconds", "fps"]
    for col in numeric_cols:
        if col in table:
            table[col] = pd.to_numeric(table[col], errors="coerce")
            table[col] = table[col].round(4)
    if "training_pairs_used" in table:
        table["training_pairs_used"] = pd.to_numeric(table["training_pairs_used"], errors="coerce").astype("Int64")
    if "dataset" in table:
        table["dataset"] = table["dataset"].map(display_dataset)
    if "direction" in table:
        table["direction"] = table["direction"].map(pretty_direction)
    if "submission_type" in table:
        table["submission_type"] = table["submission_type"].map(display_submission_type)
    if "num_test_frames" in table:
        table["num_test_frames"] = pd.to_numeric(table["num_test_frames"], errors="coerce").astype("Int64")
    if "num_evaluated_pairs" in table:
        evaluated = pd.to_numeric(table["num_evaluated_pairs"], errors="coerce")
        if "num_test_frames" in table:
            evaluated = evaluated.fillna(pd.to_numeric(table["num_test_frames"], errors="coerce"))
        table["num_evaluated_pairs"] = evaluated.astype("Int64")
    if "common_frames" in table:
        table["common_frames"] = pd.to_numeric(table["common_frames"], errors="coerce").astype("Int64")
    rename = {key: value for key, value in DISPLAY_COLUMNS.items() if key in table}
    display_cols = [col for col in DISPLAY_COLUMNS if col in table]
    if "num_test_frames" in table and "num_evaluated_pairs" not in table:
        display_cols.append("num_test_frames")
        rename["num_test_frames"] = "Evaluated frames"
    return table[display_cols].rename(columns=rename)


def format_leaderboard_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare la table du leaderboard sans colonnes internes de soumission."""
    table = format_table(df)
    return table.drop(columns=["Submission", "# Pairs", "Official Subset"], errors="ignore")


def method_info_table(metadata: dict, runtime: dict) -> pd.DataFrame:
    """Construit la table d'information propre a une methode selectionnee."""
    def has_value(value: object) -> bool:
        if isinstance(value, str):
            return bool(value.strip()) and value.strip().lower() not in {"nan", "none"}
        return not pd.isna(value)

    estimation_pairs = metadata.get("estimation_pairs_used", metadata.get("training_pairs_used", ""))
    rows = [
        ("Method Name", metadata.get("method_name", "")),
        ("Dataset", display_dataset(metadata.get("dataset", ""))),
        ("Direction", pretty_direction(metadata.get("direction", ""))),
        ("Number of Estimation Pairs", estimation_pairs),
    ]
    runtime_seconds = runtime.get("runtime_seconds", runtime.get("runtime_total_s", ""))
    fps = runtime.get("fps", runtime.get("fps_global", ""))
    if has_value(runtime_seconds):
        rows.append(("Runtime", runtime_seconds))
    if has_value(fps):
        rows.append(("FPS", fps))
    return pd.DataFrame(rows, columns=["Field", "Value"])


def render_table_no_scroll(df: pd.DataFrame) -> None:
    """Affiche une table HTML complete sans zone de defilement interne."""
    table = df.copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        table[col] = table[col].round(4)
    table = table.astype(object).where(pd.notna(table), "")
    html = table.to_html(index=False, escape=True, classes="benchmark-no-scroll-table", border=0)
    css = """
<style>
table.benchmark-no-scroll-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin: 0.35rem 0 1rem 0;
    border: 1px solid #d7dce2;
    border-radius: 6px;
}
table.benchmark-no-scroll-table th,
table.benchmark-no-scroll-table td {
    text-align: center !important;
    vertical-align: middle !important;
    padding: 0.55rem 0.65rem;
    border: 1px solid #e1e5eb;
    white-space: normal;
    overflow-wrap: anywhere;
}
table.benchmark-no-scroll-table th {
    background: #f7f8fa;
    color: #697386;
    font-weight: 700;
}
table.benchmark-no-scroll-table td {
    color: #111827;
}
</style>
""".strip()
    st.markdown(
        f"{css}\n{html}",
        unsafe_allow_html=True,
    )


def _homography_candidates(method_path: Path) -> tuple[list[Path], bool]:
    """Cherche les fichiers d'homographie NPY, avec repli vers la methode de base."""
    candidates = list(method_path.glob("H*.npy")) + list(method_path.glob("homography*.npy"))
    if candidates:
        return candidates, False
    if method_path.name.endswith("_cropped_like_rasmd"):
        base_path = method_path.with_name(method_path.name.replace("_cropped_like_rasmd", ""))
        base_candidates = list(base_path.glob("H*.npy")) + list(base_path.glob("homography*.npy"))
        if base_candidates:
            return base_candidates, True
    return [], False


def _homography_npz_candidates(method_path: Path) -> tuple[list[Path], bool]:
    """Cherche les fichiers d'homographie NPZ, avec repli vers la methode de base."""
    candidates = list(method_path.glob("homography*.npz"))
    if candidates:
        return candidates, False
    if method_path.name.endswith("_cropped_like_rasmd"):
        base_path = method_path.with_name(method_path.name.replace("_cropped_like_rasmd", ""))
        base_candidates = list(base_path.glob("homography*.npz"))
        if base_candidates:
            return base_candidates, True
    return [], False


def _load_homography_npz(path: Path) -> np.ndarray | None:
    """Charge la matrice H depuis un fichier NPZ de methode Patch MAGSAC++."""
    try:
        data = np.load(path, allow_pickle=False)
        if "H" in data:
            H = data["H"]
        else:
            H = data[data.files[0]]
        if H.shape == (3, 3):
            return H
    except Exception:
        return None
    return None


def homography_latex(H: np.ndarray, direction: str) -> str:
    """Formate une matrice d'homographie en LaTeX pour l'affichage."""
    if H.ndim == 3:
        H = H[0]
    direction_label = "VIS\\rightarrow SWIR"
    if direction == "SWIR_to_VIS":
        direction_label = "SWIR\\rightarrow VIS"
    return rf"""
    H_{{\mathrm{{{direction_label}}}}} =
    \begin{{bmatrix}}
    {H[0,0]:.4f} & {H[0,1]:.4f} & {H[0,2]:.2f} \\
    {H[1,0]:.4f} & {H[1,1]:.4f} & {H[1,2]:.2f} \\
    {H[2,0]:.2e} & {H[2,1]:.2e} & {H[2,2]:.4f}
    \end{{bmatrix}}
    """


def metric_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calcule un resume statistique des metriques principales."""
    rows = []
    for column, label in [
        ("goa", "GOA"),
        ("edge_f1_8px", "Edge Alignment"),
        ("mind_ssd", "MIND-SSD"),
        ("reprojection_error", "Reprojection Error"),
    ]:
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Metric": label,
                "Mean": values.mean(),
                "Median": values.median(),
                "Std": values.std(),
                "Min": values.min(),
                "Max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def cases_table(metrics: pd.DataFrame, metric: str, best: bool, n: int = 3) -> pd.DataFrame:
    """Construit la table des meilleurs ou pires cas pour une metrique donnee."""
    if metric not in metrics:
        return pd.DataFrame()
    table = metrics.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table = table.dropna(subset=[metric])
    if table.empty:
        return pd.DataFrame()
    ascending = metric != "goa"
    if not best:
        ascending = not ascending
    table = table.sort_values(metric, ascending=ascending).head(n)
    return pd.DataFrame(
        {
            "Frame": table.apply(_display_frame_label, axis=1).to_numpy(),
            DISPLAY_COLUMNS.get(metric, metric): pd.to_numeric(table[metric], errors="coerce").to_numpy(),
        }
    )


def _display_frame_label(row: pd.Series) -> str:
    """Retourne un identifiant public compact sans exposer le nom de fichier."""
    if "frame_id" in row and pd.notna(row["frame_id"]):
        try:
            return f"Frame {int(row['frame_id'])}"
        except (TypeError, ValueError):
            pass
    if "pair_id" in row and pd.notna(row["pair_id"]):
        numbers = re.findall(r"\d+", str(row["pair_id"]))
        if numbers:
            return f"Frame {int(numbers[-1])}"
    return "Frame"


def _frame_stem(row: pd.Series) -> str:
    """Recupere l'identifiant de frame utilise pour retrouver les images associees."""
    if "pair_id" in row and pd.notna(row["pair_id"]):
        return str(row["pair_id"])
    if "frame_id" in row and pd.notna(row["frame_id"]):
        try:
            return f"opencv_rgb_image_{int(row['frame_id']):06d}"
        except Exception:
            return str(row["frame_id"])
    return ""


def _frame_stem_variants(stem: str) -> list[str]:
    """Construit les variantes de nom possibles pour un meme frame."""
    if not stem:
        return []
    variants = [stem]
    numeric_tokens = re.findall(r"\d+", stem)
    for token in numeric_tokens:
        frame_id = int(token)
        variants.extend(
            [
                f"frame_{frame_id:06d}",
                f"rgb_image_{frame_id:06d}",
                f"opencv_rgb_image_{frame_id:06d}",
                f"swir_image_{frame_id:06d}",
                f"rasmd_{frame_id:06d}",
            ]
        )
    return list(dict.fromkeys(variants))


def _image_for_stem(folder: Path, stem: str, patterns: list[str]) -> Path | None:
    """Trouve une image correspondant a un identifiant de frame et a des motifs de nom."""
    if not stem or not folder.exists():
        return None
    for candidate_stem in _frame_stem_variants(stem):
        for pattern in patterns:
            candidate = folder / pattern.format(stem=candidate_stem)
            if candidate.exists():
                return candidate
    for token in re.findall(r"\d+", stem):
        padded = f"{int(token):06d}"
        matches = (
            sorted(folder.glob(f"*{padded}*.jpg"))
            + sorted(folder.glob(f"*{padded}*.jpeg"))
            + sorted(folder.glob(f"*{padded}*.png"))
            + sorted(folder.glob(f"*{token}*.jpg"))
            + sorted(folder.glob(f"*{token}*.jpeg"))
            + sorted(folder.glob(f"*{token}*.png"))
        )
        wanted = [
            word
            for word in ["vis_aligned", "swir_aligned", "_vis", "_swir", "visible", "overlay", "concatenate", "_con"]
            if word in " ".join(patterns)
        ]
        for match in matches:
            name = match.name.lower()
            if not wanted or any(key in name for key in wanted):
                return match
    return None


def _case_image_label(default_label: str, image_path: Path) -> str:
    """Deduit le libelle VIS/SWIR/aligne a partir du nom du fichier image."""
    name = image_path.name.lower()
    if "vis_aligned" in name:
        return "VIS aligned"
    if "swir_aligned" in name:
        return "SWIR aligned"
    if "_vis" in name or "visible" in name:
        return "VIS"
    if "_swir" in name:
        return "SWIR"
    return default_label


def case_images(method_path: Path, row: pd.Series) -> list[tuple[str, Path]]:
    """Retourne les images VIS, SWIR et superposition pour un cas visuel."""
    stem = _frame_stem(row)
    split_dir = method_path / "split_images"
    concat_dir = method_path / "concatenate_images"
    candidates = [
        ("VIS", _image_for_stem(split_dir, stem, ["{stem}_vis_aligned.jpg", "{stem}_vis.jpg", "{stem}_visible.jpg"])),
        ("SWIR", _image_for_stem(split_dir, stem, ["{stem}_swir_aligned.jpg", "{stem}_swir.jpg"])),
        (
            "Superposition",
            _image_for_stem(
                concat_dir,
                stem,
                [
                    "{stem}_overlay.jpg",
                    "{stem}_superposition.jpg",
                    "{stem}_concatenate.jpg",
                    "{stem}_con.jpg",
                    "{stem}.jpg",
                ],
            ),
        ),
    ]
    return [(_case_image_label(label, path), path) for label, path in candidates if path is not None]


def _ranked_panel_images(panels_dir: Path, prefix: str, n: int = 3) -> list[Path]:
    """Selectionne les panneaux best/worst deja generes pour la reprojection."""
    selected = []
    for rank in range(1, n + 1):
        matches = sorted(panels_dir.glob(f"{prefix}_{rank:02d}_frame_*.png"))
        matches += sorted(panels_dir.glob(f"{prefix}_{rank:02d}_frame_*.jpg"))
        if matches:
            selected.append(matches[0])
    if len(selected) < n:
        extra = sorted(panels_dir.glob(f"{prefix}_*.png")) + sorted(panels_dir.glob(f"{prefix}_*.jpg"))
        for image_path in extra:
            if image_path not in selected:
                selected.append(image_path)
            if len(selected) >= n:
                break
    return selected[:n]


def render_reprojection_case_panels(method_path: Path, best: bool) -> bool:
    """Affiche les panneaux visuels de reprojection si disponibles."""
    panels_dir = method_path / "results" / "reprojection_aligned" / "reprojection_aligned_panels"
    if not panels_dir.exists():
        return False
    prefix = "best" if best else "worst"
    images = _ranked_panel_images(panels_dir, prefix)
    if not images:
        return False
    for image_path in images:
        st.image(str(image_path), caption=_clean_frame_label(image_path.stem), width="stretch")
    return True


def render_case_gallery(method_path: Path, metrics: pd.DataFrame, metric: str, best: bool, n: int = 3) -> None:
    """Affiche une galerie des meilleurs ou pires cas pour une metrique."""
    if metric not in metrics:
        return
    table = metrics.copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table = table.dropna(subset=[metric])
    if table.empty:
        return
    ascending = metric != "goa"
    if not best:
        ascending = not ascending
    selected = table.sort_values(metric, ascending=ascending).head(n)
    title = "Best cases" if best else "Worst cases"
    st.markdown(f"### {title}")
    render_centered_table(cases_table(metrics, metric, best, n), max_height=260)
    full_cases = 0
    for _, row in selected.iterrows():
        images = case_images(method_path, row)
        if len(images) < 3:
            continue
        full_cases += 1
        frame_label = _display_frame_label(row)
        st.caption(f"{frame_label} — {DISPLAY_COLUMNS.get(metric, metric)} = {float(row[metric]):.4f}")
        cols = st.columns(3)
        for col, (label, image_path) in zip(cols, images):
            col.image(str(image_path), caption=label, width="stretch")
    if full_cases == 0:
        st.info("No complete VIS/SWIR/superposition visual cases are available for these frames.")


def render_method_mind_map(method_path: Path) -> None:
    """Affiche la carte MIND avec son echelle numerique, sans nom de fichier."""
    raw_path = method_path / "results" / "mind_map_mean_raw.npy"
    if raw_path.exists():
        mind_map = np.load(raw_path, allow_pickle=False)
        finite = mind_map[np.isfinite(mind_map)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [2, 98])
            if vmax <= vmin:
                vmax = vmin + 1e-8
            st.markdown("### MIND map")
            fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
            image = ax.imshow(mind_map, cmap="jet", vmin=float(vmin), vmax=float(vmax))
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="MIND-SSD")
            left, center, right = st.columns([1, 2, 1])
            center.pyplot(fig, use_container_width=True)
            plt.close(fig)
            return
    for image_path in [
        method_path / "results" / "mind_map_mean_GLOBAL_SCALE.png",
        method_path / "results" / "mind_map_mean.png",
    ]:
        if image_path.exists():
            st.markdown("### MIND map")
            left, center, right = st.columns([1, 2, 1])
            center.image(str(image_path), width=650)
            center.caption("Numeric MIND scale is unavailable for this legacy image.")
            return


def _official_visual_method_path(method_path: Path) -> Path:
    """Retourne toujours la sortie finale non cropped."""
    if method_path.name.endswith("_cropped_like_rasmd"):
        return method_path.with_name(method_path.name.replace("_cropped_like_rasmd", ""))
    return method_path


def _diagnostic_visual_method_path(method_path: Path) -> Path:
    """Retourne la sortie finale split_images."""
    if method_path.name.endswith("_cropped_like_rasmd"):
        base_path = method_path.with_name(method_path.name.replace("_cropped_like_rasmd", ""))
        if (base_path / "concatenate_images").exists():
            return base_path
    return method_path


def _visual_example_dir(method_path: Path, mode: str) -> tuple[Path | None, str]:
    """Retourne les visualisations des sorties finales non cropped."""
    diagnostic_path = _diagnostic_visual_method_path(method_path)
    return diagnostic_path / "concatenate_images", ""


def _comparison_paths(paths: list[str], mode: str) -> list[str]:
    """Utilise uniquement les sorties finales non cropped."""
    return [str(_diagnostic_visual_method_path(Path(path))) for path in paths]


def _image_stem_key(image_path: Path) -> str:
    """Normalise le nom d'une image pour associer VIS, SWIR et superposition."""
    stem = image_path.stem
    for suffix in [
        "_vis_aligned",
        "_swir_aligned",
        "_visible",
        "_vis",
        "_swir",
        "_overlay",
        "_concatenate",
        "_con",
    ]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _first_matching_image(folder: Path, stem: str, suffixes: list[str]) -> Path | None:
    """Trouve la premiere image VIS/SWIR compatible avec un stem et des suffixes."""
    if not folder.exists():
        return None
    stem_variants = [stem]
    if stem.startswith("rasmd_") and not stem.startswith("rasmd_image_"):
        stem_variants.append(stem.replace("rasmd_", "rasmd_image_", 1))
    for candidate_stem in stem_variants:
        for suffix in suffixes:
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = folder / f"{candidate_stem}{suffix}{ext}"
                if candidate.exists():
                    return candidate
    return None


def _visual_case_triplets(method_path: Path, mode: str, limit: int = 1) -> tuple[list[tuple[str, list[tuple[str, Path]]]], str]:
    """Construit les triplets VIS, SWIR et superposition pour les exemples visuels."""
    image_dir, note = _visual_example_dir(method_path, mode)
    if image_dir is None:
        return [], note
    split_dir = image_dir.parent / "split_images"
    overlay_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    cases = []
    for overlay_path in overlay_images:
        stem = _image_stem_key(overlay_path)
        vis_path = _first_matching_image(split_dir, stem, ["_vis_aligned", "_vis", "_visible"])
        swir_path = _first_matching_image(split_dir, stem, ["_swir_aligned", "_swir"])
        panels = []
        if vis_path is not None:
            panels.append(("VIS aligned" if "aligned" in vis_path.stem else "VIS", vis_path))
        if swir_path is not None:
            panels.append(("SWIR aligned" if "aligned" in swir_path.stem else "SWIR", swir_path))
        panels.append(("Superposition", overlay_path))
        if len(panels) >= 3:
            cases.append((stem, panels[:3]))
        if len(cases) >= limit:
            break
    return cases, note


def _clean_frame_label(value: str) -> str:
    """Transforme un stem technique comme rgb_image_000500 en Frame 500."""
    numbers = re.findall(r"\d+", str(value))
    return f"Frame {int(numbers[-1])}" if numbers else "Frame"


def _clean_direction_label(value: str) -> str:
    """Affiche une direction sans identifiants techniques ni underscores."""
    return str(value).replace("_to_", " → ").replace("_", " ")


def render_compare_visual_examples(paths: list[str], mode: str) -> None:
    """Affiche les exemples visuels des methodes comparees dans le mode choisi."""
    st.markdown("## Evaluated visual examples")

    shown = False
    for path in paths:
        method_path = Path(path)
        bundle = load_method_bundle(method_path)
        metadata = bundle["metadata"]
        method_name = str(metadata.get("method_name", method_path.name)).replace("_", " ")
        method_title = (
            f"{method_name} | {display_dataset(metadata.get('dataset', ''))} | "
            f"{_clean_direction_label(metadata.get('direction', ''))}"
        )
        cases, note = _visual_case_triplets(method_path, mode)
        if not cases:
            shown = True
            st.markdown(f"### {method_title}")
            st.info(note or "No visual example is available for this method.")
            continue
        shown = True
        st.markdown(f"### {method_title}")
        if note:
            st.caption(note)
        for frame_name, panels in cases:
            st.markdown(f"#### {_clean_frame_label(frame_name)}")
            cols = st.columns(3)
            for col, (label, image_path) in zip(cols, panels):
                col.image(str(image_path), caption=label, width="stretch")
    if not shown:
        st.info("No visual examples are available for the selected methods.")


def render_generated_compare_reports(
    official_paths: list[str],
    diagnostic_paths: list[str],
    official_dir: str | Path,
    diagnostic_dir: str | Path,
) -> None:
    """Affiche un seul rapport base sur les split_images finaux."""
    st.markdown("## Generated comparison images and tables")
    render_compare_visual_examples(official_paths, "Final split_images")
    render_comparison_artifact_dir(official_dir)


def page_home() -> None:
    """Affiche la page Protocol avec les regles et le workflow du benchmark."""
    df = load_benchmark_leaderboard()
    df = official_leaderboard_view(df)
    render_protocol_strip(df)
    st.markdown("## Benchmark Workflow")
    st.markdown(
        """
<div class="benchmark-workflow">
  <div class="workflow-step">Download Full RAW Unaligned Dataset<span>available VIS/SWIR pairs for the selected dataset</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">Estimate One Global Homography<span>use any number of available pairs and declare them</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">Submit H or Final Aligned Pair<span>two accepted submission types</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">Submission Validation<span>metadata, frame IDs, image readability, frame identity</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">Official Evaluation Frame IDs<span>fixed 100 frames defined by the benchmark</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">Final aligned outputs<span>evaluate the images produced by each method</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">GOA / Edge Alignment / MIND-SSD / Reprojection Error<span>same official metrics</span></div>
  <div class="workflow-arrow">↓</div>
  <div class="workflow-step">Leaderboard<span>only official subset evaluations enter the ranking</span></div>
</div>
<div class="protocol-callout">Same frame IDs. Final aligned outputs. Same metrics.</div>
        """
        ,
        unsafe_allow_html=True,
    )

    st.markdown("## Benchmark Policy")
    st.write(
        "This benchmark does not evaluate code. It evaluates a submitted result on a fixed official "
        "VIS-SWIR subset. A participant may submit either a global homography or the final aligned/reference "
        "image pairs produced by their method."
    )
    st.write(
        "Both submission types are evaluated directly from their final aligned images with the same metrics: "
        "GOA, Edge Alignment, MIND-SSD, and Reprojection Error. The evaluation server must not be used for parameter "
        "tuning; participants should choose parameters before submission."
    )
    st.write(
        "To enter the official leaderboard, the evaluated result must be produced by one final global "
        "homography for the selected dataset and direction. The method may use any strategy to estimate "
        "that homography, including classical features, deep matching, dense matching, calibration, "
        "geometric or photometric preprocessing."
    )
    st.write(
        "Full-image diagnostic metrics may be saved for debugging, but they are not used for the official "
        "leaderboard. The interface intentionally does not expose the full dataset; it provides rankings, "
        "method metadata, controlled diagnostics, and limited visual examples."
    )
    st.write(
        "For aligned-image submissions, the benchmark does not assume that the final images remain in RAW "
        "coordinates. Methods may calibrate, transform, or resize internally. The ZIP is validated by "
        "checking that every submitted VIS and SWIR image can be matched back to the corresponding official "
        "RAW frame with a robust same-modality feature check."
    )

    st.markdown("## Official Leaderboard vs Custom Evaluation")
    st.write(
        "The user downloads the full RAW unaligned dataset for the selected dataset. They may use any number "
        "of available pairs from that dataset to estimate one single global homography, but they must declare "
        "how many pairs were used and describe the estimation data."
    )
    st.write(
        "The official leaderboard does not score the full dataset. It scores the final alignment only on a "
        "fixed set of 100 frame IDs defined by the benchmark. These frame IDs are the same for all submissions "
        "for the same dataset and direction, and the user cannot choose arbitrary frames for official ranking."
    )
    st.write(
        "Custom Evaluation may be useful for diagnostics on other frames or ranges, but those results must not "
        "enter the official ranking."
    )

    st.markdown("## Final aligned output evaluation")
    st.write(
        "Each method is evaluated from its final aligned VIS/SWIR images."
    )
    st.write(
        "The submitted outputs are evaluated directly."
    )
    st.write(
        "Comparisons use common frame IDs and the same metric implementations for every final output."
    )


def page_leaderboard() -> None:
    """Affiche le leaderboard officiel avec filtres dataset, direction et metrique."""
    df = load_benchmark_leaderboard()
    df = official_leaderboard_view(df)
    render_protocol_strip(df)
    if df.empty:
        st.info("No evaluated methods are available yet.")
        return

    col1, col2, col3 = st.columns([1.2, 1.2, 1.5])
    dataset = col1.multiselect(
        "Dataset",
        DATASETS,
        default=DATASETS,
        key="leaderboard_dataset_filter",
        format_func=display_dataset,
    )
    direction = col2.multiselect(
        "Direction",
        DIRECTIONS,
        default=DIRECTIONS,
        key="leaderboard_direction_filter",
        format_func=pretty_direction,
    )
    metric_label = col3.selectbox(
        "Rank by",
        LEADERBOARD_RANK_METRICS,
        index=LEADERBOARD_RANK_METRICS.index("Reprojection Error"),
        key="leaderboard_rank_metric",
    )

    filtered = df[df["dataset"].isin(dataset) & df["direction"].isin(direction)].copy()
    ranked = build_ranking(filtered, metric_label)
    st.caption(
        "Ranking table: higher GOA, Edge Alignment and FPS are better; "
        "lower MIND-SSD, reprojection error and runtime are better."
    )
    render_centered_table(format_leaderboard_table(ranked), max_height=620)


def page_compare_methods() -> None:
    """Affiche la page de comparaison des methodes et genere les rapports visuels."""
    st.subheader("Compare Methods")
    st.caption(
        "Comparisons use each method's final aligned VIS/SWIR output."
    )
    method_items = discover_methods()
    if not method_items:
        st.info("No methods are available for comparison.")
        return

    col1, col2 = st.columns([1.2, 1.2])
    selected_datasets = col1.multiselect(
        "Dataset",
        DATASETS,
        default=DATASETS,
        key="compare_dataset_filter",
        format_func=display_dataset,
    )
    selected_directions = col2.multiselect(
        "Direction",
        DIRECTIONS,
        default=DIRECTIONS,
        key="compare_direction_filter",
        format_func=pretty_direction,
    )

    filtered_items = [
        item
        for item in method_items
        if item.get("dataset") in selected_datasets
        and item.get("direction") in selected_directions
    ]
    options = {item["label"]: item["method_path"] for item in filtered_items}
    if not options:
        st.info("No methods are available for that dataset and direction combination.")
        return

    selected = st.multiselect("Methods", list(options.keys()), key="compare_methods_select")
    if not selected:
        return

    paths = [options[label] for label in selected]
    official_paths = _comparison_paths(paths, "Final split_images")
    diagnostic_paths = official_paths
    effective_paths = official_paths
    use_diagnostic_metrics = True
    combined = metrics_with_method(effective_paths, diagnostic=use_diagnostic_metrics)
    summary = mean_metric_table(effective_paths, diagnostic=use_diagnostic_metrics)
    if len(paths) > 1 and combined.empty:
        st.warning("No common frame IDs were found between the selected methods.")
        return

    # Versionne la cle pour ne pas reutiliser les rapports produits avant que
    # les comparaisons entre directions partagent les memes frame IDs.
    selection_key = "final-split-v1|" + "|".join(sorted(paths))
    generated_official_dir = (
        st.session_state.get("benchmark_generated_official_comparison_dir")
        if st.session_state.get("benchmark_generated_selection_key") == selection_key
        else None
    )
    generated_diagnostic_dir = (
        st.session_state.get("benchmark_generated_diagnostic_comparison_dir")
        if st.session_state.get("benchmark_generated_selection_key") == selection_key
        else None
    )

    if generated_official_dir and generated_diagnostic_dir:
        if st.button("Regenerate comparison images", type="primary"):
            with st.spinner("Regenerating comparison plots and tables..."):
                official_dir = generate_comparison_artifacts(official_paths, diagnostic=True)
                diagnostic_dir = generate_comparison_artifacts(diagnostic_paths, diagnostic=True)
            st.session_state["benchmark_generated_official_comparison_dir"] = str(official_dir)
            st.session_state["benchmark_generated_diagnostic_comparison_dir"] = str(diagnostic_dir)
            st.session_state["benchmark_generated_selection_key"] = selection_key
            st.success("Official and diagnostic comparison images regenerated.")
            render_generated_compare_reports(official_paths, diagnostic_paths, official_dir, diagnostic_dir)
            return
        render_generated_compare_reports(
            official_paths,
            diagnostic_paths,
            generated_official_dir,
            generated_diagnostic_dir,
        )
        return

    action_label = "Generate method images" if len(paths) == 1 else "Generate comparison images"
    if st.button(action_label, type="primary"):
        with st.spinner("Generating comparison plots and tables..."):
            official_dir = generate_comparison_artifacts(official_paths, diagnostic=True)
            diagnostic_dir = generate_comparison_artifacts(diagnostic_paths, diagnostic=True)
        st.session_state["benchmark_generated_official_comparison_dir"] = str(official_dir)
        st.session_state["benchmark_generated_diagnostic_comparison_dir"] = str(diagnostic_dir)
        st.session_state["benchmark_generated_selection_key"] = selection_key
        st.success("Official and diagnostic comparison images generated.")
        render_generated_compare_reports(official_paths, diagnostic_paths, official_dir, diagnostic_dir)
        return

    rank_metric = None
    if len(paths) > 1:
        rank_metric = st.selectbox(
            "Rank selected methods by",
            COMPARE_RANK_METRICS,
            index=0,
            key="compare_rank_metric",
        )

    if len(paths) == 1:
        bundle = load_method_bundle(effective_paths[0], diagnostic=use_diagnostic_metrics)
        metadata = bundle["metadata"]
        runtime = bundle["runtime"]
        st.subheader(metadata["method_name"])
        render_centered_table(method_info_table(metadata, runtime), max_height=320)

    st.markdown("### Mean metric table")
    st.caption(
        "Comparison based directly on final aligned VIS/SWIR metrics. "
        "When multiple methods are selected, only common frame IDs are compared."
    )
    render_centered_table(format_table(summary))
    if len(paths) > 1 and rank_metric is not None:
        st.markdown(f"### Ranking by {rank_metric}")
        render_centered_table(format_table(build_ranking(summary, rank_metric)))


def page_submit_results() -> None:
    """Affiche le formulaire de soumission et lance l'evaluation d'une homographie."""
    st.subheader("Submit Results")
    if not ONLINE_SUBMISSIONS_ENABLED:
        st.warning(
            "Online evaluation is temporarily unavailable because the current hosting environment "
            "does not provide the GPU resources required by the evaluation pipeline."
        )
        st.markdown(
            f"""
            ### Submit by email

            Until online evaluation is restored, please send your evaluation request to
            **[{EVALUATION_CONTACT_EMAIL}](mailto:{EVALUATION_CONTACT_EMAIL}?subject=VIS-SWIR%20Benchmark%20Submission)**.

            Please include:

            - method name, authors and institution;
            - dataset and alignment direction (`VIS → SWIR` or `SWIR → VIS`);
            - number of image pairs used to estimate the homography;
            - the global 3×3 homography matrix, or a download link to the aligned-images ZIP;
            - runtime, evaluation device and a short description of the method.

            The benchmark team will run the official evaluation and contact you when the results are ready.
            For large files, please send a download link instead of attaching the ZIP directly to the email.
            """
        )
        st.info("Leaderboard browsing, method comparison and method details remain available.")
        return

    render_last_submission_result()
    if submission_job_is_running():
        return

    render_step(
        "Step 1: Download Full RAW Unaligned Dataset",
        (
            "Download the full RAW unaligned VIS/SWIR dataset. You may use any number of available pairs "
            "from this dataset to estimate one single global homography for the selected dataset and direction. "
            "You must declare how many pairs were used for the estimation."
        ),
    )
    ensure_official_subset_manifests()
    subset_cols = st.columns([1, 1])
    dataset = subset_cols[0].selectbox("Dataset", DATASETS, key="submit_dataset", format_func=display_dataset)
    direction = subset_cols[1].selectbox("Direction", DIRECTIONS, key="submit_direction", format_func=pretty_direction)
    subset_id = official_subset_id(dataset, direction)
    subset_manifest = build_subset_manifest(dataset, subset_id, direction=direction)
    st.info(
        "**Official evaluation**\n\n"
        "The final score is computed only on 100 fixed frame IDs.\n\n"
        "These frame IDs are defined by the benchmark and are identical for all submissions."
    )
    st.caption(
        f"Official scoring frame IDs for {display_dataset(dataset)}: "
        f"{frame_ids_preview(subset_manifest['official_subset_frame_ids'])}"
    )
    st.caption(
        "Direction does not change the downloaded content: both VISIBLE and SWIR are always included. "
        "Direction only defines how the submission will be evaluated."
    )
    full_zip_path = full_raw_dataset_zip_path(dataset)
    if full_zip_path.exists():
        render_prepared_zip_download(full_zip_path)
    else:
        if st.button("Prepare Full Dataset ZIP", type="primary"):
            with st.spinner("Preparing full RAW unaligned dataset ZIP..."):
                full_zip_path = create_raw_full_dataset_zip_file(dataset)
            st.success(f"ZIP prepared: {full_zip_path.name} ({file_size_label(full_zip_path)}).")
            st.rerun()

    render_step(
        "Step 2: Prepare Your Method Output",
    )
    st.write("Estimate one single global homography H for the selected dataset and direction.")
    st.write(
        "You may use any number of RAW VIS/SWIR image pairs to estimate H. However, the same H must be "
        "used as the final alignment transformation for all official evaluation frames."
    )
    st.write(
        "You can submit either the 3×3 homography matrix H directly, or a ZIP containing the official "
        "100 frame pairs already aligned using that same H."
    )
    st.caption(
        "Leaderboard scores are computed only on the fixed Official Evaluation frame IDs. "
        "Results on other frames are diagnostic only and are not ranked."
    )

    render_step("Step 3: Submit Results")
    submission_type = st.selectbox(
        "Submission type *",
        ["Homography-based submission", "Aligned-image submission"],
        key="submit_submission_type",
    )
    if submission_type == "Homography-based submission":
        st.caption(
            "Homography-based submission: upload one global homography matrix H. The benchmark will apply "
            "this H to the official RAW VIS/SWIR pairs, remove only the geometrically invalid black borders "
            "with one common crop, save the cropped aligned images and their 50/50 overlays, and compute "
            "the official metrics. It does not remove the sky or the vehicle."
        )
    else:
        st.caption(
            "Aligned-image submission: upload the final image pairs generated by your method. The benchmark "
            "does not apply H again. It validates the submitted frames against the official RAW subset, "
            "evaluates the submitted final aligned VIS/SWIR images directly."
        )
        st.warning(
            f"Important for aligned-image submission: your ZIP must contain exactly the official 100 frame IDs "
            f"for {display_dataset(dataset)}: {frame_ids_preview(subset_manifest['official_subset_frame_ids'])}."
        )
    h_mode = None
    if submission_type == "Homography-based submission":
        h_mode = st.radio(
            "Global Homography H *",
            ["Manual Input", "File Upload"],
            horizontal=True,
            key="submit_h_mode",
        )
    with st.form("submission_form"):
        st.markdown("### Method information")
        method_name = st.text_input("Method name *")
        authors = st.text_input("Authors")
        institution = st.text_input("Institution")
        paper_url = st.text_input("Paper URL")
        contact_email = st.text_input("Contact email")
        description = st.text_area("Method description")

        st.markdown("### Estimation information")
        estimation_pairs_used = st.number_input(
            "Number of estimation pairs *",
            min_value=0,
            step=1,
            value=0,
            help="Number of RAW VIS/SWIR pairs that contributed to the final global homography estimation.",
        )
        estimation_data_description = st.text_area(
            "Estimation data description",
            placeholder="Briefly describe the estimation data: training split, full dataset, external data, preprocessing, etc.",
        )
        runtime_seconds_text = st.text_input("Runtime seconds (optional)")
        fps_text = st.text_input("FPS (optional)")
        device = st.selectbox("Evaluation device", ["auto", "cpu", "cuda"], index=0)

        st.markdown("### Submission")
        H_text = None
        uploaded_h = None
        uploaded_zip = None
        single_global_homography_confirmed = False
        if submission_type == "Homography-based submission" and h_mode == "Manual Input":
            st.caption(
                "This submitted matrix is treated as the single global homography for the selected dataset and direction."
            )
            H_text = st.text_area(
                "3×3 matrix *",
                value="",
                placeholder="Paste a 3x3 homography matrix, for example:\n1 0 0\n0 1 0\n0 0 1",
                height=120,
            )
        elif submission_type == "Homography-based submission":
            st.caption(
                "This submitted matrix is treated as the single global homography for the selected dataset and direction."
            )
            uploaded_h = st.file_uploader("H_global.txt or H_global.npy *", type=["txt", "npy"])
        else:
            st.markdown(
                """
                Expected ZIP format:

                For **VIS → SWIR**:
                ```text
                frame_xxxxxx_vis_aligned.jpg
                frame_xxxxxx_swir.jpg
                ```

                For **SWIR → VIS**:
                ```text
                frame_xxxxxx_swir_aligned.jpg
                frame_xxxxxx_vis.jpg
                ```

                The benchmark validates frame identity against the official RAW subset before evaluation.
                """
            )
            single_global_homography_confirmed = st.checkbox(
                "I confirm that the submitted aligned images were generated using one single global homography for this dataset and direction.",
                value=False,
            )
            uploaded_zip = st.file_uploader(
                "ZIP file *",
                type=["zip"],
                help=(
                    "Max 200 MB, max 100 final VIS/SWIR pairs. VIS -> SWIR: frame_xxxxxx_vis_aligned + "
                    "frame_xxxxxx_swir. SWIR -> VIS: frame_xxxxxx_swir_aligned + frame_xxxxxx_vis. "
                    "Common extensions such as .jpg, .jpeg and .png are accepted."
                ),
            )
        submitted = st.form_submit_button("Submit and Evaluate")

    if submitted:
        try:
            render_step("Step 4: Validation")
            runtime_seconds = float(runtime_seconds_text) if runtime_seconds_text.strip() else None
            fps = float(fps_text) if fps_text.strip() else None
            metadata = {
                "method_name": method_name,
                "authors": authors,
                "institution": institution,
                "paper_url": paper_url,
                "contact_email": contact_email,
                "submission_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": dataset,
                "direction": direction,
                "subset_id": subset_id,
                "training_pairs_used": estimation_pairs_used,
                "description": description,
                "single_global_homography_confirmed": (
                    True
                    if submission_type == "Homography-based submission"
                    else single_global_homography_confirmed
                ),
                "estimation_pairs_used": estimation_pairs_used,
                "estimation_data_description": estimation_data_description,
            }
            runtime = {
                "runtime_seconds": runtime_seconds,
                "fps": fps,
            }
            if submission_type == "Homography-based submission":
                if h_mode == "Manual Input" and not (H_text or "").strip():
                    raise ValueError("Global homography H is required. Paste a 3×3 matrix or use File Upload.")
                H = parse_homography_text(H_text) if h_mode == "Manual Input" else read_homography_file(uploaded_h)
                start_submission_job("homography", metadata, runtime, device, H=H)
            else:
                if uploaded_zip is None:
                    raise ValueError("Upload aligned_images.zip before submitting.")
                start_submission_job("aligned_images", metadata, runtime, device, uploaded_zip=uploaded_zip)
            st.success("Evaluation started. You can switch pages and return here to refresh progress.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def page_method_details() -> None:
    """Affiche les informations, metriques et cas visuels d'une methode."""
    st.subheader("Method Details")
    method_items = discover_methods()
    if not method_items:
        st.info("No methods are available.")
        return

    col1, col2, col3 = st.columns([1, 1, 1.5])
    dataset = col1.selectbox("Dataset", DATASETS, index=0, key="details_dataset", format_func=display_dataset)
    direction = col2.selectbox("Direction", DIRECTIONS, index=0, key="details_direction", format_func=pretty_direction)

    filtered = [
        item
        for item in method_items
        if item.get("dataset") == dataset
        and item.get("direction") == direction
    ]
    if not filtered:
        st.info("No methods match those filters.")
        return

    options = {}
    for item in sorted(filtered, key=lambda value: value.get("method_name", "")):
        label = item.get("method_name", Path(item["method_path"]).name)
        if label in options:
            label = f"{label} ({item.get('source', 'method')})"
        options[label] = item["method_path"]

    selected = col3.selectbox("Method", list(options.keys()), key="details_method")
    bundle = load_method_bundle(options[selected])
    method_path = bundle["path"]
    metadata = bundle["metadata"]
    runtime = bundle["runtime"]
    metrics = bundle["metrics"]
    h_candidates, h_from_base = _homography_candidates(method_path)
    h_npz_candidates, h_npz_from_base = _homography_npz_candidates(method_path)

    st.markdown("### Method information")
    render_table_no_scroll(method_info_table(metadata, runtime))
    if h_candidates:
        H = np.load(h_candidates[0], allow_pickle=False)
        st.write("Homography matrix")
        st.latex(homography_latex(H, metadata.get("direction", "")))
    elif h_npz_candidates:
        H = _load_homography_npz(h_npz_candidates[0])
        if H is not None:
            st.write("Homography matrix")
            st.latex(homography_latex(H, metadata.get("direction", "")))
        else:
            st.info("The homography matrix is stored but could not be displayed.")
    else:
        st.info("No homography file was found for this method.")

    if not metrics.empty:
        st.markdown("### Metric summary")
        render_centered_table(metric_summary_table(metrics), max_height=320)

        st.markdown("## Quality")
        for metric in ["goa", "edge_f1_8px", "mind_ssd"]:
            if metric in metrics:
                single_metric_panel(metrics, metric)
        render_method_mind_map(method_path)

        st.markdown("## Reprojection")
        if "reprojection_error" in metrics:
            single_metric_panel(metrics, "reprojection_error", title="Reprojection")
        else:
            st.info("No reprojection metric was found for this method.")

        available_metrics = [metric for metric in PLOT_METRICS if metric in metrics]
        if available_metrics:
            st.markdown("## Best and Worst Visual Cases")
            metric = st.selectbox(
                "Select Metric",
                available_metrics,
                format_func=lambda value: DISPLAY_COLUMNS.get(value, value),
                key="details_case_metric",
            )
            render_case_gallery(method_path, metrics, metric, best=True)
            render_case_gallery(method_path, metrics, metric, best=False)


PAGES = {
    "Leaderboard": page_leaderboard,
    "Compare Methods": page_compare_methods,
    "Submit Results (Paused)": page_submit_results,
    "Method Details": page_method_details,
    "Protocol": page_home,
}

render_header()
choice = st.radio(
    "Navigation",
    list(PAGES.keys()),
    horizontal=True,
    label_visibility="visible",
    key="benchmark_navigation",
)

with st.container():
    PAGES[choice]()

import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy.ndimage import gaussian_filter1d
import cv2
import tempfile
import base64
import numpy as np
import textwrap
import streamlit.components.v1 as components
import subprocess
import sys
import os

st.set_page_config(
    page_title="VIS-SWIR Benchmark Dashboard",
    layout="wide"
)

BASE_DIR = Path(os.environ.get("RECALLAGE_BASE_DIR", Path(__file__).resolve().parent))
DATA_DIR = BASE_DIR / "datas"
PDF_DIR = BASE_DIR / "pdfs"
ASSET_DIR = BASE_DIR / "assets"
UCA_LOGO_PATH = ASSET_DIR / "uca.png"
INP_LOGO_PATH = ASSET_DIR / "inp.png"
CNRS_LOGO_PATH = ASSET_DIR / "cnrs_logo.png"
IP_LOGO_PATH = ASSET_DIR / "ip.png"
ITPS_LOGO_PATH = ASSET_DIR / "itps_logo.png"
ISITE_LOGO_PATH = ASSET_DIR / "logo_I-SITE_2023_Off 2.png"


DATASETS = {
    "Cloudy": "cloudy_test",
    "Sunny": "sunny_test",
    "Snowy": "snowy_test",
    "Rainy": "rainy_test",
}


METHODS = {
    "Patch MAGSAC++ VIS → SWIR": "patch_RANSAC_HOMOGRAPHY_calibration_VIS_to_SWIR_H_from_objectif_12mm_4_boucle",
    "Patch MAGSAC++ SWIR → VIS": "patch_RANSAC_HOMOGRAPHY_calibration_SWIR_to_VIS_H_from_objectif_12mm_4_boucle",
    "OpenCV VIS → SWIR": "opencv_alignment_VIS_TO_SWIR_train150_step5_test500_to_1499",
    "OpenCV SWIR → VIS": "opencv_alignment_SWIR_TO_VIS_train150_step5_test500_to_1499",
    "RASMD Aligned": "RASMD_aligned_baseline",
}


DIRECTIONS = {
    "VIS → SWIR": "VIS_TO_SWIR",
    "SWIR → VIS": "SWIR_TO_VIS",
}


VIEW_ONLY = os.environ.get("STREAMLIT_VIEW_ONLY", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


PATCH_METHOD_FOLDERS = {
    "VIS_TO_SWIR": "patch_RANSAC_HOMOGRAPHY_calibration_VIS_to_SWIR_H_from_objectif_12mm_4_boucle",
    "SWIR_TO_VIS": "patch_RANSAC_HOMOGRAPHY_calibration_SWIR_to_VIS_H_from_objectif_12mm_4_boucle",
}


RASMD_METHOD_FOLDER = "RASMD_aligned_baseline"


@st.cache_data(show_spinner=False)
def read_csv_cached(path: Path):
    """Read a CSV file once and reuse it across Streamlit reruns."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_text_cached(path: Path):
    """Read a text file once and reuse it across Streamlit reruns."""
    return path.read_text()


@st.cache_data(show_spinner=False)
def image_base64(path: Path):
    """Encode an image file as base64 for inline HTML display."""
    return base64.b64encode(path.read_bytes()).decode()


@st.cache_data(show_spinner=False)
def file_base64(path: Path):
    """Encode any file as base64 for inline HTML display."""
    return base64.b64encode(path.read_bytes()).decode()


@st.cache_data(show_spinner=False)
def load_homography(path: Path):
    """Load a homography matrix from a .npz or .npy file."""
    if path.suffix.lower() == ".npz":
        data = np.load(path)

        if "H" in data.files:
            return data["H"]

        if len(data.files) > 0:
            return data[data.files[0]]

        return None

    if path.suffix.lower() == ".npy":
        return np.load(path)

    return None


def python_executable():
    """Return the project virtualenv Python when available."""
    venv_python = BASE_DIR / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def command_text(command):
    """Format a command list for display."""
    return " ".join(str(part) for part in command)


def run_dashboard_command(command, cwd: Path, title: str):
    """Run one command from the dashboard and stream its output."""
    if VIEW_ONLY:
        st.error("Viewer mode is enabled, so commands cannot be executed from this dashboard.")
        return False

    st.markdown(f"### {title}")
    st.code(command_text(command), language="bash")

    status = st.empty()
    output_box = st.empty()
    output_lines = []

    status.info(f"Running: {title}")

    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            output_lines.append(line.rstrip())
            output_box.code("\n".join(output_lines[-220:]), language="text")

    return_code = process.wait()

    if return_code == 0:
        status.success("Finished successfully.")
        st.cache_data.clear()
        st.success("Finished successfully.")
    else:
        status.error(f"Command failed with exit code {return_code}.")
        st.error(f"Command failed with exit code {return_code}.")

    return return_code == 0


def set_patch_direction(direction: str):
    """Update Config.ALIGN_DIRECTION before running the Patch pipeline."""
    config_path = BASE_DIR / "projet" / "config.py"
    text = config_path.read_text()

    updated, replacements = re.subn(
        r'ALIGN_DIRECTION\s*=\s*"[^"]+"',
        f'ALIGN_DIRECTION = "{direction}"',
        text,
        count=1,
    )

    if replacements == 0:
        raise RuntimeError("Could not find ALIGN_DIRECTION in projet/config.py")

    if updated != text:
        config_path.write_text(updated)


def opencv_method_folder(direction, train_count, train_step, test_start, test_end):
    """Build the OpenCV output folder name from UI parameters."""
    return (
        f"opencv_alignment_{direction}"
        f"_train{train_count}"
        f"_step{train_step}"
        f"_test{test_start}_to_{test_end}"
    )


def cropped_method_folder(method_folder):
    """Return the folder name used for RASMD-cropped method outputs."""
    return f"{method_folder}_cropped_like_rasmd"


def split_images_path(dataset_name, method_folder):
    """Return the split_images path for one method output."""
    return (
        DATA_DIR
        / dataset_name
        / "Sortie_Extraite"
        / "paired_images"
        / method_folder
        / "split_images"
    )


def quality_relative_split_path(dataset_name, method_folder):
    """Return the split_images path relative to the projet folder."""
    return (
        Path(dataset_name)
        / "Sortie_Extraite"
        / "paired_images"
        / method_folder
        / "split_images"
    )


def paired_images_dir(dataset_name):
    """Return the paired_images directory for one dataset."""
    return DATA_DIR / dataset_name / "Sortie_Extraite" / "paired_images"


def list_comparison_method_options(dataset_name):
    """List method folders that can be passed to compare_methods."""
    paired_dir = paired_images_dir(dataset_name)

    if not paired_dir.exists():
        return {}

    ignored = {"VISIBLE", "SWIR"}
    options = {}

    for path in sorted(paired_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue

        if path.name.upper() in ignored or path.name.lower().startswith("benchmark"):
            continue

        if (path / "results").exists():
            options[path.name] = path.name

    return options


def resolve_method_folder(dataset_name, method_folder):
    """Return the existing folder for a method, with OpenCV train/test fallback."""
    paired_dir = paired_images_dir(dataset_name)
    exact_dir = paired_dir / method_folder

    if exact_dir.exists():
        return method_folder

    cropped_suffix = "_cropped_like_rasmd"
    is_cropped = method_folder.endswith(cropped_suffix)
    base_folder = method_folder[:-len(cropped_suffix)] if is_cropped else method_folder

    if not base_folder.startswith("opencv_alignment_"):
        return method_folder

    match = re.match(r"(opencv_alignment_(?:VIS_TO_SWIR|SWIR_TO_VIS)_train\d+_step\d+)_test\d+_to_\d+$", base_folder)

    if not match:
        return method_folder

    prefix = match.group(1)
    suffix = cropped_suffix if is_cropped else ""
    candidates = sorted(
        [
            p.name
            for p in paired_dir.glob(f"{prefix}_test*_to_*{suffix}")
            if p.is_dir() and p.name.endswith(cropped_suffix) == is_cropped
        ]
    )

    if len(candidates) == 0:
        return method_folder

    candidates_with_results = [
        name
        for name in candidates
        if (paired_dir / name / "results").exists()
    ]

    return (candidates_with_results or candidates)[0]


def show_comparison_alignment_frame(dataset_name, selected_methods):
    """Display frame-by-frame VIS/SWIR/concatenate images for selected methods."""
    st.markdown("## Visual comparison by frame")

    comparison_methods = {}

    for method_label in selected_methods:
        method_folder = METHODS[method_label]

        if "RASMD" in method_label:
            folder = method_folder
        else:
            folder = method_folder + "_cropped_like_rasmd"

        method_dir = get_method_dir(dataset_name, folder)

        if method_dir.exists():
            comparison_methods[method_label] = method_dir

    if not comparison_methods:
        st.info("No methods are available for visual comparison.")
        return

    method_images = {}
    common_frame_ids = None

    for method_label, method_dir in comparison_methods.items():
        direction = infer_alignment_direction(method_label, method_dir)
        vis_images, swir_images = find_split_images(method_dir, direction)
        concat_images = find_concatenate_images(method_dir)

        vis_by_frame = images_by_frame_id(vis_images)
        swir_by_frame = images_by_frame_id(swir_images)
        concat_by_frame = images_by_frame_id(concat_images)
        frame_ids = sorted(set(vis_by_frame) & set(swir_by_frame) & set(concat_by_frame))

        if len(frame_ids) == 0:
            continue

        method_images[method_label] = {
            "dir": method_dir,
            "vis": vis_by_frame,
            "swir": swir_by_frame,
            "concat": concat_by_frame,
            "direction": direction,
            "frame_ids": frame_ids,
        }

        method_frame_ids = set(frame_ids)
        common_frame_ids = (
            method_frame_ids
            if common_frame_ids is None
            else common_frame_ids & method_frame_ids
        )

    if not method_images or not common_frame_ids:
        st.info("No split/concatenate images were found for comparison.")
        return

    frame_ids = sorted(common_frame_ids)
    frame_id = st.select_slider(
        "Comparison frame",
        options=frame_ids,
        value=frame_ids[0],
        key="comparison_visual_frame"
    )
    st.caption(f"Showing frame {frame_id:06d} across all selected methods.")

    for method_label, data in method_images.items():
        st.markdown(f"### {method_label}")

        vis_title, swir_title = split_image_titles(data["direction"])

        cols = st.columns(3)

        with cols[0]:
            st.markdown(f"**{vis_title}**")
            st.image(
                str(data["vis"][frame_id]),
                width="stretch"
            )

        with cols[1]:
            st.markdown(f"**{swir_title}**")
            st.image(
                str(data["swir"][frame_id]),
                width="stretch"
            )

        with cols[2]:
            st.markdown("**Concatenate**")
            st.image(
                str(data["concat"][frame_id]),
                width="stretch"
            )

def show_metric_table_from_summary(comparison_dir: Path, metric_name: str):
    """Load and display summary statistics for one metric from the comparison folder."""
    summary_path = comparison_dir / "summary_statistics_methods.csv"

    if not summary_path.exists():
        st.info("summary_statistics_methods.csv was not found.")
        return

    df = read_csv_cached(summary_path)

    metric_col = None
    for col in df.columns:
        if col.lower() in ["metric", "métrica", "metrica"]:
            metric_col = col
            break

    if metric_col is not None:
        df = df[df[metric_col].astype(str).str.lower().str.contains(metric_name.lower())]

    if "mean_pm_std" not in df.columns and {"mean", "std"}.issubset(df.columns):
        df["mean_pm_std"] = df.apply(
            lambda row: (
                f"{row['mean']:.4f} ± {row['std']:.4f}"
                if pd.notna(row["mean"]) and pd.notna(row["std"])
                else ""
            ),
            axis=1,
        )

    desired_cols = [
        "method",
        "méthode",
        "methode",
        "mean_pm_std",
        "mean",
        "median",
        "std",
        "min",
        "max",
    ]
    wanted_cols = [
        col
        for desired in desired_cols
        for col in df.columns
        if col.lower() == desired
    ]

    df = df[wanted_cols]

    # Sort by the best mean value
    if "mean" in df.columns:
        if metric_name.lower() == "goa":
            df = df.sort_values("mean", ascending=False)
        else:
            df = df.sort_values("mean", ascending=True)

    df = df.rename(columns={
        "method": "Method",
        "mean_pm_std": "Mean ± Std",
        "mean": "Mean",
        "median": "Median",
        "std": "Std",
        "min": "Min",
        "max": "Max",
    })

    show_stats_table_html(df)

def show_stats_table_html(df):
    """Render a pandas DataFrame as a centered HTML table."""
    html = """
    <table style="
        width:100%;
        border-collapse:collapse;
        text-align:center;
        font-size:16px;
    ">
        <tr>
    """

    for col in df.columns:
        html += f'<th style="text-align:center;"><b>{col}</b></th>'

    html += "</tr>"

    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            value = row[col]

            if isinstance(value, float):
                value = f"{value:.4f}"

            html += f'<td style="text-align:center;">{value}</td>'

        html += "</tr>"

    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)

def show_existing_image(path: Path, width="stretch"):
    """Display an image if it exists, otherwise show an informational message."""
    if path.exists():
        st.image(str(path), width=width)
    else:
        st.info(f"{path.name} was not found.")


def show_existing_table(path: Path, table_type=None):
    """Display an existing CSV table with optional runtime or ranking formatting."""
    if not path.exists():
        st.info(f"{path.name} was not found.")
        return

    df = read_csv_cached(path)

    if table_type == "runtime":
        value_columns = [
            column
            for column in ["runtime_total_s", "training_pairs_used", "hardware"]
            if column in df.columns
        ]
        if value_columns:
            has_information = pd.Series(False, index=df.index)
            for column in value_columns:
                values = df[column]
                normalized = values.astype(str).str.strip().str.lower()
                has_information |= values.notna() & ~normalized.isin(
                    {"", "-", "nan", "none", "nat", "<na>"}
                )
            df = df.loc[has_information].copy()

        # Remove dataset
        df = df.drop(
            columns=["dataset"],
            errors="ignore"
        )

        # Display names
        df = df.rename(columns={
            "method": "Method",
            "runtime_total_s": "Total time [s]",
            "runtime_hms": "Time",
            "fps_global": "FPS global"
        })

    df = df.astype(object)
    for column in df.columns:
        df.loc[df[column].isna(), column] = "-"
        normalized = df[column].astype(str).str.strip().str.lower()
        df.loc[
            normalized.isin({"", "nan", "none", "nat", "<na>"}),
            column,
        ] = "-"

    if table_type == "ranking":
        df = df.drop(columns=["direction", "p95", "diff_vs_best_mean"], errors="ignore")

        if "mean_pm_std" not in df.columns and {"mean", "std"}.issubset(df.columns):
            df["mean_pm_std"] = df.apply(
                lambda row: (
                    f"{row['mean']:.4f} ± {row['std']:.4f}"
                    if pd.notna(row["mean"]) and pd.notna(row["std"])
                    else ""
                ),
                axis=1,
            )

        if "percent_vs_best_mean" in df.columns:
            df = df.rename(columns={
                "percent_vs_best_mean": "Difference vs best [%]"
            })

        desired_cols = [
            "metric",
            "rank",
            "method",
            "mean_pm_std",
            "mean",
            "median",
            "std",
            "Difference vs best [%]",
        ]
        df = df[[col for col in desired_cols if col in df.columns]]

        df = df.rename(columns={
            "metric": "Metric",
            "rank": "Rank",
            "method": "Method",
            "mean_pm_std": "Mean ± Std",
            "mean": "Mean",
            "median": "Median",
            "std": "Std"
        })

    show_stats_table_html(df)


def method_display_name(method_folder: str):
    """Return a compact method name for season-level comparison tables."""
    cropped_suffix = "_cropped_like_rasmd"
    is_cropped = method_folder.endswith(cropped_suffix)
    base_folder = method_folder[:-len(cropped_suffix)] if is_cropped else method_folder

    for label, configured_folder in METHODS.items():
        resolved_base = configured_folder
        if base_folder == resolved_base:
            return f"{label} cropped" if is_cropped else label

        if (
            base_folder.startswith("opencv_alignment_")
            and resolved_base.startswith("opencv_alignment_")
        ):
            base_match = re.match(
                r"(opencv_alignment_(?:VIS_TO_SWIR|SWIR_TO_VIS)_train\d+_step\d+)_test\d+_to_\d+$",
                base_folder,
            )
            configured_match = re.match(
                r"(opencv_alignment_(?:VIS_TO_SWIR|SWIR_TO_VIS)_train\d+_step\d+)_test\d+_to_\d+$",
                resolved_base,
            )
            if base_match and configured_match and base_match.group(1) == configured_match.group(1):
                return f"{label} cropped" if is_cropped else label

    return method_folder.replace("_", " ")


def first_row_value(df, columns):
    """Return the first available value from a one-row summary CSV."""
    col = metric_column(df, columns)
    if col is None or df.empty:
        return np.nan
    return df.iloc[0][col]


@st.cache_data(show_spinner=False)
def load_season_metrics_table():
    """Collect one summary row per station and method from existing result CSVs."""
    rows = []

    for season_label, dataset_name in DATASETS.items():
        paired_dir = paired_images_dir(dataset_name)
        if not paired_dir.exists():
            continue

        for method_dir in sorted(paired_dir.iterdir(), key=lambda p: p.name.lower()):
            if not method_dir.is_dir():
                continue
            if method_dir.name.upper() in {"VISIBLE", "SWIR"}:
                continue
            if method_dir.name.lower().startswith("benchmark"):
                continue

            results_dir = method_dir / "results"
            quality_path = results_dir / "quality_summary.csv"
            reprojection_path = (
                results_dir
                / "reprojection_aligned"
                / "reprojection_aligned_summary.csv"
            )

            if not quality_path.exists() and not reprojection_path.exists():
                continue

            row = {
                "Station": season_label,
                "Dataset": dataset_name,
                "Method": method_display_name(method_dir.name),
                "Method folder": method_dir.name,
            }

            if quality_path.exists():
                quality_df = pd.read_csv(quality_path)
                row.update({
                    "Frames": first_row_value(quality_df, ["num_frames", "num_images"]),
                    "GOA mean": first_row_value(quality_df, ["mean_goa"]),
                    "Edge Alignment mean": first_row_value(quality_df, ["mean_edge_f1_8px"]),
                    "MIND SSD mean": first_row_value(quality_df, ["mean_mind_ssd"]),
                    "Quality time [s]": first_row_value(quality_df, ["quality_eval_time_s"]),
                    "Quality FPS": first_row_value(quality_df, ["fps_quality_eval"]),
                    "Device": first_row_value(quality_df, ["device"]),
                })

            if reprojection_path.exists():
                reprojection_df = pd.read_csv(reprojection_path)
                row.update({
                    "Reprojection mean [px]": first_row_value(
                        reprojection_df,
                        ["mean_of_mean_error"],
                    ),
                    "Reprojection median [px]": first_row_value(
                        reprojection_df,
                        ["median_of_median_error"],
                    ),
                    "Reprojection frames": first_row_value(
                        reprojection_df,
                        ["num_eval_used", "num_frames"],
                    ),
                })

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    season_order = {label: idx for idx, label in enumerate(DATASETS.keys())}
    df["_season_order"] = df["Station"].map(season_order).fillna(len(season_order))
    df = df.sort_values(["_season_order", "Method"]).drop(columns=["_season_order"])

    preferred_columns = [
        "Station",
        "Method",
        "Frames",
        "GOA mean",
        "Edge Alignment mean",
        "MIND SSD mean",
        "Reprojection mean [px]",
        "Reprojection median [px]",
        "Reprojection frames",
        "Quality time [s]",
        "Quality FPS",
        "Device",
        "Method folder",
    ]
    return df[[col for col in preferred_columns if col in df.columns]]


def show_season_metrics_table():
    """Render a single large table comparing stations, metrics, and methods."""
    st.header("4 seasons results")

    df = load_season_metrics_table().copy()

    if df.empty:
        st.warning("No summary CSV files were found for the four stations.")
        return

    method_options = sorted(df["Method"].dropna().unique())
    selected_method_options = st.multiselect(
        "Methods in table",
        method_options,
        default=method_options,
        key="season_metrics_methods",
    )

    if selected_method_options:
        df = df[df["Method"].isin(selected_method_options)]

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].round(4)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

def show_comparison_tab(dataset_name, selected_methods):
    """Render the global comparison tab with plots, tables, and visual comparisons."""
    st.header("Comparison")

    comparison_dir = (
        DATA_DIR
        / dataset_name
        / "Sortie_Extraite"
        / "paired_images"
        / "benchmark_comparison_methods"
    )

    if not comparison_dir.exists():
        st.warning("benchmark_comparison_methods does not exist.")
        return

    st.info("Showing the existing comparison results.")

    show_comparison_alignment_frame(dataset_name, selected_methods)
    
    st.markdown("## GOA")
    show_existing_image(comparison_dir / "goa_boxplot.png")
    show_existing_image(comparison_dir / "goa_histogram.png")
    show_existing_image(comparison_dir / "goa_frame_by_frame.png")
    show_metric_table_from_summary(comparison_dir, "goa")

    st.markdown("## MIND SSD")
    show_existing_image(comparison_dir / "mind_boxplot.png")
    show_existing_image(comparison_dir / "mind_histogram.png")
    show_existing_image(comparison_dir / "mind_frame_by_frame.png")
    st.markdown("### MIND map with global scale")
    show_existing_image(comparison_dir / "mind_global_comparison_panel.png")
    show_metric_table_from_summary(comparison_dir, "mind")

    st.markdown("## Edge Alignment")
    show_existing_image(comparison_dir / "edge_f1_8px_boxplot.png")
    show_existing_image(comparison_dir / "edge_f1_8px_histogram.png")
    show_existing_image(comparison_dir / "edge_f1_8px_frame_by_frame.png")
    show_metric_table_from_summary(comparison_dir, "edge")

    st.markdown("## Reprojection")
    show_existing_image(comparison_dir / "reprojection_boxplot.png")
    show_existing_image(comparison_dir / "reprojection_histogram.png")
    show_existing_image(comparison_dir / "reprojection_frame_by_frame_sigma3.png")
    show_existing_image(comparison_dir / "reprojection_frame_by_frame_sigma5.png")
    show_existing_image(comparison_dir / "reprojection_frame_by_frame_sigma10.png")
    show_metric_table_from_summary(comparison_dir, "reprojection")

    st.markdown("## Runtime table")
    show_existing_table(comparison_dir / "runtime_methods.csv", table_type="runtime")

    st.markdown("## Ranking table")
    show_existing_table(comparison_dir / "ranking_methods.csv", table_type="ranking")

def extract_index(path: Path):
    """Extract the first numeric index found in a file name for sorting."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def images_by_frame_id(images):
    """Map frame ids to image paths, keeping the first sorted path for duplicates."""
    by_frame = {}
    for image in sorted(images, key=extract_index):
        frame_id = extract_index(image)
        if frame_id >= 0 and frame_id not in by_frame:
            by_frame[frame_id] = image
    return by_frame


def get_original_dirs(dataset_name):
    """Return the original VIS and SWIR directories for a dataset."""
    base = DATA_DIR / dataset_name / "Sortie_Extraite" / "paired_images"
    vis_dir = base / "VISIBLE"
    swir_dir = base / "SWIR"
    return vis_dir, swir_dir

#def show_visible_html_animation(vis_images, step=10):
    #st.markdown("### Animated VIS sequence")

    #selected = vis_images[::step]

    #st.write(
        #f"Animation using {len(selected)} frames from "
        #f"{len(vis_images)} VIS images."
    #)

    #encoded_frames = []

    #for img_path in selected:
        #with open(img_path, "rb") as f:
            #encoded = base64.b64encode(f.read()).decode()
            #encoded_frames.append(
                #f"data:image/jpeg;base64,{encoded}"
            #)

    #frames_js = str(encoded_frames)

    #html = f'''
    #<div style="text-align:center;">
        #<img id="vis_anim" style="width:800px; border-radius:8px;">
    #</div>

    #<script>
        #const frames = {frames_js};
        #let i = 0;
        #const img = document.getElementById("vis_anim");

        #function updateFrame() {{
            #img.src = frames[i];
            #i = (i + 1) % frames.length;
        #}}

        #updateFrame();
        #setInterval(updateFrame, 80);
    #</script>
    #'''

    #st.components.v1.html(html, height=550)


def show_visible_sequence(vis_images):
    """Display a slider to browse the original VIS image sequence."""
    st.subheader("VIS sequence")

    if len(vis_images) == 0:
        st.info("There are no VIS images to display.")
        return

    idx = st.slider(
        "Frame VIS",
        min_value=0,
        max_value=len(vis_images) - 1,
        value=0,
        step=1
    )

    st.write(f"Frame {idx} / {len(vis_images) - 1}")

    img_path = vis_images[idx]

    encoded = image_base64(img_path)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/jpeg;base64,{encoded}"
                 style="width:1000px; max-width:100%; border-radius:6px;">
            <p style="font-size:14px;">{img_path.name}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    #show_visible_html_animation(vis_images, step=10)



def get_method_dir(dataset_name, method_folder):
    """Build the directory path for one method inside one dataset."""
    resolved_folder = resolve_method_folder(dataset_name, method_folder)
    return paired_images_dir(dataset_name) / resolved_folder


def available_datasets():
    """Return configured datasets that exist in the current DATA_DIR."""
    existing = {
        label: name
        for label, name in DATASETS.items()
        if (DATA_DIR / name).exists()
    }
    return existing or DATASETS


def available_methods_for_dataset(dataset_name):
    """Return configured methods that have an output folder for this dataset."""
    return {
        label: folder
        for label, folder in METHODS.items()
        if get_method_dir(dataset_name, folder).exists()
    }


@st.cache_data(show_spinner=False)
def list_images(folder: Path, pattern: str):
    """List images matching one pattern and sort them by numeric frame index."""
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern), key=extract_index)


@st.cache_data(show_spinner=False)
def list_images_by_patterns(folder: Path, patterns):
    """List images matching several patterns and sort them by numeric frame index."""
    if not folder.exists():
        return []

    images = []
    for pattern in patterns:
        images.extend(folder.glob(pattern))

    return sorted(images, key=extract_index)


def bounded_number_input(label, key, minimum, maximum, default, disabled=False, help=None):
    """Render a number input while clamping stale session state to current bounds."""
    if key in st.session_state:
        st.session_state[key] = min(max(st.session_state[key], minimum), maximum)
        return st.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            key=key,
            disabled=disabled,
            help=help,
        )

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=min(max(default, minimum), maximum),
        key=key,
        disabled=disabled,
        help=help,
    )


def show_image_grid(vis_images, swir_images, title=""):
    """Show initial, middle, and final VIS/SWIR image pairs in a three-column grid."""
    if title:
        st.subheader(title)

    if len(vis_images) == 0 or len(swir_images) == 0:
        st.warning("No VIS or SWIR images were found.")
        return

    n = min(len(vis_images), len(swir_images))
    idxs = [0, n // 2, n - 1]

    cols = st.columns(3)

    labels = ["Initial frame", "Middle frame", "Final frame"]

    for col, idx, label in zip(cols, idxs, labels):
        with col:
            st.markdown(f"**{label}**")
            st.markdown("VIS")
            st.image(str(vis_images[idx]), use_container_width=True)
            st.markdown("SWIR")
            st.image(str(swir_images[idx]), use_container_width=True)


@st.cache_data(show_spinner=False)
def infer_alignment_direction(method_label: str, method_dir: Path):
    """Infer whether the displayed method aligns VIS to SWIR or SWIR to VIS."""
    text = f"{method_label} {method_dir.name}".lower()

    if "vis_to_swir" in text or "vis to swir" in text or "vis→swir" in text or "vis -> swir" in text:
        return "VIS_TO_SWIR"

    if "swir_to_vis" in text or "swir to vis" in text or "swir→vis" in text or "swir -> vis" in text:
        return "SWIR_TO_VIS"

    if "rasmd" in method_label.lower() or method_dir.name.lower() == RASMD_METHOD_FOLDER.lower():
        return "VIS_TO_SWIR"

    return None


def split_image_titles(direction):
    """Return column titles for the selected alignment direction."""
    if direction == "VIS_TO_SWIR":
        return "VIS aligned", "SWIR original"

    if direction == "SWIR_TO_VIS":
        return "VIS original", "SWIR aligned"

    return "VIS", "SWIR"


@st.cache_data(show_spinner=False)
def find_split_images(method_dir: Path, direction=None):
    """Find VIS and SWIR images inside a method's split_images folder."""
    split_dir = method_dir / "split_images"

    if not split_dir.exists():
        return [], []

    if direction == "VIS_TO_SWIR":
        vis_patterns = ["*_vis_aligned.jpg", "*VIS_aligned.jpg"]
        swir_patterns = ["*_swir.jpg", "*SWIR.jpg"]
    elif direction == "SWIR_TO_VIS":
        vis_patterns = ["*_vis.jpg", "*VIS.jpg"]
        swir_patterns = ["*_swir_aligned.jpg", "*SWIR_aligned.jpg"]
    else:
        vis_patterns = [
            "*vis.jpg",
            "*VIS.jpg",
            "*_vis_aligned.jpg",
            "*_vis.jpg",
        ]

        swir_patterns = [
            "*swir.jpg",
            "*SWIR.jpg",
            "*_swir_aligned.jpg",
            "*_swir.jpg",
        ]

    vis_images = []
    swir_images = []

    for pattern in vis_patterns:
        vis_images.extend(split_dir.glob(pattern))

    for pattern in swir_patterns:
        swir_images.extend(split_dir.glob(pattern))

    vis_images = sorted(set(vis_images), key=extract_index)
    swir_images = sorted(set(swir_images), key=extract_index)

    return vis_images, swir_images


@st.cache_data(show_spinner=False)
def find_concatenate_images(method_dir: Path):
    """Find concatenated alignment images in the common output folder variants."""
    possible_dirs = [
        method_dir / "concatenate_images",
        method_dir / "concatenate",
        method_dir / "cocatenate",
        method_dir / "concatenated",
        method_dir / "concat",
    ]

    for d in possible_dirs:
        if d.exists():
            images = sorted(
                list(d.glob("*.jpg")) +
                list(d.glob("*.png")) +
                list(d.glob("*.jpeg")),
                key=extract_index
            )
            return images

    return []


@st.cache_data(show_spinner=False)
def find_reprojection_panel_images(panels_dir: Path):
    """Find best and worst reprojection panel images."""
    if not panels_dir.exists():
        return [], []

    best_imgs = sorted(panels_dir.glob("best_*.png"))
    worst_imgs = sorted(panels_dir.glob("worst_*.png"))
    return best_imgs, worst_imgs


def show_concatenate_sequence(method_dir: Path):
    """Display a slider to browse concatenated alignment images for a method."""
    st.subheader("Alignment sequence")

    concat_images = find_concatenate_images(method_dir)

    if len(concat_images) == 0:
        st.info("No concatenate images were found.")
        return

    idx = st.slider(
        "Frame concatenate",
        min_value=0,
        max_value=len(concat_images) - 1,
        value=0,
        step=1,
        key=f"concat_slider_{method_dir.name}"
    )

    st.write(f"Frame {idx} / {len(concat_images) - 1}")

    img_path = concat_images[idx]

    encoded = image_base64(img_path)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/jpeg;base64,{encoded}"
                 style="width:1000px; max-width:100%; border-radius:6px;">
            <p style="font-size:14px;">{img_path.name}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_alignment(method_dir: Path, direction=None):
    """Show representative aligned VIS, SWIR, and concatenate images for a method."""
    st.subheader("Alignment")

    vis_title, swir_title = split_image_titles(direction)
    vis_images, swir_images = find_split_images(method_dir, direction)
    concat_images = find_concatenate_images(method_dir)

    if len(vis_images) == 0 or len(swir_images) == 0:
        st.warning("No images were found in split_images.")
        return

    n = min(len(vis_images), len(swir_images))

    if len(concat_images) > 0:
        n = min(n, len(concat_images))

    idxs = [0, n // 2, n - 1]
    labels = ["Initial image", "Middle image", "Final image"]

    for idx, label in zip(idxs, labels):
        st.markdown(f"### {label}")

        cols = st.columns(3)

        with cols[0]:
            st.markdown(vis_title)
            st.image(str(vis_images[idx]), use_container_width=True)

        with cols[1]:
            st.markdown(swir_title)
            st.image(str(swir_images[idx]), use_container_width=True)

        with cols[2]:
            st.markdown("Concatenate")
            if len(concat_images) > 0:
                st.image(str(concat_images[idx]), use_container_width=True)
            else:
                st.info("No concatenate/cocatenate folder exists.")


@st.cache_data(show_spinner=False)
def load_scores(method_dir: Path):
    """Load the registration quality scores CSV for a method."""
    score_path = method_dir / "results" / "scores_qualite_recalage.csv"

    if not score_path.exists():
        score_path = method_dir / "scores_qualite_recalage.csv"

    if not score_path.exists():
        return None, None

    df = read_csv_cached(score_path)
    return df, score_path


def metric_column(df, possible_names):
    """Return the matching column name for a metric from a list of possible names."""
    cols_lower = {c.lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]

    return None


def show_metric(df, metric_name, possible_columns):
    """Display statistics and plots for one metric column."""
    col = metric_column(df, possible_columns)

    if col is None:
        st.info(f"The {metric_name} metric was not found.")
        return

    values = df[col].dropna()

    st.markdown(f"## {metric_name}")

    stats_html = f"""
    <table style="
        width:100%;
        border-collapse:collapse;
        text-align:center;
        font-size:16px;
    ">
        <tr>
            <th style="text-align:center;"><b>Mean</b></th>
            <th style="text-align:center;"><b>Median</b></th>
            <th style="text-align:center;"><b>Std</b></th>
            <th style="text-align:center;"><b>Min</b></th>
            <th style="text-align:center;"><b>Max</b></th>
        </tr>
        <tr>
            <td style="text-align:center;">{values.mean():.4f}</td>
            <td style="text-align:center;">{values.median():.4f}</td>
            <td style="text-align:center;">{values.std():.4f}</td>
            <td style="text-align:center;">{values.min():.4f}</td>
            <td style="text-align:center;">{values.max():.4f}</td>
        </tr>
    </table>
    """

    st.markdown(stats_html, unsafe_allow_html=True)

    # ==================================
    # BOXPLOT + HISTOGRAMA
    # ==================================

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5,3))
        ax1.boxplot(values)
        ax1.set_title("Boxplot")
        ax1.set_ylabel(metric_name)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.hist(values, bins=30)
        ax2.set_title("Histogram")
        ax2.set_xlabel(metric_name)
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # ==================================
    # FRAME BY FRAME (sigma = 5)
    # ==================================

    smooth = gaussian_filter1d(values.values, sigma=5)

    fig3, ax3 = plt.subplots(figsize=(10,3))

    ax3.plot(
        values.values,
        alpha=0.25,
        label="Raw"
    )

    ax3.plot(
        smooth,
        linewidth=2,
        label="σ = 5"
    )

    ax3.legend()

    ax3.set_title(f"Frame by frame - {metric_name} (σ=5)")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel(metric_name)

    st.pyplot(fig3)


def show_results(method_dir: Path):
    """Display all available quality metrics and related result images for a method."""
    st.subheader("Results")

    df, score_path = load_scores(method_dir)

    if df is None:
        st.warning("scores_qualite_recalage.csv was not found.")
        return

    show_metric(df, "GOA", ["goa", "GOA"])

    show_metric(df, "MIND SSD", ["mind_ssd", "MIND_SSD", "mind"])
    show_mind_maps(method_dir)

    show_metric(df, "Edge Alignment", ["edge_f1_8px", "Edge Alignment"])

    show_reprojection_aligned(method_dir)

def show_saved_images_from_dir(title, folder: Path, patterns=("*.png", "*.jpg", "*.jpeg")):
    """Display up to three saved images from a folder."""
    st.subheader(title)

    if not folder.exists():
        st.info(f"Does not exist: {folder.name}")
        return

    images = list_images_by_patterns(folder, patterns)

    if not images:
        st.info(f"There are no images in {folder.name}.")
        return

    cols = st.columns(min(3, len(images)))
    for col, img in zip(cols, images[:3]):
        with col:
            st.image(str(img), caption=img.name, use_container_width=True)


def show_histogram_files(method_dir: Path):
    """Load and display the homography matrix associated with a method."""
    st.subheader("Homography")

    possible_files = [
        method_dir / "homography.npz",
        method_dir / "homography.npy",
        method_dir / "homography_opencv_global.npy",
    ]

    H = None
    used_file = None

    for file in possible_files:

        if not file.exists():
            continue

        H = load_homography(file)

        if H is not None:
            used_file = file
            break

    if H is None:
        st.info("homography.npz, homography.npy, or homography_opencv_global.npy was not found.")
        return

    if H.ndim == 3:
        H = H[0]

    if H.shape != (3, 3):
        st.warning(f"The homography found is not 3x3. Shape: {H.shape}")
        return

    direction = "VIS\\rightarrow SWIR"

    method_str = str(method_dir)

    if (
        "SWIR_to_VIS" in method_str
        or "SWIR_TO_VIS" in method_str
        or "SWIR_to_VIS".lower() in method_str.lower()
        or "SWIR_TO_VIS".lower() in method_str.lower()
    ):
        direction = "SWIR\\rightarrow VIS"

    st.caption(f"File used: {used_file.name}")

    latex_H = rf"""
    H_{{\mathrm{{{direction}}}}} =
    \begin{{bmatrix}}
    {H[0,0]:.4f} & {H[0,1]:.4f} & {H[0,2]:.2f} \\
    {H[1,0]:.4f} & {H[1,1]:.4f} & {H[1,2]:.2f} \\
    {H[2,0]:.2e} & {H[2,1]:.2e} & {H[2,2]:.4f}
    \end{{bmatrix}}
    """

    st.latex(latex_H)


def show_calibration(method_dir: Path):
    """Display VIS and SWIR calibration debug images when available."""
    st.subheader("Calibration")

    calib_candidates = [
        method_dir / "calibrations",
        method_dir / "results",
        method_dir,
    ]

    found = []

    for folder in calib_candidates:
        for name in ["vis_calib_debug.png", "swir_calib_debug.png"]:
            p = folder / name
            if p.exists():
                found.append(p)

    if not found:
        st.info("There are no calibration images for this method.")
        return

    for img in found:

        if "vis" in img.name.lower():
            st.markdown("### VIS Calibration")
        else:
            st.markdown("### SWIR Calibration")

        st.image(
            str(img),
            width="stretch"
        )

def show_train_images(method_dir: Path):
    """Display the training images used by a patch or OpenCV method."""
    st.subheader("Train images")

    train_dir = method_dir / "train_images"

    if not train_dir.exists():
        st.info("train_images folder does not exist.")
        return

    vis_dir = train_dir / "VISIBLE"
    swir_dir = train_dir / "SWIR"

    vis_images = list_images(vis_dir, "*.jpg")
    swir_images = list_images(swir_dir, "*.jpg")

    n = min(len(vis_images), len(swir_images))

    if n == 0:
        st.info("No VIS/SWIR train images were found.")
        return

    first_idx = extract_index(vis_images[0])
    last_idx = extract_index(vis_images[-1])

    st.write(f"Used **{n} train images**, from image **{first_idx}** to **{last_idx}**.")

    if n > 1:
        step = extract_index(vis_images[1]) - extract_index(vis_images[0])
        st.write(f"Approximate step: **{step}**")

    show_image_grid(vis_images, swir_images)


def show_pre_h_crop_images(method_dir: Path):
    """Display the pre-homography crop images used by patch methods."""
    st.subheader("PRE-H crop images")

    crop_dir = method_dir / "pre_h_crop_images"

    if not crop_dir.exists():
        st.info("pre_h_crop_images folder does not exist.")
        return

    vis_dir = crop_dir / "VISIBLE"
    swir_dir = crop_dir / "SWIR"

    vis_images = list_images(vis_dir, "*.jpg")
    swir_images = list_images(swir_dir, "*.jpg")

    if len(vis_images) == 0 or len(swir_images) == 0:
        st.info("No VIS/SWIR images were found in pre_h_crop_images.")
        return

    show_image_grid(vis_images, swir_images)


@st.cache_data(show_spinner=False)
def parse_runtime_txt(runtime_path: Path):
    """Parse runtime_metrics.txt into a dictionary of metric names and values."""
    data = {}

    text = read_text_cached(runtime_path)

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip().replace("s", "").strip()

        try:
            value = float(value)
        except ValueError:
            pass

        data[key] = value

    return data


def show_runtime(method_dir: Path):
    """Display runtime metrics for a method."""
    st.subheader("Runtime")

    runtime_txt = method_dir / "runtime_metrics.txt"

    if not runtime_txt.exists():
        st.info("runtime_metrics.txt was not found.")
        return

    runtime_data = parse_runtime_txt(runtime_txt)

    train_images = runtime_data.get("Train images", "-")
    test_images = runtime_data.get("Test images", "-")
    total_time = runtime_data.get("Temps total", None)
    fps_global = runtime_data.get("FPS global", None)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Train images", f"{int(train_images)}" if train_images != "-" else "-")
    col2.metric("Test images", f"{int(test_images)}" if test_images != "-" else "-")
    col3.metric("Total time", f"{total_time:.2f} s" if total_time is not None else "-")
    col4.metric("FPS global", f"{fps_global:.2f}" if fps_global is not None else "-")


def show_mind_maps(method_dir: Path):
    """Display the saved mean MIND map if one exists."""
    st.subheader("MIND maps")

    possible_paths = [
        method_dir / "results" / "mind_map_mean.png",
        method_dir / "results" / "mind_maps" / "mind_map_mean.png",
        method_dir / "results" / "MIND" / "mind_map_mean.png",
    ]

    for img_path in possible_paths:
        if img_path.exists():
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(
                    str(img_path),
                    caption="mind_map_mean.png",
                    width=600
                )
            return

    st.info("mind_map_mean.png was not found.")

def show_reprojection_aligned(method_dir: Path):
    """Display reprojection statistics, plots, and best/worst panels."""
    reproj_dir = method_dir / "results" / "reprojection_aligned"

    if not reproj_dir.exists():
        st.info("results/reprojection_aligned folder was not found.")
        return

    st.markdown("## Reprojection")

    framewise_path = reproj_dir / "reprojection_aligned_framewise.csv"

    if not framewise_path.exists():
        st.info("reprojection_aligned_framewise.csv was not found.")
        return

    frame_df = read_csv_cached(framewise_path)

    value_col = None
    for c in frame_df.columns:
        if (
            "reproj" in c.lower()
            or "error" in c.lower()
            or "mean" in c.lower()
            or "median" in c.lower()
        ):
            value_col = c
            break

    if value_col is None:
        st.info("No reprojection column was found in the framewise CSV.")
        return

    values = frame_df[value_col].dropna()

    # ==================================
    # STATISTICS TABLE
    # ==================================

    stats_html = f"""
    <table style="
        width:100%;
        border-collapse:collapse;
        text-align:center;
        font-size:16px;
    ">
        <tr>
            <th style="text-align:center;"><b>Mean</b></th>
            <th style="text-align:center;"><b>Median</b></th>
            <th style="text-align:center;"><b>Std</b></th>
            <th style="text-align:center;"><b>Min</b></th>
            <th style="text-align:center;"><b>Max</b></th>
        </tr>
        <tr>
            <td style="text-align:center;">{values.mean():.4f}</td>
            <td style="text-align:center;">{values.median():.4f}</td>
            <td style="text-align:center;">{values.std():.4f}</td>
            <td style="text-align:center;">{values.min():.4f}</td>
            <td style="text-align:center;">{values.max():.4f}</td>
        </tr>
    </table>
    """

    st.markdown(stats_html, unsafe_allow_html=True)

    # ==================================
    # BOXPLOT + HISTOGRAM
    # ==================================

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.boxplot(values)
        ax1.set_title("Boxplot")
        ax1.set_ylabel("Reprojection")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(values, bins=30)
        ax2.set_title("Histogram")
        ax2.set_xlabel("Reprojection")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # ==================================
    # FRAME BY FRAME sigma = 5
    # ==================================

    smooth = gaussian_filter1d(values.values, sigma=5)

    fig3, ax3 = plt.subplots(figsize=(10, 3))

    ax3.plot(
        values.values,
        alpha=0.25,
        label="Raw"
    )

    ax3.plot(
        smooth,
        linewidth=2,
        label="σ = 5"
    )

    ax3.legend()
    ax3.set_title("Frame by frame - Reprojection (σ=5)")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Reprojection")

    st.pyplot(fig3)

    # ==================================
    # PANELS: EACH IMAGE IN ONE ROW
    # ==================================

    panels_dir = reproj_dir / "reprojection_aligned_panels"

    if panels_dir.exists():
        best_imgs, worst_imgs = find_reprojection_panel_images(panels_dir)

        st.markdown("### 3 best")

        for img in best_imgs[:3]:

            st.image(
                str(img),
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 3 worst")

        for img in worst_imgs[:3]:

            st.image(
                str(img),
                use_container_width=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

def show_method_tab(dataset_name, method_label, method_folder, comparison_mode=False):
    """Render one method tab with homography, images, runtime, and metric results."""
    method_dir = get_method_dir(dataset_name, method_folder)

    st.subheader(method_label)
    #st.code(str(method_dir))
    if method_dir.name != method_folder:
        st.caption(f"Using available folder: {method_dir.name}")

    if not method_dir.exists():
        st.error("This method folder does not exist.")
        return

    is_patch = "Patch" in method_label
    is_opencv = "OpenCV" in method_label
    is_rasmd = "RASMD" in method_label

    # ============================
    # STANDARD METHOD
    # ============================

    if is_patch or is_opencv:
        show_histogram_files(method_dir)

    if is_patch:
        show_calibration(method_dir)

    if is_patch or is_opencv:
        show_train_images(method_dir)

    if is_patch:
        show_pre_h_crop_images(method_dir)

    direction = infer_alignment_direction(method_label, method_dir)

    show_alignment(method_dir, direction)
    show_concatenate_sequence(method_dir)

    if is_patch or is_opencv:
        show_runtime(method_dir)

    show_results(method_dir)

    # ============================
    # EXTRA: CROPPED COMPARISON
    # ============================

    if comparison_mode and not is_rasmd:
        cropped_folder = method_folder + "_cropped_like_rasmd"
        cropped_dir = get_method_dir(dataset_name, cropped_folder)

        st.markdown("---")
        st.markdown("## Comparison with shared crop")

        st.info(
            "Because comparison mode is enabled, the cropped method version is also "
            "shown for a fair comparison using the shared region across methods."
        )

        #st.code(str(cropped_dir))

        if not cropped_dir.exists():
            st.warning("The cropped folder does not exist for this method.")
            return

        cropped_direction = infer_alignment_direction(method_label, cropped_dir)

        show_alignment(cropped_dir, cropped_direction)
        show_concatenate_sequence(cropped_dir)
        show_results(cropped_dir)


@st.cache_data(show_spinner=False)
def list_pdfs(folder: Path):
    """List PDF files from a folder."""
    if not folder.exists():
        return []

    return sorted(
        list(folder.glob("*.pdf")) + list(folder.glob("*.PDF")),
        key=lambda path: path.name.lower()
    )


def select_poster_and_report(pdfs):
    """Select only the poster and report PDFs, in that order."""
    poster = next(
        (path for path in pdfs if "poster" in path.name.lower()),
        None
    )

    report = next(
        (
            path for path in pdfs
            if "report" in path.name.lower() or "reporte" in path.name.lower()
        ),
        None
    )

    documents = []

    if poster is not None:
        documents.append(("Poster", poster))

    if report is not None:
        documents.append(("Report", report))

    return documents


def show_pdf_embed(title: str, path: Path, height=760):
    """Embed one PDF file in the Streamlit page."""
    encoded_pdf = file_base64(path)

    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded_pdf}"
            width="100%"
            height="{height}"
            style="border:1px solid #ddd; border-radius:6px;"
        ></iframe>
        """,
        unsafe_allow_html=True
    )


def show_pdf_tab():
    """Render the poster first, then the report."""
    st.header("Documents")

    documents = select_poster_and_report(list_pdfs(PDF_DIR))

    if not documents:
        st.info(f"Add a poster PDF and a report PDF to this folder: {PDF_DIR}")
        return

    if len(documents) == 1:
        title, pdf_path = documents[0]
        show_pdf_embed(title, pdf_path)
        return

    cols = st.columns(2)

    for col, (title, pdf_path) in zip(cols, documents):
        with col:
            show_pdf_embed(title, pdf_path, height=720)


def show_academic_header():
    """Render the academic dashboard header inspired by the scientific poster."""
    uca_logo = file_base64(UCA_LOGO_PATH) if UCA_LOGO_PATH.exists() else ""
    inp_logo = file_base64(INP_LOGO_PATH) if INP_LOGO_PATH.exists() else ""
    cnrs_logo = file_base64(CNRS_LOGO_PATH) if CNRS_LOGO_PATH.exists() else ""
    ip_logo = file_base64(IP_LOGO_PATH) if IP_LOGO_PATH.exists() else ""

    uca_logo_html = (
        f'<img class="poster-header__uca-img" src="data:image/png;base64,{uca_logo}" '
        'alt="Université Clermont Auvergne logo">'
        if uca_logo else
        '<div class="poster-header__fallback-logo">Université<br>Clermont<br>Auvergne</div>'
    )

    partner_logos_html = (
        '<div class="poster-header__partner-stack">'
        '<div class="poster-header__partner-row">'
        + (
            f'<img class="poster-header__inp-img" src="data:image/png;base64,{inp_logo}" '
            'alt="Clermont Auvergne INP logo">'
            if inp_logo else
            '<div class="poster-header__fallback-logo">Clermont<br>Auvergne<br>INP</div>'
        )
        + (
            f'<img class="poster-header__cnrs-img" src="data:image/png;base64,{cnrs_logo}" '
            'alt="CNRS logo">'
            if cnrs_logo else
            '<div class="poster-header__fallback-logo">CNRS</div>'
        )
        + '</div>'
        + (
            f'<img class="poster-header__ip-img" src="data:image/png;base64,{ip_logo}" '
            'alt="Institut Pascal logo">'
            if ip_logo else
            '<div class="poster-header__fallback-logo">Institut Pascal</div>'
        )
        + '</div>'
    )

    header_html = textwrap.dedent("""
        <style>
            body {
                margin: 0;
            }

            .main .block-container {
                padding-top: 1.2rem;
            }

            .poster-header {
                width: 100%;
                background: #ffffff;
                margin: 0 0 1rem 0;
                font-family: Arial, Helvetica, sans-serif;
            }

            .poster-header__grid {
                display: grid;
                grid-template-columns: 15% 70% 15%;
                align-items: start;
                min-height: 178px;
                padding: 18px 18px 10px 18px;
                column-gap: 12px;
            }

            .poster-header__left,
            .poster-header__right {
                display: flex;
                align-items: center;
            }

            .poster-header__left {
                justify-content: center;
                padding-top: 10px;
            }

            .poster-header__right {
                justify-content: center;
            }

            .poster-header__uca-img {
                display: block;
                width: min(100%, 168px);
                height: auto;
                object-fit: contain;
            }

            .poster-header__center {
                text-align: center;
                color: #000000;
                min-width: 0;
                padding-top: 22px;
            }

            .poster-header__title {
                margin: 0;
                font-size: clamp(1.35rem, 2.35vw, 2.35rem);
                line-height: 1.05;
                font-weight: 800;
                letter-spacing: 0;
            }

            .poster-header__authors {
                margin-top: 11px;
                color: #f28c28;
                font-size: clamp(0.86rem, 1.2vw, 1.05rem);
                font-weight: 500;
            }

            .poster-header__authors strong {
                font-weight: 800;
            }

            .poster-header__affiliation {
                margin-top: 7px;
                color: #d97921;
                font-size: clamp(0.72rem, 1vw, 0.9rem);
                font-weight: 500;
            }

            .poster-header__fallback-logo {
                color: #25445f;
                font-size: 0.9rem;
                font-weight: 800;
                line-height: 1.1;
                text-align: center;
            }

            .poster-header__partner-stack {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 9px;
                width: 100%;
            }

            .poster-header__partner-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                align-items: center;
                justify-items: center;
                column-gap: 8px;
                width: 100%;
            }

            .poster-header__inp-img,
            .poster-header__cnrs-img {
                display: block;
                width: 100%;
                max-width: 78px;
                height: auto;
                object-fit: contain;
            }

            .poster-header__ip-img {
                display: block;
                width: 100%;
                max-width: 180px;
                height: auto;
                object-fit: contain;
            }

            .poster-header__blue-line {
                height: 4px;
                background: #5f829d;
                margin: 0 22px;
            }

            @media (max-width: 900px) {
                .poster-header__grid {
                    grid-template-columns: 1fr;
                    padding: 14px 18px;
                    text-align: center;
                    gap: 14px;
                }

                .poster-header__left,
                .poster-header__right {
                    justify-content: center;
                    padding-top: 0;
                }

                .poster-header__center {
                    text-align: center;
                }

                .poster-header__uca-img {
                    width: min(58%, 170px);
                }

                .poster-header__partner-stack {
                    width: min(74%, 230px);
                }
            }
        </style>

        <section class="poster-header">
            <div class="poster-header__grid">
                <div class="poster-header__left">
                    {UCA_LOGO_HTML}
                </div>

                <div class="poster-header__center">
                    <h1 class="poster-header__title">
                        A Multi-Metric Benchmark for VIS&ndash;SWIR Image Registration under Multiple Weather Conditions
                    </h1>
                    <div class="poster-header__authors">
                        <strong>Lizbeth Torres Flores</strong>, Alexandre Riffard and Mathieu Labussière
                    </div>
                    <div class="poster-header__affiliation">
                        Université Clermont Auvergne, Clermont Auvergne INP, CNRS, Institut Pascal,
                        F-63000 Clermont-Ferrand, France
                    </div>
                </div>

                <div class="poster-header__right">
                    {PARTNER_LOGOS_HTML}
                </div>
            </div>
            <div class="poster-header__blue-line"></div>
        </section>
        """)

    header_html = header_html.replace("{UCA_LOGO_HTML}", uca_logo_html)
    header_html = header_html.replace("{PARTNER_LOGOS_HTML}", partner_logos_html)

    components.html(header_html, height=225, scrolling=False)


def show_sidebar_acknowledgments():
    """Render the acknowledgments block in the Streamlit sidebar."""
    itps_logo = file_base64(ITPS_LOGO_PATH) if ITPS_LOGO_PATH.exists() else ""
    isite_logo = file_base64(ISITE_LOGO_PATH) if ISITE_LOGO_PATH.exists() else ""

    itps_logo_html = (
        f'<img class="acknowledgments__logo" src="data:image/png;base64,{itps_logo}" '
        'alt="International Research Centre on Innovative Transportation and Production Systems logo">'
        if itps_logo else
        ""
    )

    isite_logo_html = (
        f'<img class="acknowledgments__logo" src="data:image/png;base64,{isite_logo}" '
        'alt="CAP 20-25 I-SITE Clermont Auvergne Project logo">'
        if isite_logo else
        ""
    )

    st.sidebar.markdown(
        textwrap.dedent(f"""
        <style>
            [data-testid="stSidebarUserContent"] {{
                min-height: calc(100vh - 3rem);
                display: flex;
                flex-direction: column;
                padding-bottom: 15rem;
            }}

            [data-testid="stSidebar"] .acknowledgments {{
                position: fixed;
                left: 1.4rem;
                bottom: 1.35rem;
                width: min(15.8rem, calc(100vw - 3rem));
                z-index: 10;
            }}

            .acknowledgments {{
                margin-top: 0;
                padding: 0.25rem 0 0.25rem 0;
                background: transparent;
                font-family: Arial, Helvetica, sans-serif;
            }}

            .acknowledgments__title {{
                margin: 0 0 0.55rem 0;
                color: inherit;
                font-size: 1rem;
                font-weight: 800;
            }}

            .acknowledgments__text {{
                margin: 0 0 0.75rem 0;
                color: inherit;
                font-size: 0.76rem;
                line-height: 1.34;
                text-align: left;
                opacity: 0.9;
            }}

            .acknowledgments__logos {{
                display: flex;
                flex-direction: column;
                gap: 0.55rem;
                align-items: center;
            }}

            .acknowledgments__logo {{
                display: block;
                width: 100%;
                max-width: 250px;
                height: auto;
                object-fit: contain;
            }}
        </style>

        <section class="acknowledgments">
            <h2 class="acknowledgments__title">Acknowledgments</h2>
            <p class="acknowledgments__text">
                This work was supported by the International Research Centre
                "Innovative Transport and Production Systems" of the CAP 20-25 I-SITE.
            </p>
            <div class="acknowledgments__logos">
                {itps_logo_html}
                {isite_logo_html}
            </div>
        </section>
        """),
        unsafe_allow_html=True
    )


def selected_method_folder(method_family, direction, train_count, train_step, test_start, test_end, cropped):
    """Return the method folder selected in the automation UI."""
    if method_family == "Patch MAGSAC++":
        folder = PATCH_METHOD_FOLDERS[direction]
    elif method_family == "OpenCV":
        folder = opencv_method_folder(direction, train_count, train_step, test_start, test_end)
    else:
        folder = RASMD_METHOD_FOLDER

    return cropped_method_folder(folder) if cropped and folder != RASMD_METHOD_FOLDER else folder


def run_alignment_step(method_family, dataset_name, direction, train_start, train_step, train_count, test_start, test_end, calib_samples):
    """Run Patch or OpenCV alignment from the dashboard."""
    py = python_executable()

    if method_family == "Patch MAGSAC++":
        set_patch_direction(direction)
        command = [
            py,
            "main.py",
            "--bag",
            dataset_name,
            "--train-start",
            train_start,
            "--train-step",
            train_step,
            "--train-count",
            train_count,
            "--test-start",
            test_start,
            "--test-end",
            test_end,
            "--calib-samples",
            calib_samples,
        ]
        return run_dashboard_command(
            command,
            BASE_DIR / "projet",
            f"Patch MAGSAC++ alignment ({direction})",
        )

    if method_family == "OpenCV":
        command = [
            py,
            "opencv_align_vis_swir.py",
            "--dataset",
            dataset_name,
            "--direction",
            direction,
            "--train-start",
            train_start,
            "--train-step",
            train_step,
            "--train-count",
            train_count,
            "--test-start",
            test_start,
            "--test-end",
            test_end,
        ]
        return run_dashboard_command(
            command,
            BASE_DIR,
            f"OpenCV alignment ({direction})",
        )

    st.info("RASMD has no alignment command in this dashboard.")
    return True


def run_crop_step(method_family, dataset_name, direction, train_count, train_step, test_start, test_end, num_refs):
    """Crop one method output using the RASMD reference region."""
    if method_family == "RASMD":
        st.info("RASMD is already the crop reference.")
        return True

    if method_family == "Patch MAGSAC++":
        method_folder = PATCH_METHOD_FOLDERS[direction]
    else:
        method_folder = opencv_method_folder(direction, train_count, train_step, test_start, test_end)

    command = [
        python_executable(),
        "crop_patch_auto_like_rasmd.py",
        "--method-split",
        split_images_path(dataset_name, method_folder),
        "--rasmd-split",
        split_images_path(dataset_name, RASMD_METHOD_FOLDER),
        "--out-split",
        split_images_path(dataset_name, cropped_method_folder(method_folder)),
        "--num-refs",
        num_refs,
    ]

    return run_dashboard_command(
        command,
        BASE_DIR,
        f"Crop {method_family} output like RASMD",
    )


def run_quality_step(dataset_name, method_folder):
    """Run the quality metrics benchmark for one method folder."""
    command = [
        python_executable(),
        "-u",
        "-m",
        "registration.evaluation.quality_bench",
        quality_relative_split_path(dataset_name, method_folder),
        "--device",
        "auto",
    ]

    return run_dashboard_command(
        command,
        BASE_DIR / "projet",
        f"Quality metrics: {method_folder}",
    )


def run_reprojection_step(dataset_name, method_folder, num_eval):
    """Run reprojection evaluation for one method folder."""
    command = [
        python_executable(),
        "reprojection_aligned.py",
        "--dataset",
        dataset_name,
        "--method",
        method_folder,
        "--num-eval",
        num_eval,
        "--no-ransac",
    ]

    return run_dashboard_command(
        command,
        BASE_DIR,
        f"Reprojection: {method_folder}",
    )


def run_final_comparison_step(dataset_name, methods):
    """Run the final multi-method comparison."""
    command = [
        python_executable(),
        "-m",
        "registration.evaluation.compare_methods",
        "--dataset",
        dataset_name,
        "--methods",
        *methods,
    ]

    return run_dashboard_command(
        command,
        BASE_DIR / "projet",
        "Final comparison",
    )


def show_compile_results_page():
    """Show controls that automate the evaluation workflow."""
    if VIEW_ONLY:
        st.header("Viewer mode")
        st.info("This shared dashboard is read-only. Existing results can be viewed, but new results cannot be compiled.")
        return

    st.header("Compile new results")

    st.info(
        "Use this page to run the same evaluation steps without typing every command "
        "in the terminal. Long steps can take several minutes."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        dataset_label = st.selectbox("Dataset", list(DATASETS.keys()), key="compile_dataset")
        dataset_name = DATASETS[dataset_label]
        method_family = st.selectbox(
            "Method",
            ["Patch MAGSAC++", "OpenCV", "RASMD"],
            key="compile_method_family",
        )

    if method_family == "RASMD":
        st.session_state["compile_use_cropped"] = False

    has_alignment = method_family in ["Patch MAGSAC++", "OpenCV"]
    use_cropped_disabled = method_family == "RASMD"

    with col_b:
        direction_label = st.selectbox(
            "Direction",
            list(DIRECTIONS.keys()),
            key="compile_direction",
            disabled=not has_alignment,
        )
        direction = DIRECTIONS[direction_label]
        use_cropped = st.checkbox(
            "Use cropped output",
            value=False,
            disabled=use_cropped_disabled,
            key="compile_use_cropped",
        )
        use_cropped = use_cropped and method_family != "RASMD"

    compile_vis_dir, compile_swir_dir = get_original_dirs(dataset_name)
    compile_vis_count = len(list_images(compile_vis_dir, "rgb_image_*.jpg"))
    compile_swir_count = len(list_images(compile_swir_dir, "swir_image_*.jpg"))
    dataset_frame_count = min(compile_vis_count, compile_swir_count)
    max_frame_index = max(dataset_frame_count - 1, 0)

    st.caption(
        f"Dataset frames available: {dataset_frame_count} "
        f"(valid frame indices: 0 to {max_frame_index})."
    )

    st.caption(
        "Patch MAGSAC++ and OpenCV support both VIS -> SWIR and SWIR -> VIS. "
        "RASMD is an existing aligned baseline, so it has no direction or alignment step."
    )

    st.markdown("### Train / test split")
    split_cols = st.columns(6)
    split_disabled = not has_alignment or use_cropped
    calibration_disabled = method_family != "Patch MAGSAC++" or use_cropped
    default_train_start = min(500, max_frame_index)
    default_test_start = min(500, max_frame_index)
    default_test_end = min(1499, max_frame_index)
    max_train_step = max(1, dataset_frame_count)

    with split_cols[0]:
        train_start = bounded_number_input(
            "Train start",
            "compile_train_start",
            0,
            max_frame_index,
            default_train_start,
            disabled=split_disabled,
        )
    with split_cols[1]:
        train_step = bounded_number_input(
            "Train step",
            "compile_train_step",
            1,
            max_train_step,
            5,
            disabled=split_disabled,
        )
    max_train_count = max(1, ((max_frame_index - train_start) // train_step) + 1)
    with split_cols[2]:
        train_count = bounded_number_input(
            "Train count",
            "compile_train_count",
            1,
            max_train_count,
            min(150, max_train_count),
            disabled=split_disabled,
        )
    with split_cols[3]:
        test_start = bounded_number_input(
            "Test start",
            "compile_test_start",
            0,
            max_frame_index,
            default_test_start,
            disabled=split_disabled,
        )
    min_test_end = test_start
    with split_cols[4]:
        test_end = bounded_number_input(
            "Test end",
            "compile_test_end",
            min_test_end,
            max_frame_index,
            max(default_test_end, min_test_end),
            disabled=split_disabled,
        )
    with split_cols[5]:
        calib_samples = bounded_number_input(
            "Calibration samples",
            "compile_calib_samples",
            1,
            max(1, train_count),
            min(50, train_count),
            disabled=calibration_disabled,
        )

    if use_cropped:
        st.info(
            "Using an existing cropped output disables alignment, crop, train/test, "
            "and calibration controls. You can run quality metrics or reprojection on "
            "the selected cropped folder."
        )

    method_folder = selected_method_folder(
        method_family,
        direction,
        train_count,
        train_step,
        test_start,
        test_end,
        use_cropped,
    )

    st.markdown("### Selected output")
    st.code(method_folder, language="text")

    st.markdown("### Actions")
    st.markdown(
        """
        <style>
            div[data-testid="stNumberInput"] {
                max-width: 220px;
                margin-left: auto;
                margin-right: auto;
            }

            div[data-testid="stButton"] {
                display: flex;
                justify-content: center;
                width: 100%;
            }

            div[data-testid="stButton"] > button {
                width: 100%;
                min-width: 120px;
            }

            div[data-testid="stButton"] button {
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    alignment_disabled = not has_alignment or use_cropped
    crop_disabled = not has_alignment or use_cropped
    workflow_disabled = not has_alignment or use_cropped

    _, actions_table, _ = st.columns([0.25, 4.5, 0.25])

    with actions_table:
        input_cols = st.columns(5)

        with input_cols[1]:
            num_refs = st.number_input(
                "Crop reference frames",
                min_value=1,
                max_value=100,
                value=20,
                disabled=crop_disabled,
                help="Used only by Run crop.",
            )
        with input_cols[3]:
            num_eval = bounded_number_input(
                "Reprojection frames",
                "compile_num_eval",
                1,
                max(1, dataset_frame_count),
                min(1000, max(1, dataset_frame_count)),
            )

        button_cols = st.columns(5)

        with button_cols[0]:
            run_alignment = st.button(
                "Run alignment",
                disabled=alignment_disabled,
                use_container_width=True,
            )
        with button_cols[1]:
            run_crop = st.button(
                "Run crop",
                disabled=crop_disabled,
                use_container_width=True,
            )
        with button_cols[2]:
            run_quality = st.button("Run quality", use_container_width=True)
        with button_cols[3]:
            run_reprojection = st.button("Run reprojection", use_container_width=True)
        with button_cols[4]:
            run_full_method = st.button(
                "Run full workflow",
                disabled=workflow_disabled,
                help="Runs alignment, crop, quality metrics, and reprojection in sequence.",
                use_container_width=True,
            )

    if run_alignment:
        run_alignment_step(method_family, dataset_name, direction, train_start, train_step, train_count, test_start, test_end, calib_samples)

    if run_crop:
        run_crop_step(method_family, dataset_name, direction, train_count, train_step, test_start, test_end, num_refs)

    if run_quality:
        run_quality_step(dataset_name, method_folder)

    if run_reprojection:
        run_reprojection_step(dataset_name, method_folder, num_eval)

    if run_full_method:
        ok = run_alignment_step(method_family, dataset_name, direction, train_start, train_step, train_count, test_start, test_end, calib_samples)
        if ok:
            ok = run_crop_step(method_family, dataset_name, direction, train_count, train_step, test_start, test_end, num_refs)
        if ok:
            cropped_folder = selected_method_folder(
                method_family,
                direction,
                train_count,
                train_step,
                test_start,
                test_end,
                True,
            )
            ok = run_quality_step(dataset_name, cropped_folder)
        if ok:
            run_reprojection_step(dataset_name, cropped_folder, num_eval)

    st.markdown("### Global comparison")
    st.write("Run this after the selected methods have quality metrics and reprojection results.")

    comparison_options = list_comparison_method_options(dataset_name)
    preferred_comparison_folders = [
        cropped_method_folder(PATCH_METHOD_FOLDERS["VIS_TO_SWIR"]),
        cropped_method_folder(PATCH_METHOD_FOLDERS["SWIR_TO_VIS"]),
        cropped_method_folder(opencv_method_folder("VIS_TO_SWIR", train_count, train_step, test_start, test_end)),
        cropped_method_folder(opencv_method_folder("SWIR_TO_VIS", train_count, train_step, test_start, test_end)),
        RASMD_METHOD_FOLDER,
    ]
    default_comparison_folders = [
        folder
        for folder in preferred_comparison_folders
        if folder in comparison_options
    ]

    selected_comparison_folders = st.multiselect(
        "Methods to compare",
        list(comparison_options.keys()),
        default=default_comparison_folders,
        key="compile_comparison_methods",
    )

    if selected_comparison_folders:
        preview_command = [
            python_executable(),
            "-m",
            "registration.evaluation.compare_methods",
            "--dataset",
            dataset_name,
            "--methods",
            *selected_comparison_folders,
        ]
        st.code(command_text(preview_command), language="bash")

    if len(selected_comparison_folders) < 2:
        st.warning("Select at least two methods to compare.")

    if st.button("Run final comparison", disabled=len(selected_comparison_folders) < 2):
        run_final_comparison_step(dataset_name, selected_comparison_folders)


show_academic_header()

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: none;
            padding-left: 1.35rem;
            padding-right: 1.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

mode_options = ["Show existing results"]
if not VIEW_ONLY:
    mode_options.append("Compile new results")

mode = st.sidebar.radio("Mode", mode_options)

if VIEW_ONLY:
    st.sidebar.caption("Viewer mode: existing results only.")

show_sidebar_acknowledgments()

if mode == "Compile new results":
    show_compile_results_page()
    st.stop()

dataset_options = available_datasets()

dataset_label = st.sidebar.selectbox(
    "Dataset",
    list(dataset_options.keys())
)

dataset_name = dataset_options[dataset_label]

vis_dir, swir_dir = get_original_dirs(dataset_name)

vis_images = list_images(vis_dir, "rgb_image_*.jpg")
swir_images = list_images(swir_dir, "swir_image_*.jpg")

method_options = available_methods_for_dataset(dataset_name)
default_methods = [
    label
    for label in ["Patch MAGSAC++ VIS → SWIR", "OpenCV VIS → SWIR", "RASMD Aligned"]
    if label in method_options
]
if not default_methods:
    default_methods = list(method_options.keys())[:3]

selected_methods = st.sidebar.multiselect(
    "Methods",
    list(method_options.keys()),
    default=default_methods,
)

comparison_mode = st.sidebar.checkbox("Comparison mode", value=False)

st.header(f"Selected dataset: {dataset_name}")

if len(vis_images) == 0 or len(swir_images) == 0:
    st.error("No original VIS/SWIR images were found.")
    st.stop()

n_images = min(len(vis_images), len(swir_images))

st.success(f"Found {n_images} images, from image 0 to image {n_images - 1}.")

tabs_names = ["Selected dataset"] + selected_methods

if comparison_mode:
    tabs_names.append("Comparison")

tabs_names.append("4 seasons")
tabs_names.append("Documents")

tabs = st.tabs(tabs_names)

with tabs[0]:
    show_image_grid(
        vis_images,
        swir_images,
        title="Original unaligned images"
    )

    show_visible_sequence(vis_images)

for i, method_label in enumerate(selected_methods, start=1):
    with tabs[i]:
        method_folder = method_options[method_label]
        show_method_tab(
            dataset_name,
            method_label,
            method_folder,
            comparison_mode=comparison_mode
        )

if comparison_mode:
    with tabs[1 + len(selected_methods)]:
        show_comparison_tab(dataset_name, selected_methods)

with tabs[-2]:
    show_season_metrics_table()

with tabs[-1]:
    show_pdf_tab()

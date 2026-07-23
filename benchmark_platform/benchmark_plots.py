"""Composants de visualisation pour metriques et artefacts du benchmark.

Ces fonctions affichent des tableaux centres, boxplots, histogrammes, courbes
par frame et images deja generees dans l'interface Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None


METRIC_LABELS = {
    "goa": "GOA",
    "edge_f1_8px": "Edge Alignment 8 px",
    "mind_ssd": "MIND SSD",
    "reprojection_error": "Reprojection",
}


DISPLAY_COLUMN_LABELS = {
    "metric": "Metric",
    "method": "Method",
    "method_name": "Method",
    "dataset": "Dataset",
    "direction": "Direction",
    "rank": "Rank",
    "count": "Count",
    "mean": "Mean",
    "median": "Median",
    "std": "Std",
    "min": "Min",
    "max": "Max",
    "p25": "P25",
    "p75": "P75",
    "p90": "P90",
    "p95": "P95",
    "mean_pm_std": "Mean +/- Std",
    "difference_vs_best_percent": "Difference vs best (%)",
    "runtime_total_s": "Runtime total (s)",
    "training_pairs_used": "Training pairs",
    "hardware": "Hardware",
}


LOWER_IS_BETTER_METRICS = {"MIND SSD", "Reprojection"}


def _read_generated_csv(path: Path) -> pd.DataFrame:
    """Lit un CSV genere; un artefact vide ou incomplet est traite comme absent."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _sort_metric_table(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Ordena una tabla segun si la metrica se maximiza o minimiza."""
    if df.empty or "mean" not in df:
        return df
    table = df.copy()
    table["mean"] = pd.to_numeric(table["mean"], errors="coerce")
    ascending = metric_name in LOWER_IS_BETTER_METRICS
    return table.sort_values("mean", ascending=ascending, na_position="last").reset_index(drop=True)


def _sort_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Trie une table resume separement pour chaque metrique."""
    if df.empty or "metric" not in df or "mean" not in df:
        return df
    chunks = []
    for metric_name, group in df.groupby("metric", sort=False):
        chunks.append(_sort_metric_table(group, metric_name))
    return pd.concat(chunks, ignore_index=True) if chunks else df


def _sort_runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena la tabla de runtime de menor a mayor tiempo total."""
    if df.empty or "runtime_total_s" not in df:
        return df
    table = df.copy()
    table["runtime_total_s"] = pd.to_numeric(table["runtime_total_s"], errors="coerce")
    return table.sort_values("runtime_total_s", ascending=True, na_position="last").reset_index(drop=True)


def _display_table(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    """Renomme les colonnes techniques en labels lisibles pour la page."""
    table = df.copy()
    if drop_columns:
        table = table.drop(columns=[col for col in drop_columns if col in table], errors="ignore")
    return table.rename(columns={col: DISPLAY_COLUMN_LABELS.get(col, col.replace("_", " ").title()) for col in table.columns})


def render_centered_table(df: pd.DataFrame, max_height: int = 520) -> None:
    """Renderiza una tabla HTML centrada con scroll interno controlado."""
    table = df.copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        table[col] = table[col].round(4)
    table = table.astype(object).where(pd.notna(table), "")
    html = table.to_html(
        index=False,
        escape=True,
        classes="benchmark-centered-table",
        border=0,
    )
    st.markdown(
        dedent(f"""
<style>
.benchmark-table-wrap {{
    width: 100%;
    max-height: {max_height}px;
    overflow: auto;
    border: 1px solid #d7dce2;
    border-radius: 6px;
    margin: 0.35rem 0 1rem 0;
}}
table.benchmark-centered-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}}
table.benchmark-centered-table th,
table.benchmark-centered-table td {{
    text-align: center !important;
    vertical-align: middle !important;
    padding: 0.55rem 0.65rem;
    border: 1px solid #e1e5eb;
    white-space: nowrap;
}}
table.benchmark-centered-table th {{
    background: #f7f8fa;
    color: #697386;
    font-weight: 700;
}}
table.benchmark-centered-table td {{
    color: #111827;
}}
</style>
<div class="benchmark-table-wrap">{html}</div>
""").strip(),
        unsafe_allow_html=True,
    )


def _numeric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    """Retourne une serie numerique propre pour une metrique."""
    return pd.to_numeric(df[metric], errors="coerce").dropna()


def _smooth(values: np.ndarray, sigma: int = 5) -> np.ndarray:
    """Suaviza una curva con scipy si existe, o con rolling mean como fallback."""
    if gaussian_filter1d is not None and len(values) > 3:
        return gaussian_filter1d(values, sigma=sigma)
    return pd.Series(values).rolling(window=11, min_periods=1, center=True).mean().to_numpy()


def stats_table(values: pd.Series) -> None:
    """Affiche les statistiques de base d'une metrique dans une table compacte."""
    if values.empty:
        st.info("No numeric values are available for this metric.")
        return
    stats_html = f"""
    <table style="width:100%; border-collapse:collapse; text-align:center; font-size:18px; margin:0.35rem 0 0.9rem 0;">
        <tr style="border-bottom:1px solid #d7dce2;">
            <th style="padding:0.45rem;">Mean</th>
            <th style="padding:0.45rem;">Median</th>
            <th style="padding:0.45rem;">Std</th>
            <th style="padding:0.45rem;">Min</th>
            <th style="padding:0.45rem;">Max</th>
        </tr>
        <tr>
            <td style="padding:0.45rem;">{values.mean():.4f}</td>
            <td style="padding:0.45rem;">{values.median():.4f}</td>
            <td style="padding:0.45rem;">{values.std():.4f}</td>
            <td style="padding:0.45rem;">{values.min():.4f}</td>
            <td style="padding:0.45rem;">{values.max():.4f}</td>
        </tr>
    </table>
    """
    st.markdown(stats_html, unsafe_allow_html=True)


def single_metric_panel(df: pd.DataFrame, metric: str, title: str | None = None) -> None:
    """Renderiza boxplot, histograma y curva frame-by-frame de una metrica."""
    if metric not in df:
        st.info(f"Metric not found: {metric}.")
        return
    values = _numeric_series(df, metric)
    if values.empty:
        return

    title = title or METRIC_LABELS.get(metric, metric)
    st.markdown(f"### {title}")
    stats_table(values)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        ax.boxplot(values.to_numpy(), vert=True, patch_artist=True, boxprops={"facecolor": "#dbeafe"})
        ax.set_title("Boxplot")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        ax.hist(values.to_numpy(), bins=30, color="#2f6f9f", alpha=0.85)
        ax.set_title("Histogram")
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    raw = values.to_numpy(dtype=float)
    ax.plot(raw, alpha=0.25, color="#2f6f9f", label="Raw")
    ax.plot(_smooth(raw, sigma=3), linewidth=2.2, color="#b45309", label="sigma = 3")
    ax.set_title(f"Frame by frame - {title}")
    ax.set_xlabel("Frame")
    ax.set_ylabel(title)
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_single_method_quality(metrics: pd.DataFrame, method_path: str | Path | None = None) -> None:
    """Affiche GOA, Edge Alignment, MIND SSD et la carte MIND d'une methode."""
    st.markdown("## Quality")
    for metric in ["goa", "edge_f1_8px", "mind_ssd"]:
        if metric in metrics:
            single_metric_panel(metrics, metric)

    if method_path is None:
        return
    method_path = Path(method_path)
    mind_candidates = [
        method_path / "results" / "mind_map_mean_GLOBAL_SCALE.png",
        method_path / "results" / "mind_map_mean.png",
    ]
    for image_path in mind_candidates:
        if image_path.exists():
            st.markdown("### MIND map")
            st.image(str(image_path), width=650)
            break


def _panel_images(panels_dir: Path, prefix: str) -> list[Path]:
    """Liste les images de panneaux best/worst generees par reprojection_aligned."""
    return sorted(panels_dir.glob(f"{prefix}_*.png")) + sorted(panels_dir.glob(f"{prefix}_*.jpg"))


def render_single_method_reprojection(metrics: pd.DataFrame, method_path: str | Path | None = None) -> None:
    """Affiche la metrique de reprojection et les panneaux meilleurs/pires s'ils existent."""
    st.markdown("## Reprojection")
    if "reprojection_error" not in metrics:
        st.info("No reprojection metric was found for this method.")
        return
    single_metric_panel(metrics, "reprojection_error", title="Reprojection")

    if method_path is None:
        return
    reproj_dir = Path(method_path) / "results" / "reprojection_aligned"
    image_pairs = [
        (reproj_dir / "reprojection_aligned_boxplot.png", reproj_dir / "reprojection_aligned_histogram.png"),
        (reproj_dir / "reprojection_aligned_curve.png", None),
    ]
    for left, right in image_pairs:
        if not left.exists():
            continue
        cols = st.columns(2) if right and right.exists() else st.columns([1])
        cols[0].image(str(left), width="stretch")
        if right and right.exists():
            cols[1].image(str(right), width="stretch")

    panels_dir = reproj_dir / "reprojection_aligned_panels"
    if panels_dir.exists():
        best = _panel_images(panels_dir, "best")[:3]
        worst = _panel_images(panels_dir, "worst")[:3]
        if best:
            st.markdown("### 3 best")
            for image_path in best:
                st.image(str(image_path), width="stretch")
        if worst:
            st.markdown("### 3 worst")
            for image_path in worst:
                st.image(str(image_path), width="stretch")


def render_metric_comparison(df: pd.DataFrame, metric: str) -> None:
    """Affiche les comparaisons inter-methodes pour une metrique donnee."""
    if metric not in df:
        return
    title = METRIC_LABELS.get(metric, metric)
    st.markdown(f"## {title}")
    metric_boxplots(df, [metric])
    metric_histograms(df, [metric])
    frame_curves(df, [metric])


def render_existing_comparison_panels(dataset: str) -> bool:
    """Affiche les panneaux preexistants gardes dans le dataset."""
    comparison_dir = (
        Path(__file__).resolve().parents[1]
        / "datas"
        / f"{dataset}_test"
        / "Sortie_Extraite"
        / "paired_images"
        / "benchmark_comparison_methods"
    )
    if not comparison_dir.exists():
        return False

    shown = False
    panels = {
        "GOA": ["goa_boxplot.png", "goa_histogram.png", "goa_frame_by_frame.png"],
        "Edge Alignment 8 px": [
            "edge_f1_8px_boxplot.png",
            "edge_f1_8px_histogram.png",
            "edge_f1_8px_frame_by_frame.png",
        ],
        "MIND SSD": ["mind_boxplot.png", "mind_histogram.png", "mind_frame_by_frame.png", "mind_global_comparison_panel.png"],
        "Reprojection": [
            "reprojection_boxplot.png",
            "reprojection_histogram.png",
            "reprojection_frame_by_frame_sigma3.png",
        ],
    }
    for title, names in panels.items():
        existing = [comparison_dir / name for name in names if (comparison_dir / name).exists()]
        if not existing:
            continue
        shown = True
        st.markdown(f"## {title}")
        for i in range(0, len(existing), 2):
            cols = st.columns(min(2, len(existing) - i))
            for col, image_path in zip(cols, existing[i : i + 2]):
                col.image(str(image_path), width="stretch")

    for table_name in ["ranking_methods.csv", "summary_statistics_methods.csv", "runtime_methods.csv"]:
        table_path = comparison_dir / table_name
        if table_path.exists():
            table_label = {
                "ranking_methods.csv": "Ranking table",
                "summary_statistics_methods.csv": "Summary statistics table",
                "runtime_methods.csv": "Runtime table",
            }.get(table_name, "Results table")
            st.markdown(f"### {table_label}")
            render_centered_table(_read_generated_csv(table_path))
            shown = True
    return shown


def render_comparison_artifact_dir(comparison_dir: str | Path) -> bool:
    """Affiche tous les artefacts generes pour une comparaison de methodes."""
    comparison_dir = Path(comparison_dir)
    if not comparison_dir.exists():
        return False

    shown = False
    summary_path = comparison_dir / "summary_statistics_methods.csv"
    summary_df = _read_generated_csv(summary_path)

    panels = [
        ("GOA", ["GOA"], [["goa_boxplot.png", "goa_histogram.png"], ["goa_frame_by_frame.png"]]),
        (
            "Edge Alignment 8 px",
            ["Edge Alignment 8 px", "Edge Alignment"],
            [["edge_f1_8px_boxplot.png", "edge_f1_8px_histogram.png"], ["edge_f1_8px_frame_by_frame.png"]],
        ),
        (
            "MIND SSD",
            ["MIND SSD"],
            [
                ["mind_boxplot.png", "mind_histogram.png"],
                ["mind_frame_by_frame.png"],
                ["mind_global_comparison_panel.png"],
            ],
        ),
        (
            "Reprojection",
            ["Reprojection"],
            [
                ["reprojection_boxplot.png", "reprojection_histogram.png"],
                ["reprojection_frame_by_frame_sigma3.png"],
            ],
        ),
    ]
    for title, metric_names, image_rows in panels:
        metric_df = (
            summary_df[summary_df["metric"].isin(metric_names)].copy()
            if not summary_df.empty and "metric" in summary_df
            else pd.DataFrame()
        )
        existing_rows = [
            [comparison_dir / name for name in row if (comparison_dir / name).exists()]
            for row in image_rows
        ]
        existing_rows = [row for row in existing_rows if row]
        if metric_df.empty and not existing_rows:
            continue
        shown = True
        st.markdown(f"## {title}")
        if not metric_df.empty:
            st.markdown(f"### {title} table")
            metric_df = _sort_metric_table(metric_df, metric_names[0])
            render_centered_table(_display_table(metric_df, drop_columns=["direction", "count"]))
        for row in existing_rows:
            cols = st.columns(len(row))
            for col, image_path in zip(cols, row):
                col.image(str(image_path), width="stretch")

    table_titles = {
        "runtime_methods.csv": "Runtime table",
        "ranking_methods.csv": "Ranking table",
        "summary_statistics_methods.csv": "Summary statistics table",
    }
    table_drop_columns = {
        "runtime_methods.csv": ["dataset", "direction"],
        "ranking_methods.csv": ["direction"],
        "summary_statistics_methods.csv": ["direction"],
    }
    for table_name, table_title in table_titles.items():
        table_path = comparison_dir / table_name
        if table_path.exists():
            df = _read_generated_csv(table_path)
            if not df.empty:
                title = table_title
                drop_columns = table_drop_columns.get(table_name)
                if table_name == "ranking_methods.csv" and df.get("method", pd.Series(dtype=object)).nunique() <= 1:
                    title = "Metric table"
                    drop_columns = ["direction", "rank"]
                elif table_name == "runtime_methods.csv":
                    df = _sort_runtime_table(df)
                elif table_name == "summary_statistics_methods.csv":
                    df = _sort_summary_table(df)
                st.markdown(f"### {title}")
                render_centered_table(_display_table(df, drop_columns=drop_columns))
                shown = True
    return shown


def render_single_method_artifact_dir(comparison_dir: str | Path) -> bool:
    """Affiche les artefacts quand la comparaison contient une seule methode."""
    comparison_dir = Path(comparison_dir)
    if not comparison_dir.exists():
        return False

    shown = False
    summary_path = comparison_dir / "summary_statistics_methods.csv"
    summary_df = _read_generated_csv(summary_path)

    panels = [
        ("GOA", ["GOA"], [["goa_boxplot.png", "goa_histogram.png"], ["goa_frame_by_frame.png"]]),
        (
            "Edge Alignment 8 px",
            ["Edge Alignment 8 px", "Edge Alignment"],
            [["edge_f1_8px_boxplot.png", "edge_f1_8px_histogram.png"], ["edge_f1_8px_frame_by_frame.png"]],
        ),
        (
            "MIND SSD",
            ["MIND SSD"],
            [
                ["mind_boxplot.png", "mind_histogram.png"],
                ["mind_frame_by_frame.png"],
                ["mind_global_comparison_panel.png"],
            ],
        ),
        (
            "Reprojection",
            ["Reprojection"],
            [
                ["reprojection_boxplot.png", "reprojection_histogram.png"],
                ["reprojection_frame_by_frame_sigma3.png"],
            ],
        ),
    ]
    for title, metric_names, image_rows in panels:
        metric_df = (
            summary_df[summary_df["metric"].isin(metric_names)].copy()
            if not summary_df.empty and "metric" in summary_df
            else pd.DataFrame()
        )
        existing_rows = [
            [comparison_dir / name for name in row if (comparison_dir / name).exists()]
            for row in image_rows
        ]
        existing_rows = [row for row in existing_rows if row]
        if metric_df.empty and not existing_rows:
            continue
        shown = True
        st.markdown(f"## {title}")
        if not metric_df.empty:
            st.markdown(f"### {title} table")
            metric_df = _sort_metric_table(metric_df, metric_names[0])
            render_centered_table(_display_table(metric_df, drop_columns=["direction", "count"]))
        for row in existing_rows:
            cols = st.columns(len(row))
            for col, image_path in zip(cols, row):
                col.image(str(image_path), width="stretch")

    runtime_path = comparison_dir / "runtime_methods.csv"
    if runtime_path.exists():
        runtime_df = pd.read_csv(runtime_path)
        runtime_df = _sort_runtime_table(runtime_df)
        value_cols = [
            col
            for col in ["runtime_total_s", "training_pairs_used", "hardware"]
            if col in runtime_df
        ]
        has_runtime_values = any(runtime_df[col].notna().any() for col in value_cols)
        if not runtime_df.empty and has_runtime_values:
            st.markdown("### Runtime table")
            render_centered_table(_display_table(runtime_df, drop_columns=["dataset", "direction"]))
            shown = True

    if not summary_df.empty:
        st.markdown("### Summary statistics table")
        render_centered_table(_display_table(_sort_summary_table(summary_df), drop_columns=["direction", "count"]))
        shown = True
    return shown


def metric_histograms(df: pd.DataFrame, metrics: list[str]) -> None:
    """Dessine des histogrammes par methode pour une liste de metriques."""
    for metric in metrics:
        if metric in df:
            fig, ax = plt.subplots(figsize=(8, 4))
            for method_name, group in df.groupby("method_name"):
                values = _numeric_series(group, metric)
                if not values.empty:
                    ax.hist(values, bins=30, alpha=0.55, label=method_name)
            ax.set_title(metric)
            ax.set_xlabel(metric)
            ax.set_ylabel("count")
            ax.legend(loc="best")
            st.pyplot(fig, width="stretch")
            plt.close(fig)


def metric_boxplots(df: pd.DataFrame, metrics: list[str]) -> None:
    """Dessine des boxplots par methode pour une liste de metriques."""
    for metric in metrics:
        if metric in df:
            labels = []
            values = []
            for method_name, group in df.groupby("method_name"):
                series = _numeric_series(group, metric)
                if not series.empty:
                    labels.append(method_name)
                    values.append(series.to_numpy())
            if not values:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(values, labels=labels, showfliers=True)
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig, width="stretch")
            plt.close(fig)


def frame_curves(df: pd.DataFrame, metrics: list[str]) -> None:
    """Dessine les courbes de metrique par frame pour chaque methode."""
    x_col = "frame_id" if "frame_id" in df else "pair_id"
    for metric in metrics:
        if metric in df:
            fig, ax = plt.subplots(figsize=(9, 4))
            for method_name, group in df.groupby("method_name"):
                plot_df = group[[x_col, metric]].copy()
                plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
                plot_df = plot_df.dropna(subset=[metric])
                if not plot_df.empty:
                    ax.plot(plot_df[x_col], plot_df[metric], label=method_name, linewidth=1.5)
            ax.set_title(metric)
            ax.set_xlabel(x_col)
            ax.set_ylabel(metric)
            ax.legend(loc="best")
            st.pyplot(fig, width="stretch")
            plt.close(fig)


def mean_bars(summary: pd.DataFrame, metrics: list[str]) -> None:
    """Dessine des barres de moyennes par methode a partir d'une table resume."""
    for metric in metrics:
        if metric in summary:
            plot_df = summary[["method_name", metric]].copy()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna(subset=[metric])
            if plot_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(plot_df["method_name"], plot_df[metric])
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

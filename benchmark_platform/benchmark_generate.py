"""Generation d'artefacts comparatifs persistants.

Cree des graphiques, tableaux CSV, rankings et panneaux MIND pour la page
Compare Methods. Les fichiers sont gardes dans benchmark_results/comparisons/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import cv2

from benchmark_compare import filter_bundles_to_common_frame_ids, load_method_bundle
from benchmark_config import RESULTS_ROOT
from benchmark_plots import _smooth


METRICS = {
    "goa": {
        "title": "GOA",
        "ylabel": "GOA",
        "direction": "higher is better",
        "frame_name": "goa_frame_by_frame.png",
        "box_name": "goa_boxplot.png",
        "hist_name": "goa_histogram.png",
    },
    "mind_ssd": {
        "title": "MIND SSD",
        "ylabel": "MIND SSD",
        "direction": "lower is better",
        "frame_name": "mind_frame_by_frame.png",
        "box_name": "mind_boxplot.png",
        "hist_name": "mind_histogram.png",
    },
    "edge_f1_8px": {
        "title": "Edge Alignment 8 px",
        "ylabel": "Edge Alignment 8 px",
        "direction": "higher is better",
        "frame_name": "edge_f1_8px_frame_by_frame.png",
        "box_name": "edge_f1_8px_boxplot.png",
        "hist_name": "edge_f1_8px_histogram.png",
    },
    "reprojection_error": {
        "title": "Reprojection",
        "ylabel": "Mean Reprojection Error (px)",
        "direction": "lower is better",
        "frame_name": "reprojection_frame_by_frame_sigma3.png",
        "box_name": "reprojection_boxplot.png",
        "hist_name": "reprojection_histogram.png",
    },
}


def _safe_label(text: str) -> str:
    """Convertit un nom de methode en fragment securise pour les noms de fichiers."""
    keep = []
    for char in text:
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).lower().split("_"))[:80]


def comparison_output_dir(method_paths: list[str | Path]) -> Path:
    """Cree une route deterministe pour une combinaison de methodes comparees."""
    resolved = [str(Path(p).resolve()) for p in method_paths]
    digest = hashlib.sha1("|".join(sorted(resolved)).encode("utf-8")).hexdigest()[:12]
    return RESULTS_ROOT / "comparisons" / digest


def _comparison_uses_multiple_datasets(bundles: list[dict]) -> bool:
    """Return whether dataset names are needed to disambiguate labels."""
    datasets = {
        str(bundle.get("metadata", {}).get("dataset", "")).strip()
        for bundle in bundles
        if str(bundle.get("metadata", {}).get("dataset", "")).strip()
    }
    return len(datasets) > 1


def _method_label(bundle: dict, include_dataset: bool = True) -> str:
    """Build a compact method label, always preserving the alignment direction."""
    metadata = bundle["metadata"]
    label = metadata.get("method_name", Path(bundle["path"]).name)
    direction = str(metadata.get("direction", "")).replace("_to_", " → ")
    dataset = metadata.get("dataset", "")
    details = [str(dataset), direction] if include_dataset else [direction]
    details = [detail for detail in details if detail]
    return f"{label} ({', '.join(details)})" if details else str(label)


def _series_by_method(bundles: list[dict], metric: str) -> dict[str, pd.Series]:
    """Extrait une serie numerique par methode pour une metrique donnee."""
    series = {}
    include_dataset = _comparison_uses_multiple_datasets(bundles)
    for bundle in bundles:
        metrics = bundle["metrics"]
        if metric not in metrics:
            continue
        values = pd.to_numeric(metrics[metric], errors="coerce").dropna()
        if values.empty:
            continue
        series[_method_label(bundle, include_dataset=include_dataset)] = values.reset_index(drop=True)
    return series


def summarize_values(values: pd.Series) -> dict:
    """Calcule les statistiques descriptives utilisees dans les rapports comparatifs."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "mean_pm_std": "",
        }
    mean = float(values.mean())
    std = float(values.std())
    return {
        "count": int(len(values)),
        "mean": mean,
        "median": float(values.median()),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "mean_pm_std": f"{mean:.4f} +/- {std:.4f}",
    }


def plot_frame_by_frame(title: str, method_series: dict[str, pd.Series], ylabel: str, out_path: Path, sigma: int = 3) -> None:
    """Sauvegarde une courbe frame-by-frame lissee pour chaque methode."""
    fig, ax = plt.subplots(figsize=(13, 5))
    plotted = False
    for method_name, values in method_series.items():
        values = pd.to_numeric(values, errors="coerce")
        if values.dropna().empty:
            continue
        filled = values.interpolate().bfill().ffill().to_numpy(dtype=float)
        x = np.arange(len(filled))
        ax.plot(x, filled, alpha=0.18, linewidth=0.8)
        ax.plot(x, _smooth(filled, sigma=sigma), linewidth=2.3, label=method_name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Official frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - Frame by Frame")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_boxplot(title: str, method_series: dict[str, pd.Series], ylabel: str, out_path: Path) -> None:
    """Sauvegarde un boxplot pour comparer les distributions d'une metrique."""
    labels = []
    data = []
    for method_name, values in method_series.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            continue
        labels.append(method_name)
        data.append(values)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(data) * 2.4), 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - Boxplot")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_histogram(title: str, method_series: dict[str, pd.Series], xlabel: str, out_path: Path) -> None:
    """Sauvegarde des histogrammes superposes pour comparer la dispersion par methode."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for method_name, values in method_series.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            continue
        ax.hist(values, bins=30, alpha=0.45, label=method_name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title} - Histogram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def build_ranking_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Cree une table de ranking par metrique en utilisant le bon sens."""
    rows = []
    for metric_name, group in summary_df.groupby("metric", sort=False):
        direction = group.iloc[0]["direction"]
        ascending = "lower is better" in direction
        ranked = group.sort_values("mean", ascending=ascending, na_position="last")
        if ranked.empty:
            continue
        best = ranked.iloc[0]["mean"]
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            if pd.isna(row["mean"]) or pd.isna(best):
                percent = np.nan
            elif ascending:
                percent = 100 * (row["mean"] - best) / best if best else np.nan
            else:
                percent = 100 * (best - row["mean"]) / abs(best) if best else np.nan
            rows.append({
                "metric": metric_name,
                "rank": rank,
                "method": row["method"],
                "direction": direction,
                "mean_pm_std": row["mean_pm_std"],
                "mean": row["mean"],
                "median": row["median"],
                "std": row["std"],
                "p95": row["p95"],
                "difference_vs_best_percent": percent,
            })
    return pd.DataFrame(rows)


def sort_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Trie chaque metrique du meilleur au moins bon dans la table resume."""
    if summary_df.empty or "metric" not in summary_df or "mean" not in summary_df:
        return summary_df
    chunks = []
    for metric_name, group in summary_df.groupby("metric", sort=False):
        table = group.copy()
        table["mean"] = pd.to_numeric(table["mean"], errors="coerce")
        ascending = "lower is better" in str(table.iloc[0].get("direction", ""))
        chunks.append(table.sort_values("mean", ascending=ascending, na_position="last"))
    return pd.concat(chunks, ignore_index=True) if chunks else summary_df


def _load_mind_raw_map(method_path: Path) -> np.ndarray | None:
    """Charge la carte MIND moyenne brute si la methode l'a produite."""
    raw_path = method_path / "results" / "mind_map_mean_raw.npy"
    if raw_path.exists():
        try:
            return np.load(raw_path)
        except Exception:
            return None
    return None


def generate_mind_global_comparison_panel(bundles: list[dict], out_dir: Path) -> None:
    """Genere un panneau MIND avec la meme echelle de couleur pour toutes les methodes."""
    mind_maps: list[tuple[str, np.ndarray | None]] = []
    include_dataset = _comparison_uses_multiple_datasets(bundles)
    for bundle in bundles:
        method_path = Path(bundle["path"])
        mean_map = _load_mind_raw_map(method_path)
        mind_maps.append((_method_label(bundle, include_dataset=include_dataset), mean_map))

    if not mind_maps:
        return

    values = [m[~np.isnan(m)].ravel() for _, m in mind_maps if m is not None]
    values = [v for v in values if len(v) > 0]
    if values:
        all_values = np.concatenate(values)
        vmin = np.percentile(all_values, 2)
        vmax = np.percentile(all_values, 98)
    else:
        vmin, vmax = 0.0, 1.0

    cols = min(3, len(mind_maps))
    rows = int(np.ceil(len(mind_maps) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.ravel():
        ax.axis("off")

    image = None
    missing_methods = []
    for ax, (method_name, mean_map) in zip(axes.ravel(), mind_maps):
        if mean_map is None:
            missing_methods.append(method_name)
            ax.text(
                0.5,
                0.5,
                "MIND map\nunavailable",
                ha="center",
                va="center",
                fontsize=12,
                color="#666666",
                transform=ax.transAxes,
            )
            ax.set_title(method_name, fontsize=10)
            ax.axis("off")
            continue
        image = ax.imshow(mean_map, cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_title(method_name, fontsize=10)
        ax.axis("off")

        norm = np.clip((mean_map - vmin) / (vmax - vmin + 1e-8), 0, 1)
        colored = cv2.applyColorMap(
            (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        colored = np.where(np.isnan(mean_map)[..., None], 0, colored).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"mind_map_global_{_safe_label(method_name)}.png"), colored)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="MIND SSD")
    fig.suptitle("MIND map comparison - Global scale", fontsize=14)
    fig.savefig(out_dir / "mind_global_comparison_panel.png", dpi=240)
    plt.close(fig)

    (out_dir / "mind_global_scale.txt").write_text(
        (
            f"global_vmin={vmin:.8f}\n"
            f"global_vmax={vmax:.8f}\n"
            f"missing_methods={'; '.join(missing_methods)}\n"
        ),
        encoding="utf-8",
    )


def generate_comparison_artifacts(method_paths: list[str | Path], diagnostic: bool = False) -> Path:
    """Genere tous les CSV et images consommes par la page Compare Methods."""
    if not method_paths:
        raise ValueError("Select at least one method.")

    out_dir = comparison_output_dir(method_paths)
    if diagnostic:
        out_dir = out_dir.with_name(f"{out_dir.name}_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = [load_method_bundle(path, diagnostic=diagnostic) for path in method_paths]
    bundles = filter_bundles_to_common_frame_ids(bundles)
    # On garde les frame IDs communs pour expliquer exactement ce qui a ete compare.
    common_frames = []
    for bundle in bundles:
        metrics = bundle["metrics"]
        if "frame_id" in metrics and not metrics.empty:
            common_frames.append(set(pd.to_numeric(metrics["frame_id"], errors="coerce").dropna().astype(int)))
    common_frame_ids = sorted(set.intersection(*common_frames)) if len(common_frames) == len(bundles) and common_frames else []
    (out_dir / "common_frame_ids.txt").write_text(
        "\n".join(str(frame_id) for frame_id in common_frame_ids),
        encoding="utf-8",
    )

    summary_rows = []
    for metric, info in METRICS.items():
        method_series = _series_by_method(bundles, metric)
        if not method_series:
            continue

        for method_name, values in method_series.items():
            summary_rows.append({
                "metric": info["title"],
                "method": method_name,
                "direction": info["direction"],
                **summarize_values(values),
            })

        plot_frame_by_frame(info["title"], method_series, info["ylabel"], out_dir / info["frame_name"])
        plot_boxplot(info["title"], method_series, info["ylabel"], out_dir / info["box_name"])
        plot_histogram(info["title"], method_series, info["ylabel"], out_dir / info["hist_name"])

    summary_df = sort_summary_table(pd.DataFrame(summary_rows))
    summary_df.to_csv(out_dir / "summary_statistics_methods.csv", index=False)
    ranking_df = build_ranking_table(summary_df) if not summary_df.empty else pd.DataFrame()
    ranking_df.to_csv(out_dir / "ranking_methods.csv", index=False)

    runtime_rows = []
    include_dataset = _comparison_uses_multiple_datasets(bundles)
    for bundle in bundles:
        runtime = bundle.get("runtime", {})
        metadata = bundle.get("metadata", {})
        runtime_rows.append({
            "method": _method_label(bundle, include_dataset=include_dataset),
            "dataset": metadata.get("dataset"),
            "direction": metadata.get("direction"),
            "runtime_total_s": runtime.get("runtime_seconds", runtime.get("runtime_total_s", np.nan)),
            "training_pairs_used": metadata.get(
                "estimation_pairs_used",
                metadata.get("training_pairs_used", np.nan),
            ),
            "hardware": runtime.get("hardware", runtime.get("device", metadata.get("hardware", ""))),
        })
    pd.DataFrame(runtime_rows).to_csv(out_dir / "runtime_methods.csv", index=False)
    generate_mind_global_comparison_panel(bundles, out_dir)

    return out_dir

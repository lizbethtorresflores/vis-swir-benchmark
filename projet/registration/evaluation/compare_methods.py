import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
import cv2

PROJECT_ROOT = Path("/home/lizbeth/Desktop/intership/recallage-multimodal-zz3")


# ============================================================
# PATHS
# ============================================================

def find_method_dir(dataset: str, method_keyword: str) -> Path:
    paired_dir = PROJECT_ROOT / "datas" / dataset / "Sortie_Extraite" / "paired_images"

    if not paired_dir.exists():
        raise FileNotFoundError(f"Le dossier paired_images n'existe pas : {paired_dir}")

    candidates = [
        p for p in paired_dir.iterdir()
        if p.is_dir()
        and method_keyword.lower() in p.name.lower()
        and p.name.upper() not in ["VISIBLE", "SWIR"]
        and not p.name.lower().startswith("benchmark")
    ]

    if len(candidates) == 0:
        raise FileNotFoundError(f"Méthode '{method_keyword}' introuvable dans {paired_dir}")

    if len(candidates) > 1:
        print(f"⚠️ Plusieurs dossiers pour '{method_keyword}' :")
        for c in candidates:
            print(" -", c.name)
        print("Utilisation de :", candidates[0].name)

    return candidates[0]

def load_runtime_metrics(path: Path) -> dict:
    original_path = path

    # Si le runtime n'existe pas dans le dossier cropped, chercher dans le dossier original.
    if not path.exists() and "_cropped_like_rasmd" in str(path):
        fallback_path = Path(str(path).replace("_cropped_like_rasmd", ""))

        if fallback_path.exists():
            print(f"[INFO] Runtime no trouvé dans cropped, utilisé depuis original: {fallback_path}")
            path = fallback_path

    if not path.exists():
        print(f"[WARNING] Runtime introuvable : {original_path}")
        return {
            "runtime_total_s": np.nan,
            "fps_global": np.nan,
        }

    runtime_total = np.nan
    fps_global = np.nan

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Temps total"):
                runtime_total = float(line.split(":")[1].replace("s", "").strip())

            elif line.startswith("FPS global"):
                fps_global = float(line.split(":")[1].strip())

    return {
        "runtime_total_s": runtime_total,
        "fps_global": fps_global,
    }

def build_metric_files(method_dir: Path) -> dict:
    results = method_dir / "results"

    # ---------------------------------------------------------
    # Reprojection : prend en charge l'ancien benchmark et le nouveau mode aligned.
    # ---------------------------------------------------------

    reprojection_candidates = [
        results / "reprojection_aligned" / "reprojection_aligned_framewise.csv",
        results / "reprojection_bench" / "reprojection_framewise.csv",
    ]

    reprojection_file = None

    for p in reprojection_candidates:
        if p.exists():
            reprojection_file = p
            break

    return {
        "quality": results / "scores_qualite_recalage.csv",
        "reprojection": reprojection_file,
        "runtime": method_dir / "runtime_metrics.txt",
    }


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )

def seconds_to_hms(sec):
    if pd.isna(sec):
        return "NA"

    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"
# ============================================================
# CSV UTILS
# ============================================================

def load_csv(path: Path, name: str):
    if not path.exists():
        print(f"[WARNING] {name} introuvable : {path}")
        return None

    df = pd.read_csv(path)
    print(f"[OK] {name}: {path}")
    return df


def find_column(df, possible_names):
    if df is None:
        return None

    cols_lower = {c.lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]

    for col in df.columns:
        col_l = col.lower()
        for name in possible_names:
            if name.lower() in col_l:
                return col

    return None


def summarize_metric(values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if len(values) == 0:
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
            "p99": np.nan,
            "iqr": np.nan,
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
        "p99": float(values.quantile(0.99)),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "mean_pm_std": f"{mean:.4f} ± {std:.4f}",
    }


# ============================================================
# PLOTS MULTI-METHOD
# ============================================================

def plot_frame_by_frame(
    metric_name,
    method_series,
    ylabel,
    out_path,
    sigma=10
):
    plt.figure(figsize=(13, 5))
    plotted = False

    for method_name, values in method_series.items():

        values = pd.to_numeric(values, errors="coerce")

        if values.dropna().empty:
            continue

        # Remplir les NaN pour que gaussian_filter fonctionne.
        values_filled = (
            values.interpolate()
                  .bfill()
                  .ffill()
                  .to_numpy()
        )

        # Lissage gaussien.
        smooth = gaussian_filter1d(
            values_filled,
            sigma=sigma
        )

        x = np.arange(len(values))

        # signal original discret
        plt.plot(
            x,
            values_filled,
            alpha=0.20,
            linewidth=0.8
        )

        # signal lissé
        plt.plot(
            x,
            smooth,
            linewidth=2.5,
            label=method_name
        )

        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} - Frame by Frame (Gaussian smoothing)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.gca().xaxis.set_major_locator(
        ticker.MultipleLocator(50)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Graphique enregistré : {out_path}")


def plot_boxplot(metric_name, method_series, ylabel, out_path):
    data = []
    labels = []

    for method_name, values in method_series.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) > 0:
            data.append(values)
            labels.append(method_name)

    if len(data) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} - Boxplot")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Boxplot enregistré : {out_path}")


def plot_histogram(metric_name, method_series, xlabel, out_path):
    plt.figure(figsize=(9, 5))
    plotted = False

    for method_name, values in method_series.items():
        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) > 0:
            plt.hist(values, bins=30, alpha=0.45, label=method_name)
            plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(f"{metric_name} - Histogram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Histogramme enregistré : {out_path}")


# ============================================================
# COMPARISON TABLE
# ============================================================

def compute_winner(group_df, direction: str):
    valid = group_df.dropna(subset=["mean"])

    if len(valid) == 0:
        return None

    if "lower is better" in direction:
        winner_row = valid.loc[valid["mean"].idxmin()]
    elif "higher is better" in direction:
        winner_row = valid.loc[valid["mean"].idxmax()]
    else:
        return None

    return winner_row["method"]


def build_ranking_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric_name in summary_df["metric"].unique():
        sub = summary_df[summary_df["metric"] == metric_name].copy()

        if len(sub) == 0:
            continue

        direction = sub.iloc[0]["direction"]

        if "lower is better" in direction:
            sub = sub.sort_values("mean", ascending=True)
        elif "higher is better" in direction:
            sub = sub.sort_values("mean", ascending=False)
        else:
            sub = sub.sort_values("mean", ascending=False)

        best_mean = sub.iloc[0]["mean"]

        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            if pd.isna(row["mean"]) or pd.isna(best_mean):
                diff_vs_best = np.nan
                percent_vs_best = np.nan
            else:
                if "lower is better" in direction:
                    diff_vs_best = row["mean"] - best_mean
                    percent_vs_best = 100 * (row["mean"] - best_mean) / best_mean if best_mean != 0 else np.nan
                elif "higher is better" in direction:
                    diff_vs_best = best_mean - row["mean"]
                    percent_vs_best = 100 * (best_mean - row["mean"]) / abs(best_mean) if best_mean != 0 else np.nan
                else:
                    diff_vs_best = np.nan
                    percent_vs_best = np.nan

            rows.append({
                "metric": metric_name,
                "direction": direction,
                "rank": rank,
                "method": row["method"],
                "mean": row["mean"],
                "mean_pm_std": row.get("mean_pm_std", ""),
                "median": row["median"],
                "std": row["std"],
                "p95": row["p95"],
                "diff_vs_best_mean": diff_vs_best,
                "percent_vs_best_mean": percent_vs_best,
            })

    return pd.DataFrame(rows)

def load_mind_raw_map(method_dir: Path):
    raw_path = method_dir / "results" / "mind_map_mean_raw.npy"
    if raw_path.exists():
        return np.load(raw_path)
    print(f"[WARNING] Carte MIND brute introuvable : {raw_path}")
    return None


def save_mind_global_png(mean_map, out_path, vmin, vmax):
    norm = np.clip((mean_map - vmin) / (vmax - vmin + 1e-8), 0, 1)

    colored = cv2.applyColorMap(
        (np.nan_to_num(norm, nan=0.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    colored = np.where(
        np.isnan(mean_map)[..., None],
        0,
        colored
    ).astype(np.uint8)

    cv2.imwrite(str(out_path), colored)

def generate_mind_comparison_panel(method_dirs: dict, out_dir: Path):
    mind_maps = {}

    for method_name, method_dir in method_dirs.items():
        mean_map = load_mind_raw_map(method_dir)
        if mean_map is not None:
            mind_maps[method_name] = mean_map

    if len(mind_maps) == 0:
        print("[WARNING] Aucune carte MIND disponible pour le panneau comparatif")
        return

    all_values = np.concatenate([
        m[~np.isnan(m)].ravel()
        for m in mind_maps.values()
    ])

    global_vmin = np.percentile(all_values, 2)
    global_vmax = np.percentile(all_values, 98)

    n = len(mind_maps)
    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, (method_name, mean_map) in enumerate(mind_maps.items(), start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(mean_map, cmap="jet", vmin=global_vmin, vmax=global_vmax)
        plt.title(method_name)
        plt.axis("off")

    cbar = plt.colorbar(fraction=0.025, pad=0.02)
    cbar.set_label("MIND-SSD")

    plt.suptitle("MIND map comparison - Global scale", fontsize=14)
    plt.tight_layout()
    out_path = out_dir / "mind_global_comparison_panel.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Panneau MIND enregistré : {out_path}")

def generate_global_mind_maps(method_dirs: dict, out_dir: Path):
    mind_maps = {}

    for method_name, method_dir in method_dirs.items():
        mean_map = load_mind_raw_map(method_dir)
        if mean_map is not None:
            mind_maps[method_name] = (method_dir, mean_map)

    if len(mind_maps) < 2:
        print("[WARNING] Pas assez de cartes MIND brutes pour l'échelle globale")
        return

    all_values = np.concatenate([
        m[~np.isnan(m)].ravel()
        for _, m in mind_maps.values()
    ])

    global_vmin = np.percentile(all_values, 2)
    global_vmax = np.percentile(all_values, 98)

    scale_txt = out_dir / "mind_global_scale.txt"
    with open(scale_txt, "w") as f:
        f.write(f"global_vmin={global_vmin:.8f}\n")
        f.write(f"global_vmax={global_vmax:.8f}\n")

    for method_name, (method_dir, mean_map) in mind_maps.items():
        safe_method = safe_name(method_name)

        out_method = method_dir / "results" / "mind_map_mean_GLOBAL_SCALE.png"
        save_mind_global_png(mean_map, out_method, global_vmin, global_vmax)

        out_comparison = out_dir / f"mind_map_global_{safe_method}.png"
        save_mind_global_png(mean_map, out_comparison, global_vmin, global_vmax)

        print(f"[OK] MIND global scale: {out_method}")

    print(f"[OK] Échelle globale MIND enregistrée : {scale_txt}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare plusieurs méthodes VIS-SWIR sur GOA, Edge Alignment, MIND-SSD, Reprojection, Runtime et FPS"
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", required=True)

    args = parser.parse_args()

    dataset = args.dataset
    method_keywords = args.methods

    paired_dir = PROJECT_ROOT / "datas" / dataset / "Sortie_Extraite" / "paired_images"
    out_dir = paired_dir / "benchmark_comparison_methods"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("COMPARAISON MULTI-MÉTHODES")
    print("=" * 80)
    print(f"Dataset : {dataset}")
    print(f"Output  : {out_dir}")

    # ------------------------------------------------------------
    # Détecter les méthodes
    # ------------------------------------------------------------

    method_dirs = {}
    method_labels = {}

    for keyword in method_keywords:
        method_dir = find_method_dir(dataset, keyword)

        # ============================================================
        # LIBELLÉS COURTS POUR LES GRAPHIQUES
        # ============================================================

        def beautify_method_name(name: str) -> str:
            n = name.lower()

            direction = ""
            if "swir_to_vis" in n:
                direction = " SWIR→VIS"
            elif "vis_to_swir" in n:
                direction = " VIS→SWIR"

            if "opencv" in n:
                return "OpenCV" + direction

            if "patch_ransac" in n:
                return "Patch MAGSAC++" + direction

            if "rasmd" in n:
                return "RASMD aligned"

            return name


        # ------------------------------------------------------------

        label = beautify_method_name(method_dir.name)

        method_dirs[label] = method_dir
        method_labels[keyword] = label

        print(f"[METHOD] {keyword} -> {method_dir}")

    # ------------------------------------------------------------
    # Charger les CSV
    # ------------------------------------------------------------

    method_data = {}

    for method_name, method_dir in method_dirs.items():
        files = build_metric_files(method_dir)

        method_data[method_name] = {
        "quality": load_csv(files["quality"], f"{method_name} Quality GOA/Edge Alignment/MIND"),

        "reprojection": (
            load_csv(files["reprojection"], f"{method_name} Reprojection")
            if files["reprojection"] is not None
            else None
        ),
        "runtime": load_runtime_metrics(files["runtime"]),
    }

    # ------------------------------------------------------------
    # Definición de métricas
    # ------------------------------------------------------------

    metrics = {
        "GOA": {
            "source": "quality",
            "cols": ["goa", "GOA"],
            "direction": "higher is better",
            "ylabel": "GOA",
        },
        "Edge Alignment": {
            "source": "quality",
            "cols": ["edge_f1_8px", "Edge Alignment"],
            "direction": "higher is better",
            "ylabel": "Edge Alignment",
        },
        "MIND": {
            "source": "quality",
            "cols": ["mind_ssd", "mind", "MIND"],
            "direction": "lower is better",
            "ylabel": "MIND-SSD",
        },

        "Reprojection": {
            "source": "reprojection",
            "cols": ["mean_error"],
            "direction": "lower is better",
            "ylabel": "Mean Reprojection Error (px)",
        },


    }

    # ------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------

    summary_rows = []

    for metric_name, info in metrics.items():
        source = info["source"]
        cols = info["cols"]
        direction = info["direction"]

        for method_name, data in method_data.items():
            df = data[source]
            col = find_column(df, cols)

            if df is None or col is None:
                print(f"[WARNING] {metric_name} introuvable pour {method_name}")
                continue

            stats = summarize_metric(df[col])

            summary_rows.append({
                "dataset": dataset,
                "method": method_name,
                "metric": metric_name,
                "source": source,
                "column_used": col,
                "direction": direction,
                **stats,
            })

    summary_df = pd.DataFrame(summary_rows)
    runtime_rows = []

    for method_name, data in method_data.items():
        runtime = data["runtime"]

        runtime_rows.append({
            "dataset": dataset,
            "method": method_name,
            "runtime_total_s": runtime["runtime_total_s"],
            "runtime_hms": seconds_to_hms(runtime["runtime_total_s"]),
            "fps_global": runtime["fps_global"],
        })

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_csv = out_dir / "runtime_methods.csv"
    runtime_df.to_csv(runtime_csv, index=False)
    summary_csv = out_dir / "summary_statistics_methods.csv"
    summary_df.to_csv(summary_csv, index=False)

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    ranking_df = build_ranking_table(summary_df)

    ranking_csv = out_dir / "ranking_methods.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    # ------------------------------------------------------------
    # Pivot table compact
    # ------------------------------------------------------------

    if len(summary_df) > 0:
        pivot_mean = summary_df.pivot_table(
            index="metric",
            columns="method",
            values="mean",
            aggfunc="first",
        )

        pivot_median = summary_df.pivot_table(
            index="metric",
            columns="method",
            values="median",
            aggfunc="first",
        )

        pivot_mean.to_csv(out_dir / "pivot_mean_metrics.csv")
        pivot_median.to_csv(out_dir / "pivot_median_metrics.csv")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    for metric_name, info in metrics.items():
        source = info["source"]
        cols = info["cols"]
        ylabel = info["ylabel"]

        method_series = {}

        for method_name, data in method_data.items():
            df = data[source]
            col = find_column(df, cols)

            if df is None or col is None:
                continue

            method_series[method_name] = pd.to_numeric(df[col], errors="coerce")

        if len(method_series) == 0:
            continue

        name = safe_name(metric_name)

        if metric_name == "Reprojection":
            for sigma in [3, 5, 10]:
                plot_frame_by_frame(
                    metric_name=f"{metric_name} sigma={sigma}",
                    method_series=method_series,
                    ylabel=ylabel,
                    out_path=out_dir / f"{name}_frame_by_frame_sigma{sigma}.png",
                    sigma=sigma,
                )
        else:
            plot_frame_by_frame(
                metric_name=metric_name,
                method_series=method_series,
                ylabel=ylabel,
                out_path=out_dir / f"{name}_frame_by_frame.png",
            )

        plot_boxplot(
            metric_name=metric_name,
            method_series=method_series,
            ylabel=ylabel,
            out_path=out_dir / f"{name}_boxplot.png",
        )

        plot_histogram(
            metric_name=metric_name,
            method_series=method_series,
            xlabel=ylabel,
            out_path=out_dir / f"{name}_histogram.png",
        )
    generate_global_mind_maps(method_dirs, out_dir)
    generate_mind_comparison_panel(method_dirs, out_dir)
    print("\n" + "=" * 80)
    print("BENCHMARK MULTI-MÉTHODES TERMINÉ")
    print("=" * 80)
    print(f"Summary      : {summary_csv}")
    print(f"Ranking      : {ranking_csv}")
    print(f"Pivot mean   : {out_dir / 'pivot_mean_metrics.csv'}")
    print(f"Pivot median : {out_dir / 'pivot_median_metrics.csv'}")
    print(f"Plots        : {out_dir}")
    print(f"Runtime      : {runtime_csv}")


if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    extracted     = df["pair_id"].str.extract(r"seq_(\d+)")[0]
    df["seq_num"] = pd.to_numeric(extracted, errors="coerce") \
                      .fillna(pd.Series(range(len(df)))) \
                      .astype(int)
    df = df.sort_values("seq_num").reset_index(drop=True)

    if "mind_sim" not in df.columns:
        mind_range       = df["mind_ssd"].max() - df["mind_ssd"].min()
        df["mind_sim"]   = 1 - (df["mind_ssd"] - df["mind_ssd"].min()) / (mind_range + 1e-8)

    unmatched = df[extracted.isna()]
    if not unmatched.empty:
        print(f"  ⚠ {len(unmatched)} pair_id sans 'seq_<n>' : {unmatched['pair_id'].tolist()[:5]}")

    return df



def plot_overview(df: pd.DataFrame, output_dir: Path) -> None:
    seq = df["seq_num"].values
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(seq, df["goa"],          lw=2,        label="GOA (meilleur→1)")
    ax.plot(seq, df["edge_f1_8px"], lw=2,        label="Edge Alignment (meilleur→1)")
    ax.plot(seq, df["mind_sim"],      lw=1.5, ls="--",
            label=f"MIND sim. (meilleur→1) [plage réelle {df['mind_sim'].min():.3f}–{df['mind_sim'].max():.3f}]")
    ax.set_xlim(seq.min(), seq.max())   
    ax.set_xlabel("N° de séquence", fontsize=11)
    ax.set_ylabel("Score de qualité", fontsize=11)
    ax.set_title("Vue unifiée — qualité recalage VIS-SWIR", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower center", ncol=3, fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_overview.png", dpi=150)
    plt.close(fig)
    print("  ✓ plot_overview.png")


def plot_subplots(df: pd.DataFrame, output_dir: Path) -> None:
    seq     = df["seq_num"].values
    fig     = plt.figure(figsize=(14, 9))
    gs      = GridSpec(3, 1, hspace=0.45)
    configs = [
        ("goa", "GOA", "meilleur→1", "steelblue",
         (df["goa"].min() - 0.01, df["goa"].max() + 0.01)), 
        ("edge_f1_8px", "Edge Alignment", "meilleur→1", "darkorange",
         (df["edge_f1_8px"].min() - 0.01, df["edge_f1_8px"].max() + 0.01)),
        ("mind_ssd", "MIND SSD", "meilleur→0",  "seagreen",
         (df["mind_ssd"].min() - 0.003, df["mind_ssd"].max() + 0.003)),
    ]
    for i, (col, name, unit, color, ylim) in enumerate(configs):
        ax = fig.add_subplot(gs[i])
        ax.plot(seq, df[col], lw=1.8, color=color, label=f"{name} ({unit})")
        ax.set_xlim(seq.min(), seq.max())       
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("N° de séquence", fontsize=11)
    fig.suptitle("Métriques de qualité recalage VIS-SWIR", fontsize=13, y=1.01)
    fig.savefig(output_dir / "plot_subplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ plot_subplots.png")


def plot_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax  = plt.subplots(figsize=(8, 6))
    sc       = ax.scatter(df["goa"], df["edge_f1_8px"], c=df["mind_ssd"],
                          cmap="RdYlGn_r", s=25, alpha=0.75, edgecolors="none")
    cbar     = fig.colorbar(sc, ax=ax)
    cbar.set_label("MIND SSD (meilleur→0)", fontsize=10)
    goa_med  = df["goa"].median()
    edge_med = df["edge_f1_8px"].median()
    x_min, x_max = df["goa"].min(), df["goa"].max()
    y_min, y_max = df["edge_f1_8px"].min(), df["edge_f1_8px"].max()

    x_off = (x_max - x_min) * 0.18
    y_off = (y_max - y_min) * 0.08
    quadrants = [
        (goa_med + x_off, edge_med + y_off, "Bon recalage", "green"),
        (goa_med + x_off, edge_med - y_off, "Alignement partiel", "orange"),
        (goa_med - x_off, edge_med + y_off, "Contours proches", "orange"),
        (goa_med - x_off, edge_med - y_off, "Recalage dégradé", "red"),
    ]
    ax.axvline(goa_med,  color="gray", ls="--", lw=1, alpha=0.6, label=f"médiane GOA {goa_med:.3f}")
    ax.axhline(edge_med, color="gray", ls=":",  lw=1, alpha=0.6, label=f"médiane Edge Alignment {edge_med:.3f}")
  

    for x, y, label, color in quadrants:
        ax.text(x, y, label, fontsize=8, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))
    ax.set_xlabel("GOA (meilleur→1)", fontsize=11)
    ax.set_ylabel("Edge Alignment (meilleur→1)", fontsize=11)
    ax.set_title("Espace GOA–Edge Alignment coloré par MIND SSD", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_scatter_goa_edge_alignment.png", dpi=150)
    plt.close(fig)
    print("  ✓ plot_scatter_goa_egda.png")



def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation métriques qualité recalage")
    parser.add_argument("input", help="Chemin vers split_images/, results/, ou un .csv (relatif à datas/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Dossier de sortie des graphiques (défaut: même que results/)")
    args = parser.parse_args()

    projet_root = Path(__file__).resolve().parents[3]
    input_path  = (projet_root / "datas" / args.input).resolve()

    if input_path.suffix == ".csv":
        csv_path = input_path                              
    elif (input_path / "scores_qualite_recalage.csv").exists():
        csv_path = input_path / "scores_qualite_recalage.csv" 
    else:
        csv_path = input_path.parent / "results" / "scores_qualite_recalage.csv"  

    output_dir = csv_path.parent

    if not csv_path.exists():
        print(f"[ERREUR] CSV introuvable : {csv_path}")
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 CSV     : {csv_path}")
    print(f"📂 Sortie  : {output_dir}\n")

    df = load_df(csv_path)
    plot_overview(df, output_dir)                                           
    plot_subplots(df, output_dir)
    plot_scatter(df, output_dir)
    print(f"\n✅ Graphiques sauvegardés dans : {output_dir.resolve()}")




if __name__ == "__main__":
    main()


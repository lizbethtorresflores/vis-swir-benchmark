import os
import csv
import time
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import Config
import main as irvis_pipeline


def run_single_benchmark(num_images: int) -> float:
    print(f"[BENCH] Lancement NUM_IMAGES={num_images}")
    Config.NUM_IMAGES = num_images

    _argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0]]

    start = time.perf_counter()
    try:
        irvis_pipeline.main()
        print(f"[BENCH] main() terminé OK")
    except Exception as e:
        print(f"[BENCH] ERREUR dans main() : {e}")
        raise
    finally:
        sys.argv = _argv_backup  

    return time.perf_counter() - start



def run_benchmark(min_n: int, max_n: int, step: int, repeat: int, csv_path: str) -> None:
    config         = Config()
    script_dir     = os.path.dirname(os.path.abspath(__file__))
    paths          = config.setup_paths(script_dir)
    base_data_path = os.path.dirname(paths["rgb_dir"])
    sequence_name  = Path(base_data_path).parts[-3]

    fieldnames = [
        "sequence_name", "num_images", "run_idx", "elapsed_seconds",
        "grid_rows", "grid_cols", "grid_overlap_percent",
        "num_point_lg", "num_images_to_warp",
        "apply_distortion_vis", "apply_distortion_swir",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    csv_exists = os.path.exists(csv_path)

    print(f"[BENCH] Séquence    : {sequence_name}")
    print(f"[BENCH] Plage       : {min_n} → {max_n} (pas {step})")
    print(f"[BENCH] Répétitions : {repeat}")
    print(f"[BENCH] CSV sortie  : {csv_path}")
    print()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for num_images in range(min_n, max_n + 1, step):
            for run_idx in range(repeat):
                elapsed = run_single_benchmark(num_images)
                writer.writerow({
                    "sequence_name":         sequence_name,
                    "num_images":            num_images,
                    "run_idx":               run_idx,
                    "elapsed_seconds":       elapsed,
                    "grid_rows":             Config.GRID_ROWS,
                    "grid_cols":             Config.GRID_COLS,
                    "grid_overlap_percent":  Config.GRID_OVERLAP_PERCENT,
                    "num_point_lg":          Config.NUM_POINT_LG,
                    "num_images_to_warp":    Config.NUM_IMAGES_TO_WARP,
                    "apply_distortion_vis":  Config.APPLY_DISTORTION_VIS,
                    "apply_distortion_swir": Config.APPLY_DISTORTION_SWIR,
                })
                print(f"BENCH seq={sequence_name} NUM_IMAGES={num_images} run={run_idx} time={elapsed:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark temps d'exécution vs nombre d'images")
    parser.add_argument("--min",    type=int, default=5,  help="NUM_IMAGES minimal (défaut: 5)")
    parser.add_argument("--max",    type=int, default=50, help="NUM_IMAGES maximal (défaut: 50)")
    parser.add_argument("--step",   type=int, default=5,  help="Pas (défaut: 5)")
    parser.add_argument("--repeat", type=int, default=3,  help="Répétitions par valeur (défaut: 3)")
    parser.add_argument("--output", "-o", default=None,
                        help="Chemin du CSV de sortie (défaut: à côté du script)")
    args = parser.parse_args()

    csv_path = args.output or str(Path(__file__).parent / "benchmark_num_images.csv")
    run_benchmark(args.min, args.max, args.step, args.repeat, csv_path)


if __name__ == "__main__":
    main()

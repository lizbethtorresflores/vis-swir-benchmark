import time
import torch

from main import *

def profile_pipeline_simple(rgb_files, swir_files, config, paths):
    times = {}

    print("\n" + "="*80)
    print("PROFILING PIPELINE (simple)")
    print("="*80)

    if config.DEVICE == "mps":
        mem0 = torch.mps.current_allocated_memory() / 1024**3
        print(f"Mémoire GPU initiale : {mem0:.2f} GB")

    # Step 0
    t0 = time.perf_counter()
    map1_vis, map2_vis, map1_swir, map2_swir = step_0_calibration_estimation(
        rgb_files, swir_files, config, paths
    )
    times["calibration"] = time.perf_counter() - t0
    print(f"✓ calibration : {times['calibration']:.2f}s")

    config.UNDISTORT_MAPS = {
        "vis": (map1_vis, map2_vis),
        "swir": (map1_swir, map2_swir),
    }

    # Step 1
    t0 = time.perf_counter()
    container = step_1_extraction(rgb_files, swir_files, config)
    times["extraction"] = time.perf_counter() - t0
    print(f"✓ extraction : {times['extraction']:.2f}s")

    # Step 2
    t0 = time.perf_counter()
    Homography, inliers = step_2_estimation(container, config, paths)
    times["estimation"] = time.perf_counter() - t0
    print(f"✓ estimation : {times['estimation']:.2f}s")

    # Step 3
    t0 = time.perf_counter()
    step_3_application(Homography, rgb_files, swir_files, config, paths)
    times["application"] = time.perf_counter() - t0
    print(f"✓ application : {times['application']:.2f}s")

    total = sum(times.values())
    print("\n" + "="*80)
    print("RÉSUMÉ TEMPS")
    print("="*80)
    for k, v in times.items():
        print(f"{k:12s}: {v:6.2f}s ({100*v/total:5.1f}%)")
    print(f"{'TOTAL':12s}: {total:6.2f}s (100.0%)")

    if config.DEVICE == "mps":
        memf = torch.mps.current_allocated_memory() / 1024**3
        print(f"Mémoire GPU finale : {memf:.2f} GB")

    return Homography, inliers, container

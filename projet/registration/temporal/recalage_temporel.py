import os
import re
import argparse
import numpy as np
from pathlib import Path
from shutil import copy2


def extract_timestamp(filename):
    """
    Extrait un timestamp en nanosecondes depuis un nom de fichier au format
    seq_<n>_ts_<secs>_<nsecs>.

    Args:
        filename: Nom du fichier (avec ou sans extension).

    Returns:
        Timestamp en nanosecondes (float), ou None si le pattern n'est pas trouvé.
    """
    match = re.search(r'seq_\d+_ts_(\d+)_(\d+)', filename)
    if match:
        secs  = int(match.group(1))
        nsecs = int(match.group(2))
        return secs * 1e9 + nsecs
    return None


def find_closest_pairs(swir_files, visible_files, max_diff_ns=20e6):
    """
    Apparie chaque image SWIR à l'image VIS dont le timestamp est le plus
    proche, dans la limite de max_diff_ns. Résout les doublons en conservant
    uniquement la meilleure paire par image VIS.

    Args:
        swir_files:    Liste de noms de fichiers SWIR.
        visible_files: Liste de noms de fichiers VIS.
        max_diff_ns:   Écart temporel maximal autorisé en nanosecondes.

    Returns:
        Tuple (final_pairs, unmatched_swir, unmatched_visible, avg_diff_ms).
        final_pairs     : liste de tuples (swir_file, visible_file, diff_ns).
        unmatched_swir  : fichiers SWIR sans correspondance.
        unmatched_visible: fichiers VIS sans correspondance.
        avg_diff_ms     : écart moyen en millisecondes (None si aucune paire).
    """
    visible_timestamps = [(extract_timestamp(f), f) for f in visible_files if extract_timestamp(f) is not None]
    visible_timestamps.sort()

    print(f"\n🔎 {len(swir_files)} images SWIR trouvées")
    print(f"🔎 {len(visible_timestamps)} images VISIBLE trouvées après extraction des timestamps\n")

    raw_pairs      = []
    unmatched_swir = []

    for swir_file in swir_files:
        swir_timestamp = extract_timestamp(swir_file)
        if swir_timestamp is None:
            unmatched_swir.append(swir_file)
            continue

        closest_visible = None
        min_diff        = float('inf')

        for visible_ts, visible_file in visible_timestamps:
            diff = abs(swir_timestamp - visible_ts)
            if diff < min_diff and diff <= max_diff_ns:
                min_diff        = diff
                closest_visible = visible_file
            elif visible_ts > swir_timestamp + max_diff_ns:
                break

        if closest_visible:
            raw_pairs.append((swir_file, closest_visible, min_diff))
        else:
            unmatched_swir.append(swir_file)

    print(f"📊 {len(raw_pairs)} paires trouvées avant filtrage des doublons\n")

    best_matches = {}
    for swir_file, visible_file, diff in raw_pairs:
        if visible_file in best_matches:
            _, _, existing_diff = best_matches[visible_file]
            if diff < existing_diff:
                best_matches[visible_file] = (swir_file, visible_file, diff)
        else:
            best_matches[visible_file] = (swir_file, visible_file, diff)

    final_pairs = list(best_matches.values())
    print(f"✅ {len(final_pairs)} paires finales après suppression des doublons\n")

    matched_visible_files = {vis for _, vis, _ in final_pairs}
    unmatched_visible     = [vis for _, vis in visible_timestamps if vis not in matched_visible_files]

    print("🔹 Liste des paires finales :")
    timestamp_diffs = []
    for swir, vis, diff in final_pairs:
        print(f"  Pairé : {swir} ↔ {vis} (Écart : {diff / 1e6:.3f} ms)")
        timestamp_diffs.append(diff / 1e6)

    print(f"\n🔻 Images SWIR sans correspondance : {len(unmatched_swir)}")
    print(f"🔻 Images VISIBLE sans correspondance : {len(unmatched_visible)}")

    avg_diff = np.mean(timestamp_diffs) if timestamp_diffs else None
    if avg_diff is not None:
        print(f"\n📊 Écart moyen des paires finales : {avg_diff:.3f} ms\n")
    else:
        print("\n⚠️ Aucun appariement valide, impossible de calculer l'écart moyen.\n")

    return final_pairs, unmatched_swir, unmatched_visible, avg_diff


def create_image_pairs(swir_dir: Path, visible_dir: Path, output_dir: Path, max_diff_ns: float = 20e6):
    """
    Orchestre l'appariement temporel et copie les paires retenues dans
    output_dir/SWIR/ et output_dir/VISIBLE/.

    Args:
        swir_dir:    Dossier contenant les images SWIR (.tiff).
        visible_dir: Dossier contenant les images VIS (.tiff).
        output_dir:  Dossier de destination pour les paires copiées.
        max_diff_ns: Écart temporel maximal autorisé en nanosecondes.
    """
    valid_exts = ('.tiff', '.tif', '.jpg', '.jpeg', '.png')

    swir_files    = [f for f in os.listdir(swir_dir)    if f.lower().endswith(valid_exts)]
    visible_files = [f for f in os.listdir(visible_dir) if f.lower().endswith(valid_exts)]

    pairs, unmatched_swir, unmatched_visible, avg_diff = find_closest_pairs(
        swir_files, visible_files, max_diff_ns=max_diff_ns   
    )

    if not pairs:
        print("⚠️ Aucun appariement trouvé, arrêt du script.")
        return

    copied_swir = copied_visible = 0

    for swir_file, visible_file, _ in pairs:
        swir_out    = output_dir / "SWIR"    / swir_file
        visible_out = output_dir / "VISIBLE" / visible_file

        swir_out.parent.mkdir(parents=True, exist_ok=True)
        visible_out.parent.mkdir(parents=True, exist_ok=True)

        try:
            copy2(swir_dir    / swir_file,    swir_out)
            copied_swir += 1
        except Exception as e:
            print(f"❌ Erreur copie SWIR {swir_file} : {e}")

        try:
            copy2(visible_dir / visible_file, visible_out)
            copied_visible += 1
        except Exception as e:
            print(f"❌ Erreur copie VISIBLE {visible_file} : {e}")

    print(f"\n✅ {copied_swir} images SWIR et {copied_visible} images VISIBLE copiées.")
    if avg_diff is not None:
        print(f"📊 Écart moyen final : {avg_diff:.3f} ms")


def main():
    """
    Point d'entrée CLI. Parse les arguments (chemin bag, sous-dossiers,
    écart max) et exécute create_image_pairs.
    """
    parser = argparse.ArgumentParser(description="Appariement temporel SWIR / VISIBLE")
    parser.add_argument("bag", help="Chemin relatif depuis datas/ vers le dossier Sortie_Extraite du bag")
    parser.add_argument("--swir",    default="camera_SWIR", help="Sous-dossier SWIR    (défaut: camera_SWIR)")
    parser.add_argument("--visible", default="camera_AV",   help="Sous-dossier VISIBLE (défaut: camera_AV)")
    parser.add_argument("--output",  "-o", default=None,    help="Dossier de sortie    (défaut: <bag>/paired_images)")
    parser.add_argument("--max-diff-ms", type=float, default=20.0,
                        help="Écart temporel max en ms (défaut: 20 ms)")
    args = parser.parse_args()

    projet_root  = Path(__file__).resolve().parents[3]   # Images_Registration_Project/
    bag_root     = (projet_root / "datas" / args.bag).resolve()
    swir_dir     = bag_root / args.swir
    visible_dir  = bag_root / args.visible
    output_dir   = Path(args.output).resolve() if args.output else bag_root / "paired_images"

    for d, name in [(swir_dir, "SWIR"), (visible_dir, "VISIBLE")]:
        if not d.exists():
            print(f"[ERREUR] Dossier {name} introuvable : {d}")
            raise SystemExit(1)

    print(f"📂 SWIR    : {swir_dir}")
    print(f"📂 VISIBLE : {visible_dir}")
    print(f"📂 Sortie  : {output_dir}")

    create_image_pairs(swir_dir, visible_dir, output_dir, max_diff_ns=args.max_diff_ms * 1e6)
    print("\n🚀 Script terminé.")


if __name__ == "__main__":
    main()

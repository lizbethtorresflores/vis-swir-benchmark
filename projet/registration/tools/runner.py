"""
runner.py
---------
Exécute une liste de scripts Python, capture leurs résultats et les stocke dans un CSV.

Usage :
    python runner.py script1.py script2.py script3.py
    python runner.py script1.py script2.py --output resultats.csv
    python runner.py script1.py script2.py --python mon_venv/bin/python
    python runner.py script1.py script2.py --python mon_venv/ --output resultats.csv

Format attendu dans les scripts exécutés (via print) :
    RUN_IDX: <numéro de boucle>
    EPOCHS: <nombre répétitions OU epochs entraînement>
    NUM_IMAGES: <nombre>             (optionnel si dans config.py)
    GRID_OVERLAP_PERCENT: <pourcentage>  (optionnel si dans config.py)
    NUM_IMAGES_TO_WARP: <nombre>     (optionnel si dans config.py)

Le runner cherche automatiquement un fichier config.py dans le même dossier que le script
et extrait NUM_IMAGES, GRID_OVERLAP_PERCENT, NUM_IMAGES_TO_WARP comme valeurs par défaut.
Les valeurs dans stdout (via print) ont priorité sur celles du config.py.

Exemple dans un script cible :
    print("RUN_IDX: 0")
    print("EPOCHS: 5")
    # NUM_IMAGES sera lu depuis config.py si présent
"""

import subprocess
import time
import csv
import argparse
import re
import sys
import os
from datetime import datetime
from pathlib import Path


# ─── Constantes ───────────────────────────────────────────────────────────────

OUTPUT_CSV_DEFAULT = "resultats.csv"
CSV_HEADERS = [
    "sequence_name",           # Nom de la séquence (depuis data/)
    "script_name",             # Nom du script (parent.name)
    "date_execution",          # Date d'exécution
    "temps_execution_s",       # Temps d'exécution
    "run_idx",                 # Numéro de la boucle d'exécution (steps)
    "epochs",                  # Nombre d'epochs (x répétitions OU epochs d'entraînement)
    "num_images",              # Nombre d'images pour entraînement/homographie
    "grid_overlap_percent",    # Overlap (si applicable)
    "num_images_to_warp",      # Nombre d'images à warp
    "statut",                  # succès ou erreur
    "erreur",                  # Message d'erreur si applicable
]

# Patterns pour parser la sortie des scripts
PATTERN_RUN_IDX = re.compile(r"RUN_IDX\s*:\s*(\d+)", re.IGNORECASE)
PATTERN_EPOCHS = re.compile(r"EPOCHS\s*:\s*(\d+)", re.IGNORECASE)
PATTERN_NUM_IMAGES = re.compile(r"NUM_IMAGES\s*:\s*(\d+)", re.IGNORECASE)
PATTERN_OVERLAP = re.compile(r"GRID_OVERLAP_PERCENT\s*:\s*(\d+)", re.IGNORECASE)
PATTERN_WARP = re.compile(r"NUM_IMAGES_TO_WARP\s*:\s*(\d+)", re.IGNORECASE)


# ─── Fonctions ────────────────────────────────────────────────────────────────

def extract_config_from_script(script_path: str) -> dict:
    """
    Tente d'extraire les valeurs de configuration depuis un config.py
    situé dans le même dossier que le script.
    
    Cherche les attributs suivants dans la classe Config :
    - NUM_IMAGES
    - GRID_OVERLAP_PERCENT
    - NUM_IMAGES_TO_WARP
    
    Retourne un dictionnaire avec les valeurs trouvées (None si absentes).
    """
    script_dir = Path(script_path).parent
    config_path = script_dir / "config.py"
    
    result = {
        "num_images": None,
        "grid_overlap_percent": None,
        "num_images_to_warp": None,
    }
    
    if not config_path.exists():
        return result
    
    try:
        # Lecture du fichier config.py
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patterns pour extraire les valeurs depuis config.py
        patterns = {
            "num_images": re.compile(r"NUM_IMAGES\s*=\s*(\d+)", re.MULTILINE),
            "grid_overlap_percent": re.compile(r"GRID_OVERLAP_PERCENT\s*=\s*(\d+)", re.MULTILINE),
            "num_images_to_warp": re.compile(r"NUM_IMAGES_TO_WARP\s*=\s*(\d+)", re.MULTILINE),
        }
        
        for key, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                result[key] = int(match.group(1))
    
    except Exception as e:
        # Si erreur de lecture, on retourne les valeurs None
        pass
    
    return result


def get_sequences_from_data_folder() -> list:
    """
    Récupère la liste des séquences depuis le dossier data/ (au même niveau que runner.py).
    Retourne la liste des noms de dossiers trouvés dans data/.
    """
    script_dir = Path(__file__).parent
    data_dir = script_dir / "datas/transfer_11680406_files_36b18a31/objectif_12mm_4_boucle"
    
    if not data_dir.exists():
        return []
    
    sequences = [d.name for d in data_dir.iterdir() if d.is_dir()]
    return sequences


def resolve_python(python_arg: str) -> str:
    """
    Résout le chemin de l'exécutable Python à utiliser.
    Accepte :
      - None                      → Python courant (sys.executable)
      - chemin vers un dossier    → détecte bin/python ou Scripts/python.exe (racine venv)
      - chemin direct vers python → utilisé tel quel

    Note : utilise os.path.lexists() pour ne pas rater les liens symboliques
    qui pointent vers une cible valide mais que Path.exists() rate parfois.
    """
    if python_arg is None:
        return sys.executable

    path = Path(python_arg)

    # Cas 1 : c'est un dossier → on suppose que c'est la racine d'un venv
    if path.is_dir():
        candidates = [
            path / "bin" / "python",          # Linux / macOS
            path / "bin" / "python3",          # Linux / macOS alternatif
            path / "bin" / "python3.9",        # version explicite
            path / "bin" / "python3.10",
            path / "bin" / "python3.11",
            path / "bin" / "python3.12",
            path / "Scripts" / "python.exe",   # Windows
        ]
        for candidate in candidates:
            # lexists() détecte le fichier même si c'est un lien symbolique cassé
            if os.path.lexists(candidate):
                resolved = candidate.resolve()
                print(f"  ℹ Venv détecté → interpréteur : {candidate} → {resolved}")
                return str(candidate)
        raise FileNotFoundError(
            f"Aucun exécutable Python trouvé dans le venv '{python_arg}'.\n"
            f"  Chemins essayés : {[str(c) for c in candidates]}"
        )

    # Cas 2 : chemin direct vers un exécutable (gère aussi les symlinks)
    if os.path.lexists(path):
        return str(path)

    raise FileNotFoundError(f"Exécutable Python introuvable : {python_arg}")


def parse_output(output: str) -> dict:
    """Extrait les métriques depuis la sortie standard d'un script."""
    result = {
        "run_idx": None,
        "epochs": None,
        "num_images": None,
        "grid_overlap_percent": None,
        "num_images_to_warp": None,
    }

    match = PATTERN_RUN_IDX.search(output)
    if match:
        result["run_idx"] = int(match.group(1))

    match = PATTERN_EPOCHS.search(output)
    if match:
        result["epochs"] = int(match.group(1))

    match = PATTERN_NUM_IMAGES.search(output)
    if match:
        result["num_images"] = int(match.group(1))

    match = PATTERN_OVERLAP.search(output)
    if match:
        result["grid_overlap_percent"] = int(match.group(1))

    match = PATTERN_WARP.search(output)
    if match:
        result["num_images_to_warp"] = int(match.group(1))

    return result


def run_script(script_path: str, python_exec: str, sequence_name: str = None) -> dict:
    """
    Exécute un script Python et retourne un dictionnaire avec :
    - temps d'exécution
    - métriques parsées depuis stdout
    - statut (succès / erreur)
    - message d'erreur éventuel
    
    Args:
        script_path: Chemin vers le script à exécuter
        python_exec: Exécutable Python à utiliser
        sequence_name: Nom de la séquence (auto-détecté depuis data/ si None)
    """
    # Auto-détection de la séquence si non fournie
    if sequence_name is None:
        sequences = get_sequences_from_data_folder()
        sequence_name = sequences[0] if sequences else "unknown"
    
    # Extraction des configs depuis config.py (fallback)
    config_values = extract_config_from_script(script_path)
    
    result = {
        "sequence_name":         sequence_name,
        "script_name":           Path(script_path).parent.name,  # parent.name comme demandé
        "date_execution":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temps_execution_s":     None,
        "run_idx":               0,  # Valeur par défaut à 0
        "epochs":                0,  # Valeur par défaut à 0
        "num_images":            config_values["num_images"] if config_values["num_images"] is not None else 0,
        "grid_overlap_percent":  config_values["grid_overlap_percent"] if config_values["grid_overlap_percent"] is not None else 0,
        "num_images_to_warp":    config_values["num_images_to_warp"] if config_values["num_images_to_warp"] is not None else 0,
        "statut":                "erreur",
        "erreur":                "",
    }

    if not Path(script_path).exists():
        result["erreur"] = f"Fichier introuvable : {script_path}"
        print(f"  ✗ {result['erreur']}")
        return result

    print(f"  → Exécution de : {script_path}  [via {python_exec}]")
    print(f"    Séquence détectée : {sequence_name}")
    
    # Afficher les valeurs extraites du config si présentes
    if any(v is not None for v in config_values.values()):
        print(f"    Config détecté :")
        if config_values["num_images"] is not None:
            print(f"      NUM_IMAGES = {config_values['num_images']}")
        if config_values["grid_overlap_percent"] is not None:
            print(f"      GRID_OVERLAP_PERCENT = {config_values['grid_overlap_percent']}")
        if config_values["num_images_to_warp"] is not None:
            print(f"      NUM_IMAGES_TO_WARP = {config_values['num_images_to_warp']}")

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [python_exec, script_path],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.perf_counter() - start
        result["temps_execution_s"] = round(elapsed, 4)

        if proc.returncode != 0:
            result["statut"] = "erreur"
            result["erreur"] = proc.stderr.strip()[:500]
            print(f"  ✗ Erreur (code {proc.returncode}) en {elapsed:.2f}s")
            if proc.stderr:
                print(f"    {proc.stderr.strip()[:200]}")
        else:
            # Parse la sortie standard (priorité sur config.py)
            metriques = parse_output(proc.stdout)
            
            # Merge : stdout override config.py et valeurs par défaut
            for key, value in metriques.items():
                if value is not None:
                    result[key] = value
            
            result["statut"] = "succès"

            # Affichage des métriques capturées
            print(f"  ✓ Succès en {elapsed:.2f}s")
            print(f"    Run IDX: {result['run_idx']}")
            print(f"    Epochs: {result['epochs']}")
            print(f"    Images: {result['num_images']}")
            print(f"    Overlap: {result['grid_overlap_percent']}%")
            print(f"    Images to warp: {result['num_images_to_warp']}")

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        result["temps_execution_s"] = round(elapsed, 4)
        result["erreur"] = "Timeout dépassé (> 1h)"
        print(f"  ✗ Timeout après {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.perf_counter() - start
        result["temps_execution_s"] = round(elapsed, 4)
        result["erreur"] = str(e)
        print(f"  ✗ Exception inattendue : {e}")

    return result


def save_to_csv(rows: list, output_path: str, append: bool = True) -> None:
    """
    Écrit les résultats dans un fichier CSV.
    
    Args:
        rows: Liste de dictionnaires avec les résultats
        output_path: Chemin du fichier CSV
        append: Si True, ajoute à la fin du fichier (mode append).
                Si False, écrase le fichier existant.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Vérifier si le fichier existe déjà
    csv_exists = path.exists()

    # Mode append ('a') ou write ('w')
    mode = "a" if append else "w"

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        
        # Écrire le header seulement si :
        # - mode write OU
        # - mode append ET le fichier n'existait pas
        if not append or not csv_exists:
            writer.writeheader()
        
        writer.writerows(rows)

    action = "ajoutés au" if (append and csv_exists) else "sauvegardés dans"
    print(f"\n✅ Résultats {action} : {path.resolve()}")


def print_summary(rows: list, python_exec: str) -> None:
    """Affiche un résumé en fin d'exécution."""
    total   = len(rows)
    succes  = sum(1 for r in rows if r["statut"] == "succès")
    erreurs = total - succes
    temps   = [r["temps_execution_s"] for r in rows if r["temps_execution_s"] is not None]

    print("\n" + "─" * 55)
    print("RÉSUMÉ")
    print("─" * 55)
    print(f"  Interpréteur     : {python_exec}")
    print(f"  Scripts exécutés : {total}")
    print(f"  Succès           : {succes}")
    print(f"  Erreurs          : {erreurs}")
    if temps:
        print(f"  Temps total      : {sum(temps):.2f}s")
        print(f"  Temps moyen      : {sum(temps)/len(temps):.2f}s")
    print("─" * 55)


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exécute des scripts Python et stocke les résultats dans un CSV."
    )
    parser.add_argument(
        "scripts",
        nargs="+",
        help="Chemins vers les scripts Python à exécuter",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_CSV_DEFAULT,
        help=f"Nom du fichier CSV de sortie (défaut : {OUTPUT_CSV_DEFAULT})",
    )
    parser.add_argument(
        "--python", "-p",
        default=None,
        metavar="VENV_OU_PYTHON",
        help=(
            "Interpréteur Python à utiliser. Accepte :\n"
            "  - Un dossier venv : --python mon_venv/\n"
            "  - Un exécutable   : --python mon_venv/bin/python\n"
            "  - (par défaut)    : Python courant"
        ),
    )
    parser.add_argument(
        "--sequence", "-s",
        default=None,
        metavar="SEQUENCE_NAME",
        help="Nom de la séquence (auto-détecté depuis data/ si omis)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écrase le fichier CSV au lieu d'y ajouter les résultats (défaut: append)",
    )
    args = parser.parse_args()

    # Résolution de l'interpréteur
    try:
        python_exec = resolve_python(args.python)
    except FileNotFoundError as e:
        print(f"\n✗ Erreur : {e}")
        sys.exit(1)

    # Détection de la séquence
    sequence_name = args.sequence
    if sequence_name is None:
        sequences = get_sequences_from_data_folder()
        if sequences:
            sequence_name = sequences[0]
            print(f"\n  ℹ Séquence auto-détectée : {sequence_name}")
            if len(sequences) > 1:
                print(f"    Autres séquences disponibles : {', '.join(sequences[1:])}")
        else:
            sequence_name = "unknown"
            print(f"\n  ⚠ Aucune séquence trouvée dans data/, utilisation de '{sequence_name}'")

    print(f"\n{'═' * 55}")
    print(f"  RUNNER — {len(args.scripts)} script(s) à exécuter")
    print(f"  Python   : {python_exec}")
    print(f"  Séquence : {sequence_name}")
    print(f"  Mode     : {'ÉCRASEMENT' if args.overwrite else 'AJOUT (append)'}")
    print(f"{'═' * 55}\n")

    rows = []
    for script in args.scripts:
        result = run_script(script, python_exec, sequence_name)
        rows.append(result)

    # append=True par défaut, sauf si --overwrite est passé
    save_to_csv(rows, args.output, append=not args.overwrite)
    print_summary(rows, python_exec)


if __name__ == "__main__":
    main()

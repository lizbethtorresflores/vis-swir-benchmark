PYTHON := ../venv/bin/python

BAG   ?=
INPUT ?=
ARGS  ?=

# ─── Données ────────────────────────────────────────────────────────────────
extract:
ifndef BAG
	$(error Usage: make extract BAG=<chemin relatif depuis datas/>)
endif
	$(PYTHON) registration/tools/ros2_bag_extractor.py "$(BAG)" $(ARGS)

pair:
ifndef BAG
	$(error Usage: make pair BAG=<chemin relatif depuis datas/ vers Sortie_Extraite>)
endif
	$(PYTHON) registration/temporal/recalage_temporel.py "$(BAG)" $(ARGS)

# ─── Pipeline principal ──────────────────────────────────────────────────────
run:
ifndef BAG
	$(error Usage: make run BAG=<chemin relatif depuis datas/>)
endif
	$(PYTHON) main.py --bag "$(BAG)" $(ARGS)

# ─── Post-traitement ────────────────────────────────────────────────────────
fuse:
ifndef BAG
	$(error Usage: make fuse BAG=<chemin relatif depuis datas/ vers le dossier de calibration>)
endif
	$(PYTHON) -m registration.fusion.fusion_HSV "$(BAG:%/=%)/split_images" $(ARGS)


# ─── Évaluation ─────────────────────────────────────────────────────────────
bench:
ifndef BAG
	$(error Usage: make bench BAG=<chemin relatif depuis datas/>)
endif
	$(PYTHON) -m registration.evaluation.quality_bench "$(BAG:%/=%)/split_images" $(ARGS)

bench-compare:
ifndef BAG
	$(error Usage: make bench-compare BAG=<séquence> INPUT="methode1/split_images methode2/split_images ...")
endif
	$(PYTHON) registration/evaluation/quality_bench.py \
		$(foreach m,$(INPUT),"$(BAG)/Sortie_Extraite/paired_images/$(m)") $(ARGS)

bench-images:
	$(PYTHON) registration/evaluation/num_images_benchmark.py $(ARGS)

plot:
ifndef BAG
	$(error Usage: make plot BAG=<chemin relatif depuis datas/>)
endif
	$(PYTHON) registration/evaluation/plot_quality.py \
		"$(BAG)/split_images" $(ARGS)

# ─── Enchaînements logiques ──────────────────────────────────────────────────
evaluate:
ifndef BAG
	$(error Usage: make evaluate BAG=<chemin relatif depuis datas/>)
endif
	$(MAKE) fuse  BAG="$(BAG)" ARGS="$(ARGS)"
	$(MAKE) bench BAG="$(BAG)" ARGS="$(ARGS)"
	$(MAKE) plot  BAG="$(BAG)" ARGS="$(ARGS)"

all:
ifndef BAG
	$(error Usage: make all BAG=<chemin relatif depuis datas/>)
endif
	$(MAKE) extract  BAG="$(BAG)"
	$(MAKE) pair     BAG="$(BAG)/Sortie_Extraite"
	$(MAKE) run      BAG="$(BAG)"
	$(MAKE) evaluate BAG="$(BAG)"

# ─── Aide ────────────────────────────────────────────────────────────────────
# ─── Aide ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           Pipeline de recalage SWIR/VIS — aide                  ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Deux conventions de chemin selon la cible :"
	@echo ""
	@echo "  [A] BAG = chemin relatif depuis datas/ vers la séquence ROS2"
	@echo "      ex: transfer_11680406_files_36b18a31/objectif_12mm_3_arret"
	@echo ""
	@echo "  [B] BAG = chemin relatif depuis datas/ vers le dossier de calibration"
	@echo "      ex: transfer_11680406_files_36b18a31/objectif_12mm_3_arret/"
	@echo "              Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration"
	@echo "      ou avec homographie croisée :"
	@echo "              .../patch_RANSAC_HOMOGRAPHY_calibration_H_from_objectif_12mm_4_boucle"
	@echo ""
	@echo "─── Données ──────────────────────────────────────────────────────"
	@echo "  make extract  BAG=<[A]>   Extraire les images d'un bag ROS2"
	@echo "  make pair     BAG=<[A]/Sortie_Extraite>"
	@echo "                            Apparier SWIR/VISIBLE par timestamp"
	@echo ""
	@echo "─── Pipeline principal ───────────────────────────────────────────"
	@echo "  make run      BAG=<[A]>   Recalage + estimation homographie"
	@echo "                            → génère le dossier de calibration [B]"
	@echo ""
	@echo "─── Post-traitement (attendent [B]) ──────────────────────────────"
	@echo "  make fuse     BAG=<[B]>   Fusion HSV VIS+SWIR"
	@echo "                            → <[B]>/fusion_HSV/"
	@echo "  make bench    BAG=<[B]>   Métriques qualité GOA/EGDA/MIND"
	@echo "                            → <[B]>/split_images/metrics.csv"
	@echo "  make plot     BAG=<[B]>   Visualisation des métriques"
	@echo "                            → <[B]>/split_images/*.png"
	@echo ""
	@echo "─── Benchmarks spéciaux ──────────────────────────────────────────"
	@echo "  make bench-compare  BAG=<[A]> INPUT=\"methode1 methode2 ...\""
	@echo "                            Compare plusieurs méthodes sur une même séquence"
	@echo "  make bench-images         Benchmark temps d'exécution vs nb d'images"
	@echo ""
	@echo "─── Enchaînements ────────────────────────────────────────────────"
	@echo "  make evaluate BAG=<[B]>   fuse + bench + plot"
	@echo "  make all      BAG=<[A]>   extract + pair + run + evaluate"
	@echo "                            ⚠  evaluate utilisera le dossier de"
	@echo "                               calibration par défaut (sans H croisée)"
	@echo ""
	@echo "─── Exemples ─────────────────────────────────────────────────────"
	@echo "  # Pipeline complet sur une séquence"
	@echo "  make all BAG=transfer_11680406_files_36b18a31/objectif_12mm_3_arret"
	@echo ""
	@echo "  # Fusion avec homographie issue d'une autre séquence"
	@echo "  make fuse BAG=transfer_11680406_files_36b18a31/objectif_12mm_3_arret/\\"
	@echo "    Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration_H_from_objectif_12mm_4_boucle"
	@echo ""
	@echo "  # Évaluation sur un dossier de calibration spécifique"
	@echo "  make evaluate BAG=transfer_11680406_files_36b18a31/objectif_12mm_3_arret/\\"
	@echo "    Sortie_Extraite/paired_images/patch_RANSAC_HOMOGRAPHY_calibration"
	@echo ""

.PHONY: extract pair run fuse bench bench-images plot evaluate all help



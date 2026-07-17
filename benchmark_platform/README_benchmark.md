# VIS-SWIR Benchmark Platform

Streamlit benchmark application for comparing VIS-SWIR global homography registration methods.

## Run

```bash
cd /home/lizbeth/Desktop/intership/recallage-multimodal-zz3
venv/bin/streamlit run benchmark_platform/benchmark_app.py --server.port 8503
```

If the port is busy, use another one, for example `--server.port 8504`.

## What The Benchmark Does

The benchmark evaluates global homographies, not source code. A participant submits metadata and one `H_global` matrix. The platform applies that homography internally to the fixed VIS-SWIR test split, applies the official common crop, computes metrics, and updates the official leaderboard.

Same test set. Same common crop. Same metrics. Same protocol.

## Accepted Homography Formats

The `Submit Results` page accepts:

- manual text: a numeric `3x3` matrix
- `.txt`: a text file containing a numeric `3x3` matrix
- `.npy`: a NumPy array with shape `3x3`

The platform validates shape, finite values, determinant, invertibility, and numerical stability before evaluation.

## Required Metadata

Required fields:

- method name
- dataset
- direction
- training pairs used
- global homography `H`

Optional fields:

- authors
- institution
- paper URL
- contact email
- runtime seconds
- FPS
- description

The submission date is generated automatically.

## Official Protocol

The official protocol is:

```text
official_common_crop
```

Steps:

1. Apply the submitted global homography.
2. Generate aligned VIS/SWIR images internally.
3. Apply the official common crop for the selected dataset and direction.
4. Compute GOA, Edge Alignment, MIND-SSD, and Reprojection Error on the official common-crop area.
5. Save official common-crop metrics in `metrics.csv`.
6. Add only those official common-crop metrics to the leaderboard.

The test split is fixed for all methods. The evaluation server must not be used for parameter tuning.

## Official Common Crop

The official common crop is defined in the target aligned-image coordinate system:

- `VIS_to_SWIR`: crop is applied in SWIR/reference coordinates.
- `SWIR_to_VIS`: crop is applied in VIS/reference coordinates.

The same crop is applied consistently to both metric images after alignment.

Different registration methods may generate different valid image regions after applying their global homography. Some methods produce larger valid areas, while others introduce black borders or invalid regions.

To ensure a fair comparison, the benchmark evaluates every method on the same official common region. All official metrics are computed only inside this common crop, so leaderboard differences come from registration quality rather than different image extents.

The common crop is part of the evaluation protocol, not part of the method. The interface should show one official result per method/dataset/direction rather than separate protocol variants as competing methods.

## Leaderboard Rules

The leaderboard uses only `metrics.csv`, which contains official common-crop metrics.

`metrics_full_image_diagnostic.csv` may be saved for debugging, but it is not used for ranking.

If the same `method_name + dataset + direction` is submitted multiple times, a new version folder is created:

```text
version_001/
version_002/
version_003/
```

The leaderboard keeps only the latest official version unless a separate version history page is added.

## Output Structure

```text
benchmark_results/
└── methods/
    └── MethodName/
        └── sunny/
            └── VIS_to_SWIR/
                └── version_001/
                    ├── metadata.json
                    ├── H_global.npy
                    ├── runtime.json
                    ├── metrics.csv
                    ├── metrics_full_image_diagnostic.csv
                    ├── split_images/
                    ├── concatenate_images/
                    ├── split_images_official_common_crop/
                    ├── concatenate_images_official_common_crop/
                    └── evaluation_log.txt
```

## Dataset Exposure

The full dataset should not be exposed through the interface. The app shows rankings, protocol information, method metadata, controlled diagnostics, and limited visual examples only.

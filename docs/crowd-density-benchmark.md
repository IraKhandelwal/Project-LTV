# Stage 2: crowd-count benchmark (candidate only)

## Purpose

This increment evaluates an image-level crowd-counting model on a public held-out dataset. It does not connect model output to the operator dashboard or use it to trigger operational alerts.

## Candidate

- **CSRNet**, configured to output a spatial density map. The sum of the map estimates the number of people in the image.
- **DM-Count**, using the authors' VGG-19 density-map architecture and the NWPU-pretrained checkpoint saved locally as `backend/models/dmcount-nwpu.pth`. The official authors' code is MIT licensed and links pretrained checkpoints for NWPU and other crowd datasets. Checkpoint SHA-256: `f2a0c92ac22b5c6aee08b59ad9f002561e75da07555b58c20dd699db8aac59b2`. Review the checkpoint and NWPU dataset terms separately before any company/product use.
- **MRCNet independent reproduction**, a paper-inspired multi-resolution crowd-counting and density-map network. The [paper](https://arxiv.org/abs/1909.12743) evaluates MRCNet on DLR-ACD and reports MAE 906, MNAE 0.21, and RMSE 1307.4; those are the paper authors' results, not ours. No author-linked code or checkpoint was found. Our version 2 has 19.76 million parameters (the paper reports 20.3 million) and MAE 2,046.28 on our exploratory 14-image test run. It is not the official model or approved for deployment.
- Candidate checkpoint: the MIT-licensed `rootstrap-org/crowd-counting` model card, trained on ShanghaiTech Part B. The model card reports MAE 10.6 on its stated evaluation; that result is not a result for this project.
- Downloaded checkpoint SHA-256: `6d890bd6232e3b48e0e228525f22946d12fdbb102afc6487e6f6f899e2b70f45` (recorded in `backend/models/csrnet-shanghai-b.pth.sha256`).
- Architecture reference: the CSRNet authors' PyTorch implementation and CVPR 2018 paper.
- Status: **research candidates only**. None is approved for deployment; record checkpoint provenance and review model/data terms before any product inclusion.

CSRNet estimates a density map and count. It does not identify or track individuals. It cannot produce persons/m² without a calibrated ground-area mapping.

## Public evaluation data

The downloaded **DLR Aerial Crowd Dataset (DLR-ACD)** is being used for this evaluation. It contains 33 aerial images split into training and test folders, with separate black-and-white annotation PNGs. The test split contains 14 images. For each image, one white annotation pixel marks one person. Use only `Test/Images` and matching `Test/Annotation` files for evaluation; do not train or fine-tune on this split. The dataset is excluded from version control and must not be bundled with the software.

**License:** DLR says the dataset is for civilian research use only; commercial use requires contacting DLR. The project is intended for company/product work, so obtain written permission before using these results for commercial development or deployment. Do not redistribute or modify the dataset.

The benchmark reports image-count MAE, RMSE, and signed bias against annotated people. These are not the SRS acceptance measures for persons/m². DLR aerial photos are more relevant to a high camera angle than ordinary street photos, but they do not validate overhead 360° vehicle-camera footage, Indian crowd representation, zones, calibration, alert accuracy, or real-time performance.

### First CSRNet result on DLR-ACD

The ShanghaiTech Part B checkpoint was evaluated on all 14 DLR-ACD test images:

| Measure | Result |
| --- | ---: |
| Mean absolute error | 4,608.96 people per image |
| Root mean squared error | 6,594.53 people per image |
| Average signed error | -4,598.32 people per image |

The negative signed error means the model usually counted far too few people. This candidate is **not suitable for these aerial images in its current form** and must not be connected to the dashboard. Results are saved locally in `backend/reports/csrnet-public-benchmark/`; the dataset and report are excluded from version control.

### DM-Count comparison on DLR-ACD

The first DM-Count run used the wrong image preprocessing and is superseded. The authors' NWPU loader resizes images to its NWPU size range and normalizes RGB values with ImageNet means and standard deviations. The local evaluator now follows that preprocessing pattern and was rerun on the same 14 test images. The corrected result replaces the earlier report in `backend/reports/dmcount-public-benchmark/`.

| Measure | CSRNet | DM-Count |
| --- | ---: | ---: |
| Mean absolute error | 4,608.96 | 5,514.11 |
| Root mean squared error | 6,594.53 | 7,619.60 |
| Average signed error | -4,598.32 | -5,514.11 |

With the corrected inputs, DM-Count still **substantially undercounts** and its average errors are worse than CSRNet's on this small DLR test split. Neither checkpoint is suitable for these aerial images as-is; neither should feed the dashboard. This comparison does not validate performance on the product camera, Indian crowds, persons/m², or real-time operation.

### MRCNet availability check

The MRCNet paper is unusually relevant to this test because its authors created DLR-ACD and evaluated MRCNet on its 19-image training / 14-image test split. It describes the architecture, data preparation, and training settings, but the [DLR dataset page](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-acd) and paper do not link an author-published code repository or pretrained checkpoint. Therefore, a direct official-checkpoint comparison is not currently available.

I rechecked the image files and labels using the evaluator's exact rule (count pixels with value 255). All 14 test labels sum to 88,140, matching the paper. The 19 training labels sum to 138,196, while the paper reports 138,151; the entire 45-person difference is in I2 (downloaded archive annotation: 5,672; paper table: 5,627). I found the original downloaded DLR ZIP in `~/Downloads` and confirmed its I2 annotation is byte-for-byte identical to the extracted file. Its SHA-256 is `885a1ebb4913f284ac134088d4877ff186702632f1f42988d3f7ccc900d41476`.

The downloaded dataset's `README.txt` also says 16 training images, but its folder contains 19 and the MRCNet paper explicitly says 19 training / 14 test images. Treat the README split count as a typo. The 45-person discrepancy is between the current DLR archive and the paper's table, not an extraction error. For experiments on this archive, use its actual labels consistently and report that difference; do not alter labels to force a match to the paper. The held-out test labels are unaffected.

#### Local MRCNet reproduction work

An independent, paper-inspired implementation is now in `backend/app/mrcnet.py`. The training entry point is `backend/scripts/train_mrcnet.py`; the held-out evaluator is `backend/scripts/evaluate_mrcnet.py`. It uses the paper's dual-resolution density outputs, VGG16 encoder, DLR GSD-based Gaussian labels, 320-pixel tiles, 256-pixel crops, rotations/flips, Adam learning rate, and combined MSE loss. The script uses a smaller configurable batch and bounded steps per epoch for a CPU workstation. Because the paper does not publish its code or checkpoint, this is a reproduction candidate and may differ from the authors' exact implementation.

The VGG16 encoder was initialized from the official PyTorch [VGG16 checkpoint](https://download.pytorch.org/models/vgg16-397923af.pth), SHA-256 `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`.

It was evaluated on all 14 held-out test images using non-overlapping 256-pixel tiles. The report files are in `backend/reports/mrcnet-dlr-reproduction/` and are excluded from version control.

| Measure | CSRNet | DM-Count | MRCNet v1 | MRCNet v2 |
| --- | ---: | ---: | ---: | ---: |
| Mean absolute error (people/image) | 4,608.96 | 5,514.11 | 2,176.83 | 2,046.28 |
| Root mean squared error (people/image) | 6,594.53 | 7,619.60 | 2,869.41 | 2,833.41 |
| Mean signed error (people/image) | -4,598.32 | -5,514.11 | -1,343.69 | -1,478.30 |
| Mean normalized absolute error | 0.5810 | 0.8845 | 0.8419 | 0.7097 |

Version 1 used 10 epochs × 100 steps × batch size 2 (2,000 crop examples and 1,000 optimizer updates). It had 27.3 million parameters, more than the paper's 20.3 million. Version 2 narrows the decoder; it has 19.76 million parameters and used 10 epochs × 100 steps × batch size 4 (4,000 crop examples and 1,000 optimizer updates). Both are limited experiments, not full reproductions of the authors' training.

Version 2 modestly improved MAE and MNAE over version 1, but it still misses about 1,478 people per image on average. Its MNAE is 0.7097, meaning its average error relative to each image's true count is about 71%; CSRNet's was 0.5810. Version 2 is not accurate enough for a real crowd-safety reading and must stay disconnected from the dashboard. The authors report MAE 906.0 on DLR-ACD; that result came from their original implementation and training setup, not ours.

#### MRCNet error inspection

I inspected the highest-error test images and compared predictions tile by tile. The per-tile sums match the whole-image counts, so the large errors are not caused by adding overlapping tiles twice or skipping a tile.

| Test image | What the tile totals show |
| --- | --- |
| I14 | Tiles with 50+ labeled people contain 17,351 people; the model predicts 11,572 (about 5,779 missed). Empty tiles add about 133 false people. |
| I27 | Tiles with 50+ labeled people contain 13,706 people; the model predicts 8,172 (about 5,534 missed). Empty tiles add about 290 false people. |
| I10 | The whole image has 285 labeled people. Its 230 empty tiles alone receive about 738 predicted people, causing most of the overcount. |

The model misses dense areas and assigns some people to empty areas. Version 1's per-tile diagnostic is saved as `backend/reports/mrcnet-dlr-reproduction-v1/tile-diagnostics.csv`. Version 2's decoder is smaller and closer to the paper's reported parameter count, but it still is not an exact reconstruction because the authors' full source code is unavailable.

The test set has now been used to inspect and compare candidates, so it should not be used repeatedly to tune the next version. The next training change should be checked on a validation subset made from the training images; only then should we run another final test comparison. Any future report must continue to label this public-data work as research only.

#### Grouped validation experiment (version 3)

To check a training change without using the DLR test split, I held out the three I28-I30 images from the OAC 2016 event group and trained on the other 16 DLR training images. Training used density-band sampling (35% empty tiles, 15% sparse, 15% medium, 35% dense), 10 epochs, 100 steps per epoch, batch size 4, and seed 37. The checkpoint is `backend/models/mrcnet-dlr-validation-v3.pth` (ignored by version control).

On the three held-out event images, v3 produced MAE 4,654.08 people/image, RMSE 4,837.06, mean overcount +4,654.08, and MNAE 1.1051. It overcounted all three images, so this sampling adjustment has **not shown an improvement**. Results are in `backend/reports/mrcnet-validation-v3/`. This validation group is small and comes from one event, so it is an early warning, not a broad accuracy estimate. The v2 test scores cannot be compared directly with this validation score because v2 trained on the I28-I30 images.

Next: inspect the v3 errors and check the model implementation and density-target pipeline before another training run. Continue using event-grouped validation; do not choose changes by repeatedly checking the previously used test set. A final product check still needs separate, representative camera images.

Reproduce the training command from `backend/`:

```sh
PYTHONPATH=. .venv/bin/python scripts/train_mrcnet.py \
  --train-dir data/public/dlr-acd/DLR_AerialCrowdDataset/Train \
  --vgg16-weights models/vgg16-397923af.pth \
  --output models/mrcnet-dlr-reproduction-v2.pth \
  --epochs 10 --steps-per-epoch 100 --batch-size 4 --seed 29
```

Then evaluate on the current test split (which has already been used for exploratory candidate comparison, so this does not count as a fresh independent test):

```sh
PYTHONPATH=. .venv/bin/python scripts/evaluate_mrcnet.py \
  --test-dir data/public/dlr-acd/DLR_AerialCrowdDataset/Test \
  --weights models/mrcnet-dlr-reproduction-v2.pth \
  --output-dir reports/mrcnet-dlr-reproduction-v2
```

Compare its MAE, RMSE, signed bias, and MNAE against the existing CSRNet and DM-Count results. Do not connect it to the dashboard based on this small offline benchmark. Keep the dataset and resulting work within DLR's stated CC BY-NC-ND 3.0 terms; obtain permission before company/product use.

## Setup

1. Install the optional packages from the backend folder: `python -m pip install -r requirements-ai.txt`.
2. Extract DLR-ACD locally so the test split has `data/public/dlr-acd/DLR_AerialCrowdDataset/Test/Images/` and `data/public/dlr-acd/DLR_AerialCrowdDataset/Test/Annotation/`. The evaluator accepts this separate image/annotation layout and counts the white points in each PNG label.
3. Download the CSRNet ShanghaiTech Part B checkpoint from the [MIT-licensed model card](https://huggingface.co/rootstrap-org/crowd-counting) to `backend/models/csrnet-shanghai-b.pth`.
4. Run from `backend/`: `PYTHONPATH=. .venv/bin/python scripts/evaluate_csrnet.py --test-dir data/public/dlr-acd/DLR_AerialCrowdDataset/Test --weights models/csrnet-shanghai-b.pth`.
5. The script writes per-image predictions to `reports/csrnet-public-benchmark/predictions.csv` and aggregate metrics to `summary.json`.

To compare DM-Count using the same held-out DLR images, download the **NWPU pretrained checkpoint** from the [official DM-Count repository's pretrained-model link](https://github.com/cvlab-stonybrook/DM-Count#pretrained-models). Save it as `backend/models/dmcount-nwpu.pth` (rename the downloaded file if needed), then run from `backend/`:

```sh
PYTHONPATH=. .venv/bin/python scripts/evaluate_csrnet.py \
  --model dmcount \
  --test-dir data/public/dlr-acd/DLR_AerialCrowdDataset/Test \
  --weights models/dmcount-nwpu.pth
```

This writes to `reports/dmcount-public-benchmark/`. The evaluator uses NWPU-style image resizing and normalization, and does not train on the test images. Lower error would make a model the better *test candidate on these 14 DLR images only*; it would not establish accurate persons/m² readings or readiness for live cameras.

Use `--limit 10` for a short pipeline check before a full run. The optional dependencies pin the CPU-only PyTorch wheel for this workstation; this benchmark does not use a GPU.

## Limitations and safety

- Every count is an experimental estimate, never an officer instruction or confirmed fact.
- The public image benchmark is not representative proof for the product's intended camera placement or Indian crowds.
- This evaluator does not estimate density per square metre, movement, inflow/outflow, risk, or identities.
- The dashboard continues to use its synthetic scenarios until a separate review approves any model integration.
- Public benchmark data and outputs are for internal evaluation only and must not be shipped with the product.

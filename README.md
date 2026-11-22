# RSNA STR Pulmonary Embolism Detection

An end-to-end pipeline for the [RSNA STR Pulmonary Embolism Detection](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection) challenge (Kaggle, 2020): find pulmonary embolism on every slice of a CT pulmonary angiogram, and predict nine labels for each series. The project rebuilds a solution from scratch after the competition and scores it with late submissions. The notebooks are meant to be read in order.

## Results

| Submission | Private | Public | Notebook |
|---|---:|---:|---|
| Slice models alone (SE-ResNeXt101 + ConvNeXt Tiny, five folds) | 0.225 | 0.232 | 06 |
| Series models, fold 0 | 0.152 | 0.152 | 06 |
| **Series models, five folds** | **0.144** | **0.143** | 06 |

For scale, the private leaderboard of 784 teams: 1st 0.140, 2nd 0.148, 10th 0.162, 50th 0.209. These are late submissions, not ranks.

Local scores, out of fold over five folds (lower is better):

| Model | Test-like | Notebook |
|---|---:|---|
| LightGBM on metadata and pixel statistics, no image model | 0.3209 | 03 |
| Slice model (SE-ResNeXt101), series labels at the label rates | 0.2884 | 04 |
| Series model on SE-ResNeXt101 embeddings | 0.2054 | 05 |
| **Three series models (final)** | **0.1955** | 05 |

Test-like leaves out the series of two DICOM writers that hold 2 of the 650 visible test series. Every choice was made on the local scores before any submission.

**The leaderboard is lower than the local scores:** the training series were labelled by one radiologist, the test series by three with adjudication. The 1st place reported about 0.18 locally for its 0.140.

## Notebooks

| # | Notebook | Question |
|---|---|---|
| 00 | [overview](notebooks/00_overview.ipynb) | What are the task, the metric and the data? |
| 01 | [eda](notebooks/01_eda.ipynb) | What is in the data, and what does it mean for the models? |
| 02 | [data_preparation](notebooks/02_data_preparation.ipynb) | How are the slices ordered, split into folds and stored? |
| 03 | [metric_and_baselines](notebooks/03_metric_and_baselines.ipynb) | How is a validation score computed, and what is the score to beat without an image model? |
| 04 | [slice_models](notebooks/04_slice_models.ipynb) | Which slice model finds PE on one slice best? |
| 05 | [series_models](notebooks/05_series_models.ipynb) | How should a model read the whole series? |
| 06 | [submission](notebooks/06_submission.ipynb) | How is the test set predicted, and what does the leaderboard say? |

### The pipeline in short

- **Data:** the first axial stack of each series, its slices sorted along the slice normal, stored as 16-bit PNG in Hounsfield units (486 GB, lossless).
- **Slice models:** SE-ResNeXt101 (focal loss) and ConvNeXt Tiny (cross-entropy) see one slice through three windows (PE, mediastinal, lung), cut to the series' lung box and stretched to 512 px. One epoch of a two-epoch schedule, five folds.
- **Series models:** a BiGRU reads the slice features of a whole series and predicts every slice and the nine series labels, with the metric as its loss. Three are averaged: on SE-ResNeXt101 embeddings, on SE-ResNeXt101 logits with the neighbour differences, and on ConvNeXt Tiny embeddings.
- **Consistency:** a fix after averaging makes every prediction obey the label rules at 0.5, at the lowest weighted cost.
- **Test set:** on Kaggle, the slice models of fold k feed the series models of fold k, and the five folds are averaged before the fix.

## Compared with the top solutions

Shared with most of the top 10:
- Two stages: a slice CNN, then a sequence model over the slice features of the whole series, trained on in-sample features.
- From the 1st place: the lung-box crop, one epoch of training, SE-ResNeXt101, and the series model (BiGRU, [f, f − previous, f − next], max and attention pooling, the metric as the loss).
- A consistency fix after the models, so every submission obeys the label rules.

Where we differ:
- One slice through three windows at 512 px; the 1st place used three neighbouring slices through one window at 576 px.
- No model for RV/LV, the label with the largest weight; the 2nd place's heart model gained 0.006 on the public leaderboard.
- No 3D CNN and no test-time augmentation; one seed per series model.

## Project structure

```
configs/       paths.yaml (+ paths.local.yaml, not tracked)
notebooks/     00–06, read in order
scripts/       data preparation, benchmarks, training, feature extraction, slice-only calibration, test prediction
src/pe/        the package: data, metric, consistency fix, models, training, inference
tests/         pytest
models/        slice model weights (not tracked)
outputs/       runs, features, series runs, submission files (not tracked)
```

## Setup

Install PyTorch for your CUDA version first, then:

```bash
pip install -r requirements.txt
pip install -e .
```

The data is 980 GB of DICOM files. Download it with the Kaggle CLI (an API token from your Kaggle account in `~/.kaggle/kaggle.json`, and the competition rules accepted on Kaggle), then unzip it into `raw_data_dir`:

```bash
pip install kaggle
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
unzip rsna-str-pulmonary-embolism-detection.zip -d <raw_data_dir>
```

`raw_data_dir` then holds `train.csv`, `test.csv`, `sample_submission.csv`, `train/` and `test/`. Set `raw_data_dir` and `processed_data_dir` in `configs/paths.yaml`, or in `configs/paths.local.yaml` for one machine. The image store needs another 486 GB.

A smoke test on the first 1,000 files, into separate files (`*_limit1000`):

```bash
python scripts/extract_dicom_headers.py --limit 1000
python scripts/prepare_data.py --limit 1000
```

## Reproducing

```bash
# data (02)
python scripts/extract_dicom_headers.py
python scripts/prepare_data.py

# slice models (04): SE-ResNeXt101 and ConvNeXt Tiny, five folds
for k in 0 1 2 3 4; do
  python scripts/train_slice.py --name s11_seresnext101_f${k}_s1 --fold $k --backbone seresnext101 --size 512 --batch 32 --accumulate 4 \
    --channels=0 --windows pe,mediastinal,lung --crop lung --loss focal --epochs 2 --lr 0.00028 --stop-after 1
done
python scripts/train_slice.py --name s8e2_512_convnext_tiny_s1 --backbone convnext_tiny --size 512 --batch 32 --accumulate 4 \
  --channels=0 --windows pe,mediastinal,lung --crop lung --epochs 2 --lr 0.00028
for k in 1 2 3 4; do
  python scripts/train_slice.py --name s12_convnext_tiny_f${k}_s1 --fold $k --backbone convnext_tiny --size 512 --batch 32 --accumulate 4 \
    --channels=0 --windows pe,mediastinal,lung --crop lung --epochs 2 --lr 0.00028 --stop-after 1
done

# series models (05)
for k in 0 1 2 3 4; do
  python scripts/extract_features.py --fold $k
  python scripts/train_series.py --name s7_noneigh_f${k}_s1 --backbones seresnext101 --fold $k --no-neighbours
  python scripts/train_series.py --name s6_selogit_f${k}_s1 --backbones seresnext101 --fold $k --input logit
  python scripts/train_series.py --name s7_cnxemb_nn_f${k}_s1 --backbones convnext_tiny --fold $k --no-neighbours
done

# submission (06): calibration of the slice models, then outputs/submission/submission_final_5folds.csv
python scripts/fit_slice_ensemble.py
python scripts/predict_test.py --variant final_5folds
```

The ConvNeXt Tiny model of fold 0 trains both epochs; its first epoch is used, as for the other folds. The scores above come from the hidden test set, which only Kaggle has: there a notebook runs the steps of `scripts/predict_test.py` from a dataset with `src/`, the weights and `ensemble.json`. Locally the script predicts the series in `test/` and gives the others the label rates.

On one RTX 5090, after the data preparation: about 11 hours for the SE-ResNeXt101 models, 7 for ConvNeXt Tiny, 7 for the features and 1 for the series models. Reading the DICOM files once takes about 4 hours from a hard disk. The notebooks train no neural network; they read the saved runs and score them again.

## Tests

```bash
pytest
```

## License

The code is under the [MIT License](LICENSE). The competition data belongs to RSNA and is not included; see the competition's rules on Kaggle.

## Use of AI tools

Large language models (Claude, GPT) were consulted for code, reviews and text. All code and outputs were checked by the author.

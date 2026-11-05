# Generative modelling case study: GANs on points, retinal scans, network flows and sketches

All code, figures and numbers for the report come from one notebook, `gan_case_study.ipynb`.

| Station | Brief part | What it does |
|---|---|---|
| A | setup | configuration, seeds, device selection, shared DCGAN builder, FID/KID tools |
| B | Part 1 | tutorial sine GAN, seed sensitivity, seven-mode Gaussian mixture, three architecture variants |
| C | Part 2.1 | OCTMNIST DCGAN, Inception FID/KID with a real-vs-real floor, domain FID, class mix, AC-GAN extension |
| D | Part 2.2 | CICIDS 2017: all-day exploration, conditional WGAN-GP on Wednesday BENIGN + DoS, PCA/t-SNE, KS/Wasserstein, correlation gap, detector AUC, TSTR vs TRTR, all-days family extension |
| E | Part 2.3 | QuickDraw birthday cake DCGAN with FID tracked over epochs, star/house/cat extension and FID against sketch complexity |
| F | summary | headline metrics table |

## Running

```bash
pip install -r requirements.txt
# quick functional check (tiny epochs and subsets, numbers not meaningful)
GAN_SMOKE=1 python -m nbconvert --to notebook --execute --inplace gan_case_study.ipynb
# full run used for the report
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 gan_case_study.ipynb
```

On Colab: upload the notebook, put the CICIDS CSVs in `data/cicids/`, and choose Runtime > Run all. The device is picked automatically (CUDA, then Apple MPS, then CPU). Run times per station are written to `outputs/run_meta.json`. The full run takes about 1.5 hours on an Apple-silicon GPU; on a CPU-only runtime it is several times slower, so choose a GPU runtime on Colab. The first cell installs any missing package (torch-fidelity and torchmetrics are not preinstalled on Colab).

## Data

| Dataset | Source | How it is obtained |
|---|---|---|
| OCTMNIST (MedMNIST v2, 28x28) | https://medmnist.com/ | downloaded automatically from the MedMNIST Zenodo record into `data/medmnist/octmnist.npz` |
| CICIDS 2017 (8 daily CSVs) | https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset | `kaggle datasets download -d chethuhn/network-intrusion-dataset` (needs a Kaggle API key) or manual download, then unzip into `data/cicids/` |
| QuickDraw numpy bitmaps | https://github.com/googlecreativelab/quickdraw-dataset | downloaded automatically from `storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/` into `data/quickdraw/` |

`data/` is not committed (about 1.4 GB).

## Outputs

* `figures/`: every figure in the report (PNG).
* `outputs/`: metric JSON and CSV files per station, training logs, `summary.csv`, `run_meta.json`.
* `outputs/checkpoints/`: trained generator weights.

## Reproducibility

Seeds are fixed for Python, NumPy and torch at the start of every experiment. GPU kernels (CUDA or MPS) are not bit-for-bit deterministic, so values from a rerun on other hardware differ slightly from the saved outputs.

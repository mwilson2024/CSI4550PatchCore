# PatchCore project

Industrial defect detection using PatchCore. Includes image enhancement, mask processing and anomaly-map fusion experiments.

## Python

Developed using Python 3.14.8.

## Setup

Create a virtual environment from the project folder:

python -m venv .venv


Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1


Or on Linux:


source .venv/bin/activate


Install the packages:


python -m pip install -r requirements.txt


Download [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) separately and place its category folders in `datasets/MVTecAD/`.

## Run


python main.py


The menu has separate training and inference options. Choose all categories or one category. Train a model before running inference. For a user-supplied image, run baseline evaluation first to save its category's threshold.

## File structure

```text
main.py                  Main menu
config.py                Paths and default settings
train.py                 Build category models
prep.py                  Prepare image conditions
degradation.py           Darkness, blur and noise
enhancement.py           CLAHE, gamma and median filtering
eval.py                  Baseline evaluation and mask metrics
morphology.py            Erosion, dilation, opening and closing
fusion.py                Original and gamma map fusion
thresholding.py          Threshold selection helpers
calibrate_thresholds.py  Threshold calibration experiments
predict_image.py         Inference on a supplied image
plot_*.py                Result charts
modeltools.py            Checkpoint helpers
result_files.py          Combine category results
baseline.py              Standalone baseline demo
experiments.py           Image processing previews
run_experiment.py        Prediction inspection demo
requirements.txt         Python packages
testing/                 Tests
datasets/MVTecAD/        Downloaded dataset
experiments/datasets/    Prepared images
checkpoints/             Saved models
results/                 Scores, masks and charts
paperwork/               Presentation material
```



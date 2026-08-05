# MiniSEA: DeepSEA-Inspired DNase Accessibility Prediction

MiniSEA is a TensorFlow implementation of a DeepSEA-inspired convolutional neural network for predicting whether a 1,000-bp DNA sequence overlaps a DNase-seq accessibility peak.

The repository includes:

- hg38 sequence extraction
- ENCODE BED peak parsing
- non-overlapping negative sampling
- DNA one-hot encoding
- chromosome-held-out train/validation/test splits
- TensorFlow model training
- AUROC, AUPRC, precision, recall, and accuracy evaluation
- coordinate-based inference

## Scope

This project reproduces the general sequence-CNN approach used by DeepSEA, but it does **not** reproduce the original full multitask DeepSEA training setup.

MiniSEA currently treats one ENCODE DNase-seq experiment as a binary classification task:

- positive: a 1,000-bp sequence centered on a DNase-seq peak
- negative: a randomly sampled 1,000-bp sequence that does not overlap a positive interval

## Architecture

<img width="143" height="744" alt="image" src="https://github.com/user-attachments/assets/c363d331-8f7c-441d-97db-47da65446c63" />


## Repository structure

```text
deepsea-dnase-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   └── README.md
├── results/
│   └── README.md
├── scripts/
│   └── download_hg38.sh
├── src/
│   └── minisea/
│       ├── __init__.py
│       ├── encoding.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── predict.py
└── tests/
    ├── test_encoding.py
    └── test_data.py
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/mhuang24/deepsea-dnase-classifier.git
cd deepsea-dnase-classifier

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

## Data setup

Download and index hg38:

```bash
bash scripts/download_hg38.sh
```

Place your ENCODE BED file at:

```text
data/raw/ENCFF021AGH.bed
```

The raw data files are intentionally excluded from Git.

## Train

```bash
python -m minisea.train \
  --genome data/raw/hg38.fa \
  --peaks data/raw/ENCFF021AGH.bed \
  --output-dir outputs/run_001
```

Useful options:

```bash
python -m minisea.train --help
```

By default, MiniSEA uses chromosome-held-out evaluation:

- validation chromosome: `chr7`
- test chromosome: `chr8`
- all remaining available chromosomes: training

## Evaluate

Training automatically evaluates the best restored model on the held-out test set and writes:

```text
outputs/run_001/
├── model.keras
├── metrics.json
├── history.csv
├── loss_curve.png
├── accuracy_curve.png
├── roc_curve.png
├── precision_recall_curve.png
└── confusion_matrix.png
```

To evaluate a saved model against a saved dataset:

```bash
python -m minisea.evaluate \
  --model outputs/run_001/model.keras \
  --dataset outputs/run_001/test_data.npz \
  --output-dir outputs/run_001/evaluation
```

## Predict a genomic coordinate

```bash
python -m minisea.predict \
  --model outputs/run_001/model.keras \
  --genome data/raw/hg38.fa \
  --chrom chr1 \
  --position 700000
```

The position is interpreted as the center coordinate of the 1,000-bp input sequence.

## Testing

```bash
pytest
```

## Results
MiniSEA was evaluated using chromosome-held-out validation and testing on a **30,000-peak development subset** of ENCODE DNase-seq data. Early stopping selected the best model at **epoch 7**.

These results validate the complete training and evaluation pipeline. A full-dataset benchmark is planned as future work.

| Metric | Result |
|--------|-------:|
| Accuracy | 0.829 |
| AUROC | 0.912 |
| AUPRC | 0.917 |
| Precision | 0.834 |
| Recall | 0.821 |

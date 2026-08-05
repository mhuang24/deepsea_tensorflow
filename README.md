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

Input: `1000 x 4` one-hot encoded DNA sequence

1. Conv1D: 320 filters, kernel size 8, ReLU
2. MaxPooling1D: pool size 4
3. Dropout: 0.20
4. Conv1D: 480 filters, kernel size 8, ReLU
5. MaxPooling1D: pool size 4
6. Dropout: 0.20
7. Conv1D: 960 filters, kernel size 8, ReLU
8. Flatten
9. Dense: 925 units, ReLU
10. Dropout: 0.50
11. Dense: 1 unit, sigmoid

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
git clone https://github.com/YOUR_USERNAME/deepsea-dnase-classifier.git
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

Add your final held-out chromosome results here after running the cleaned pipeline.

| Metric | Result |
|---|---:|
| Accuracy | TBD |
| AUROC | TBD |
| AUPRC | TBD |
| Precision | TBD |
| Recall | TBD |

Do not carry over the earlier random-split accuracy as the final result without rerunning the chromosome-held-out pipeline.


- Minerva credentials
- internal cluster paths
- proprietary lab code
- model checkpoints trained on restricted data

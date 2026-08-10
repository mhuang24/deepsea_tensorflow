# Data directory

Create the following local structure:

```text
data/
└── raw/
    ├── hg38.fa
    ├── hg38.fa.fai
    └── ENCFF021AGH.bed
```

Raw genomic data is excluded from version control.

## hg38

Run:

```bash
bash scripts/download_hg38.sh
```

## ENCODE BED file
https://www.encodeproject.org/pipelines/ENCPL336PGV/
Download the appropriate public ENCODE DNase-seq BED file and place it in `data/raw/`.

Confirm that the accession and biological context described in the main README match the file you actually use.

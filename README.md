# HiMaCon
Repository for NeurIPS 2025 Paper **HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data**


[![arXiv](https://img.shields.io/badge/arXiv-2510.11321-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2510.11321)

###  Installation & Setup
```
conda env create -f environment.yml
conda activate himacon
```

### Example Usage on LIBERO

- Preprocess Data: First download the data following the [download instruction](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html) from [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/overview.html). Then adjust `--libero_org_path` in `preprocess.sh` to the downloaded dataset path and run:
```
bash preprocess.sh
```

# HiMaCon
Repository for NeurIPS 2025 Paper **HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data**


[![arXiv](https://img.shields.io/badge/arXiv-2510.11321-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2510.11321)

---
###  Installation & Setu
```
conda env create -f environment.yml
conda activate himacon
```

---
### Example Usage on LIBERO

First, download the data following the [download instruction](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html) from [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/overview.html). Then use the following script to preprocess the data:
```
bash preprocess.sh
```
Parameters for `preprocess.sh`:
- `--libero_org_path`: Path where you downloaded the LIBERO dataset.
- `--libero_tar_path`: Path where the preprocessed LIBERO dataset will be saved.
- `--chunk_size`: Number of image frames to send to the encoder for preprocessing in each iteration.



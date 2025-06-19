# Heads up! Large Language Models Can Perform Tasks Without Your Instruction via Selective Attention Head Masking

![](https://icml.cc/media/PosterPDFs/ICML%202025/43598.png)

This repository provides code for training attention head masks and for plotting some of the figures presented in our paper ***Heads up! Large Language Models Can Perform Tasks Without Your Instruction via Selective Attention Head Masking*** (ICML'25).

## Environment
```bash
conda create -n headsup python=3.10 -y
conda activate headsup
pip install -r requirements.txt
```
We use [FlashAttention](https://github.com/Dao-AILab/flash-attention) for efficient training. You may install it as your need, or disable FlashAttention in [`train_mask.py`](train_mask.py).

## Download Attention Head Masks
Trained head mask for Meta-Llama-3.1-8B-Instruct on XNLI and FV datasets are available [here](https://drive.google.com/drive/folders/1tysu3InFFQC9xhCRhOcZCDl1WMYVg-7E?usp=sharing) (Google Drive). Put the `output` folder under this directory, then you can directly run the cells in [`eval.ipynb`](eval.ipynb) and partial cells in [`playground.ipynb`](playground.ipynb).

## Train Attention Head Masks
We provide the training scripts under `scripts/` directory. You may modify them to your own training settings.

```bash
bash scripts/llama_xnli.sh      # Train llama-3.1 on XNLI dataset
```

## Citation
```bibtex
coming soon
```

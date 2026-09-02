# LLMs from scratch

A hands-on implementation of a GPT-style language model built and trained from scratch with PyTorch, without relying on a pre-built transformer library.

## What's in this repository

The repository has two sections: **`LLMs_Build_and_Train`**, which pre-trains a small GPT from scratch, and **`LLMs_Alignment`**, which aligns a larger pretrained model on preference data.

### `LLMs_Build_and_Train/`

| File | Description |
|---|---|
| [`Train_Tokenizer.ipynb`](LLMs_Build_and_Train/Train_Tokenizer.ipynb) | Trains a Byte Pair Encoding (BPE) tokenizer with [SentencePiece](https://github.com/google/sentencepiece) on a text corpus (`wiki.txt`), producing a `.model`/`.vocab` pair. |
| [`LLM_Scratch_Simple.ipynb`](LLMs_Build_and_Train/LLM_Scratch_Simple.ipynb) | Implements and trains a decoder-only GPT-style transformer from scratch (embeddings, positional embeddings, causal multi-head self-attention, feed-forward blocks, layer norm, residual connections), then runs an interactive inference loop with the trained model. |
| `requirements.txt` | Python dependencies for both notebooks. |

The model implemented in `LLM_Scratch_Simple.ipynb` is a small GPT with:

- **Architecture:** 7 transformer blocks, 7 attention heads per block, embedding size 384, context window of 512 tokens — about **19.8M parameters**.
- **Training:** AdamW optimizer with cosine-annealed learning rate, gradient clipping, dropout regularization, checkpointing to `models/`, and optional [Weights & Biases](https://wandb.ai) logging.
- **Data:** a small excerpt of English Wikipedia (`wiki.txt`), tokenized with a SentencePiece BPE tokenizer trained by `Train_Tokenizer.ipynb`.

Not included in the repository (excluded via `.gitignore` — either too large for GitHub's 100MB file-size limit, or kept local-only by choice): the raw `wiki.txt` corpus, the tokenized `encoded_data.pt`, trained checkpoints under `models/`, the tokenizer artifacts (`wiki_tokenizer*.model` / `.vocab`), and `wandb/` run logs. These are all generated locally when you run the notebooks, as described below.

### `LLMs_Alignment/`

| File | Description |
|---|---|
| [`Align_LLM.ipynb`](LLMs_Alignment/Align_LLM.ipynb) | Aligns a pretrained LLM with [ORPO](https://arxiv.org/html/2403.07691) (odds-ratio preference optimization): tokenizes the [`orpo-dpo-mix-40k`](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) preference dataset, then fine-tunes the model so it favors each pair's chosen answer over the rejected one. |
| [`llm.py`](LLMs_Alignment/llm.py) | A LLaMA-style decoder-only architecture (RMSNorm, rotary positional embeddings, grouped-query attention) — more sophisticated than `LLM_Scratch_Simple.ipynb`'s — used to load and align the pretrained base checkpoint. |
| `requirements.txt` | Python dependencies for the notebook. |

The model aligned in `Align_LLM.ipynb` is a bigger, separately pretrained checkpoint:

- **Base model:** ~138M-parameter LLaMA-style transformer (RMSNorm, RoPE, grouped-query attention), pretrained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — the checkpoint is provided separately rather than trained in this repo.
- **Alignment:** ORPO, which combines the usual next-token cross-entropy loss with an odds-ratio term (scaled by `alpha`) that pushes the model towards each pair's chosen answer and away from the rejected one.
- **Data:** the `orpo-dpo-mix-40k` preference dataset (chosen/rejected answer pairs), tokenized and cached to `data/`.

Not included in the repository (same reasons as above): the base and aligned checkpoints under `models/`, the tokenizer under `tokenizers/`, the tokenized dataset cache under `data/`, and `wandb/` run logs. Licenses for the third-party code/data reused here (LLaMA2.c, ORPO, `orpo-dpo-mix-40k`) are in [`readme.txt`](LLMs_Alignment/readme.txt).

## Installation

### Prerequisites

- Python 3.10+
- (Optional but recommended) an NVIDIA GPU with CUDA for reasonable training speed — both notebooks fall back to CPU automatically if none is available, though alignment is more GPU-hungry given the larger base model.
- A free [Weights & Biases](https://wandb.ai) account if you want training metrics logged (optional — see below).

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/oAndreAmaral/LLMs-from-scratch.git
   cd LLMs-from-scratch
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. **Install PyTorch** for your platform/GPU first, following the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (e.g. a CUDA build if you have an NVIDIA GPU, otherwise the CPU build). This isn't pinned in either `requirements.txt` because the correct wheel depends on your OS/CUDA version.

4. **Install the dependencies** for whichever section you want to run:

   ```bash
   pip install -r LLMs_Build_and_Train/requirements.txt requests
   pip install -r LLMs_Alignment/requirements.txt
   ```

## Running

### `LLMs_Build_and_Train`

1. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

2. **Get a tokenizer** — the tokenizer files aren't in the repo, so either:
   - run `Train_Tokenizer.ipynb` to train your own `wiki_tokenizer_test.model` / `.vocab` on a local `wiki.txt` corpus, or
   - let the next step download a pretrained one automatically (see below).

3. **Train and/or run the model** — open `LLM_Scratch_Simple.ipynb`:
   - The second cell downloads the sample dataset and tokenizer files (`wiki.txt`, `wiki_tokenizer.model` / `.vocab`, `encoded_data.pt`) automatically if they aren't already present locally.
   - Review the **architecture / hyperparameter / training** parameter cells near the top and adjust as needed (e.g. `batch_size` if you have less GPU memory, `train_iters` for how long to train).
   - By default `wandb_log = True`; either run `wandb login` beforehand (or paste your API key when prompted) to log metrics, or set `wandb_log = False` in the notebook to skip it.
   - By default `load_pretrained = True`, so training resumes from `models/latest.pt` if a checkpoint already exists there.
   - Run all cells to train. Checkpoints are saved to `models/` whenever validation loss improves.
   - The final cell starts an interactive loop — type a prompt and press Enter to get a completion from the model, or type `q` to quit.

### `LLMs_Alignment`

1. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

2. **Get the base checkpoint and tokenizer** — unlike `LLM_Scratch_Simple.ipynb`, `Align_LLM.ipynb` doesn't download these automatically, so place the pretrained checkpoint at `models/base_model.pt` and the tokenizer under `tokenizers/tok16384/` before running.

3. **Align the model** — open `Align_LLM.ipynb`:
   - Review the **training / hyperparameter** parameter cells near the top and adjust as needed (e.g. `batch_size`, `alpha`, `epochs`).
   - The dataset cell tokenizes [`orpo-dpo-mix-40k`](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) the first time and caches it to `data/orpo_dataset2/`; later runs load the cached version.
   - By default `wandb_log = True`; either run `wandb login` beforehand (or paste your API key when prompted) to log metrics, or set `wandb_log = False` in the notebook to skip it.
   - Run all cells to align the model. The aligned checkpoint is saved to `models/` at the end of training.

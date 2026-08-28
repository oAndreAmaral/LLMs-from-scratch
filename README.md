# LLMs from scratch

A hands-on implementation of a GPT-style language model built and trained from scratch with PyTorch, without relying on a pre-built transformer library.

## What's in this repository

Right now the repository only contains the **`LLMs_Build_and_Train`** section (more sections will be added later).

### `LLMs_Build_and_Train/`

| File | Description |
|---|---|
| [`Train_Tokenizer.ipynb`](LLMs_Build_and_Train/Train_Tokenizer.ipynb) | Trains a Byte Pair Encoding (BPE) tokenizer with [SentencePiece](https://github.com/google/sentencepiece) on a text corpus (`wiki.txt`), producing a `.model`/`.vocab` pair. |
| [`LLM_Scratch_Simple.ipynb`](LLMs_Build_and_Train/LLM_Scratch_Simple.ipynb) | Implements and trains a decoder-only GPT-style transformer from scratch (embeddings, positional embeddings, causal multi-head self-attention, feed-forward blocks, layer norm, residual connections), then runs an interactive inference loop with the trained model. |
| `requirements.txt` | Python dependencies for both notebooks. |
| `wiki_tokenizer.model` / `wiki_tokenizer.vocab` | Pretrained tokenizer produced by `Train_Tokenizer.ipynb` (vocab size 4096), ready to use with the training notebook. |
| `wiki_tokenizer_test.model` / `wiki_tokenizer_test.vocab` | A secondary/test tokenizer artifact from an earlier training run. |

The model implemented in `LLM_Scratch_Simple.ipynb` is a small GPT with:

- **Architecture:** 7 transformer blocks, 7 attention heads per block, embedding size 384, context window of 512 tokens — about **19.8M parameters**.
- **Training:** AdamW optimizer with cosine-annealed learning rate, gradient clipping, dropout regularization, checkpointing to `models/`, and optional [Weights & Biases](https://wandb.ai) logging.
- **Data:** a small excerpt of English Wikipedia (`wiki.txt`), tokenized with the SentencePiece BPE tokenizer above.

Not included in the repository (excluded via `.gitignore` because they exceed GitHub's 100MB file-size limit, or are just local run artifacts): the raw `wiki.txt` corpus, the tokenized `encoded_data.pt`, trained checkpoints under `models/`, and `wandb/` run logs. These are generated locally when you run the notebooks, as described below.

## Installation

### Prerequisites

- Python 3.10+
- (Optional but recommended) an NVIDIA GPU with CUDA for reasonable training speed — the notebook falls back to CPU automatically if none is available.
- A free [Weights & Biases](https://wandb.ai) account if you want training metrics logged (optional — see below).

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/oAndreAmaral/LLMs-from-scratch.git
   cd LLMs-from-scratch/LLMs_Build_and_Train
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. **Install PyTorch** for your platform/GPU first, following the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (e.g. a CUDA build if you have an NVIDIA GPU, otherwise the CPU build). This isn't pinned in `requirements.txt` because the correct wheel depends on your OS/CUDA version.

4. **Install the remaining dependencies**

   ```bash
   pip install -r requirements.txt requests
   ```

## Running

1. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

2. **(Optional) Train your own tokenizer** — open `Train_Tokenizer.ipynb` and run it if you want to train a new tokenizer on your own `wiki.txt` corpus. Otherwise, skip this: the repository already ships a pretrained `wiki_tokenizer.model` / `wiki_tokenizer.vocab`.

3. **Train and/or run the model** — open `LLM_Scratch_Simple.ipynb`:
   - The second cell downloads the sample dataset and tokenizer files (`wiki.txt`, tokenizer, `encoded_data.pt`) automatically if they aren't already present locally.
   - Review the **architecture / hyperparameter / training** parameter cells near the top and adjust as needed (e.g. `batch_size` if you have less GPU memory, `train_iters` for how long to train).
   - By default `wandb_log = True`; either run `wandb login` beforehand (or paste your API key when prompted) to log metrics, or set `wandb_log = False` in the notebook to skip it.
   - By default `load_pretrained = True`, so training resumes from `models/latest.pt` if a checkpoint already exists there.
   - Run all cells to train. Checkpoints are saved to `models/` whenever validation loss improves.
   - The final cell starts an interactive loop — type a prompt and press Enter to get a completion from the model, or type `q` to quit.

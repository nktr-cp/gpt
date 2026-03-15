# GPT

Small, learning-oriented GPT implementation in Python 3.13 with `uv`.

## Setup

```bash
uv sync --dev
```

## Download dataset

The initial experiments use Karpathy's `names.txt` dataset.

```bash
uv run python -m gpt download-names
```

This writes the file into `data/raw/`.

## Train a tiny model

```bash
uv run python -m gpt train \
  --num-steps 200 \
  --log-interval 20 \
  --block-size 32 \
  --batch-size 32 \
  --n-layer 2 \
  --n-head 4 \
  --n-embd 64 \
  --prompt a \
  --max-new-tokens 24
```

The command prints training loss and a sampled continuation at the end.

To switch the MLP block to SwiGLU instead of the baseline GELU feed-forward, add
`--mlp-variant swiglu`.

## Current scope

- character-level tokenizer
- next-token batch sampling
- learned token and positional embeddings
- single-head and multi-head causal self-attention
- configurable GELU or SwiGLU feed-forward network, residual connections, and RMSNorm
- decoder-only GPT model with LM head
- minimal training loop and autoregressive generation

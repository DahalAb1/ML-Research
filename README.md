# ML Research

Anyone can import a library and call `.fit()`. What I actually want is to know
what happens underneath — well enough to explain it in my own words. So this repo
is my running log of learning machine learning: I pick something I don't fully
understand yet, chase it until I do, and write down what I found while it's still
fresh.

I'm doing this under the guidance of my professor.

## How to read this

It's a learning log, so it grows topic by topic rather than following a fixed
syllabus. Each topic usually starts from a question I was stuck on — *can a model
recover a hidden slope from noisy data? what breaks when the data is real and
messy? why does my loss spike late in training?* — and the notebooks are me
working that question out.

There's no single recipe I force every topic through. A habit I lean on a lot is
building something **by hand first** (raw tensors, gradients I derive on paper, a
training loop I write step by step) and then **rebuilding it with the high-level
tools** (`nn.Module`, `torch.optim`, built-in losses), because seeing the two side
by side is what makes the library stop feeling like magic. But that's a tool, not
a rule — some topics will be a real dataset, some a paper I worked through, some
just notes. Whatever makes the idea click.

## Why PyTorch

It's what most of research and industry actually use, so the mechanics carry over
to real work and to reading other people's code. It also scales far past these
small examples — autograd over millions of parameters, GPUs, ready-made layers —
which is exactly why I want to understand the low-level version before leaning on
the shortcuts.

## What's inside so far

Each folder has its own README with the full, honest walkthrough. Start here, then
open a folder to go deep.

| Folder | The question it chases | What's inside |
|---|---|---|
| [`Regression/`](Regression/README.md) | How does a model *learn* a line, and what changes on real data? | linear regression by hand, the same thing in PyTorch, then multi-feature regression on California housing |
| [`gradient-descent/`](gradient-descent/README.md) | How do you train when you can't afford to look at all the data each step? | SGD derived and coded by hand (single & multi-feature), the learning-rate schedule, then the PyTorch version |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Notebooks can be exported to PDF (saved under `pdfs/`) with:

```bash
python convert_to_pdf.py
```

# Kernel IR → PTX

A compiler for the Emmy Hackathon 2026. It reads one Kernel IR JSON file and writes PTX to stdout.

## Run

Python 3.12+. From the repository root:

```bash
pip install -r requirements.txt
python compile.py path/to/kernel.ir.json > out.ptx
```

Check an open problem locally:

```bash
python submit.py p00_scalar_add --compiler compile.py
```

Open set: [`problems/`](problems/)

## Docs

| Doc | What it is |
|---|---|
| [For reviewers](docs/FOR_REVIEWERS.md) | Run commands, open-set checks under [`tests/`](tests/), current scope |
| [Architecture](docs/architecture.md) | How `compile.py` is structured |
| [References](docs/references.md) | Materials used before and during the hackathon |
| [`docs/categories/`](docs/categories/) | Step-by-step notes about my approach to each problem set, from easy to hard.<br>[Pointwise](docs/categories/01-pointwise.md)<br>[Reductions](docs/categories/02-reductions.md)<br>[Matmul](docs/categories/03-matmul.md)<br>[Fused](docs/categories/04-fused.md)<br>[Attention-like](docs/categories/05-attention.md) |

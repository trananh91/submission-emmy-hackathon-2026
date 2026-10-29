# For reviewers

## Run the compiler

Python 3.12+. From the repository root:

```bash
pip install -r requirements.txt
python compile.py path/to/kernel.ir.json > out.ptx
```

## Check the open set

One problem, via [`submit.py`](../submit.py). The problem name is a folder under [`problems/`](../problems/):

```bash
python submit.py p00_scalar_add --compiler compile.py
```

All of category 1, from the repository root ([`01_pointwise.sh`](../tests/by_categories/01_pointwise.sh)):

```bash
bash tests/by_categories/01_pointwise.sh
```

Each problem prints a header, then a register count and `PASS` or `FAIL`. The last line is how many problems passed. For example with category 1:

```text
---- p00_scalar_add ----
[k_add_pointwise] 16 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (20.288 us)
---- p01_vector_add ----
[k_add_pointwise] 16 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (25.504 us)
---- p02_scale_shift ----
[k_add_pointwise] 22 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (15.552 us)
---- p03_relu_mul ----
[k_mul_pointwise] 26 registers
[k_mul_pointwise] PASS  mul: max abs err 0.000e+00  (15.2 us)

=== category 1 (pointwise): 4/4 passed ===
```

## Additional Read about the submission

1. [Architecture](architecture.md)
2. Categories, in order: [pointwise](categories/01-pointwise.md), [reductions](categories/02-reductions.md), [matmul](categories/03-matmul.md), [fused (TBD)](categories/04-fused.md), [attention-like (TBD)](categories/05-attention.md)
3. [References](references.md)

## Scope

Category 1 (pointwise) passes: `bash tests/by_categories/01_pointwise.sh` reports `4/4 passed`.

Category 2 (reductions) passes: `bash tests/by_categories/02_reductions.sh` reports `4/4 passed`.

Category 3 (matmul) passes: `bash tests/by_categories/03_matmul.sh` reports `2/2 passed`.

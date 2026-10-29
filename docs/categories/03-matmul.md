# Step 3 — Matmul

**Builds on:** [02 — Reductions](02-reductions.md)
**Open problems:** [`m01_matmul_512`](../../problems/m01_matmul_512), [`m02_matmul_rect`](../../problems/m02_matmul_rect)

## What this step establishes

Category 2 is the foundation. This step keeps that compiler: the same header, entry, registers, prologue, and `ret`, the same `Load` / `Assign` / `Write` walk, and the same reduction statements. It only adds lowering for the statement kinds these kernels introduce, and the index operators their expressions name. There is no category branch.

`m01` and `m02` share `Init` of `add` accumulators at `0.0`, 16-byte `cp.async` copies with `swizzle='NONE'`, a `Loop` over `_ki`, and `multiply` then `Accum` `add`. The K loop `_ks` is the existing `StridedLoop`. Each problem changes the shared placement or the store.

| Problem | Launch | What it adds |
|---|---|---|
| `m01` | grid 128, block 512, tile 65536, `smem_bytes` 73728 | Four `Init`. Dynamic pool: `_a_smem` at 0, `_b_smem` at 24576. Nine `CpAsyncCopy`, three commits, `CpAsyncWait` group 2. `StridedLoop` `_ks` step 64. `Loop` `_ki` to 64. `TernaryExpr` indexes. Two `v2` stores. |
| `m02` | grid 512, block 256, tile 131072, `smem_bytes` 0 | Two `Init`. Static `_a_smem` and `_b_smem` at alignment 16. Copy loops whose start is `(a0_u * 32) + a1_u`. Two copies, wait group 0. `Loop` `_ki` to 8. Two scalar atomic stores. |

`multiply` is the `mul.f32` already emitted. The inner product is that multiply followed by `add.f32` into the `Init` register. `-`, `/`, and `%` in an index are `sub.s32`, `div.s32`, and `rem.s32`. `m01`'s `(_ks + 128 < 512) ? (_ks + 128) : 448` is `setp.lt.s32` and `selp.b32`. The [architecture](../architecture.md) is this pass.

## How I approached it (step-by-step)

1. Started from the category 2 compiler. `compile_index` gained `-`, `/`, `%`, and `TernaryExpr`. `compile_pred` gained `<`. No kernel is selected by name.
2. `lower_smem` reads `smem_bytes` from the same `meta.json` record as `arg_order`. Zero stays a static buffer. An IR alignment of 0 stays 4. A non-zero size is one extern pool, packed in IR order.
3. The matmul loads leave `dtype` unset. The element size is 4, and the shared load stays `ld.shared.f32`.
4. `Init` binds each accumulator to `0f00000000` before either loop. `Loop` lowers `_ki` from 0 with step 1 and does not scan for `Accum`. `_ks`, and `m02`'s copy loops, stay `StridedLoop`.
5. `CpAsyncCopy`, `CpAsyncCommit`, and `CpAsyncWait`. Sixteen bytes is `cp.async.cg.shared.global`. `lane_index`, when present, is added in the shared buffer's own shape. `lane_rows` is not that stride.
6. Submitted `m01`. The pool, the four inits, the copy pipeline, `Loop`, and two `st.global.v2.f32` stores.
7. Submitted `m02`. Static alignment 16, the strided copy loops, wait group 0, and two `red.global.add.f32` stores. `zero_outputs` stays with the runner.
8. Re-submitted `p00` through `r04`. Those kernels have `smem_bytes` 0 and shared alignment 0, and they never contain the new statements, so they stay on the previous path.

## Gotchas

- `smem_bytes` chooses the placement. `m01` is one `.extern .shared` pool of 73728 bytes. `m02` and every category 2 kernel are static, because that field is 0.
- An IR alignment of 0 still prints `.align 4`, as `acc0_smem` does. `m02` says 16, so its two buffers print `.align 16`.
- The identity move is the `Init`, and it sits before `_ks`. The outer `StridedLoop` does not zero again. A zero inside `_ki` would keep only the last product.
- `lane_rows` is 32 on `m01` and 0 on `m02`. The shared row stride is the buffer's trailing extent: 64 for `m01`'s `_a_smem`, 8 for `m02`'s.
- `m02`'s `zero_outputs` lists `matmul`. `submit.py` sends that to the runner. The compiler emits the atomic add and does not fill zeros.
- The tile guard wraps the whole body, and the body contains `bar.sync`. Both launches have one thread per tile cell, so the guard does not skip the barrier.
- `lower_tile` is unchanged. The last axis is fastest. A `Window` parent is not the thread index. The global row and column are the index expressions.

## References used for this step

- [Architecture](../architecture.md)
- [References](../references.md)

**Next:** [04 — Fused](04-fused.md)

# Step 2 — Reductions

**Builds on:** [01 — Pointwise](01-pointwise.md)
**Open problems:** [`r01_array_sum`](../../problems/r01_array_sum), [`r02_big_sum`](../../problems/r02_big_sum), [`r03_mean`](../../problems/r03_mean), [`r04_row_sum`](../../problems/r04_row_sum)

## What this step establishes

Category 1 is the foundation. This step keeps that compiler: the same header, entry, registers, prologue, and `ret`, and the same `Load` / `Assign` / `Write` walk. It only adds lowering for the statement kinds these kernels introduce, and an op when one of them names it. There is no category branch.

`r01` through `r04` share one `add` accumulator, a warp shuffle of length 32, four warps of shared memory, and a tree of length 4. Each problem changes a number or a statement the previous one did not force. Lengths, the stride, and the row shape come from that kernel's IR and `meta.json`. `divide` is added because `r03` has it.

| Problem | Launch | What it adds |
|---|---|---|
| `r01` | grid 1, block 128, 1024 elements | `StridedLoop` from `tid` with step 128. `acc0 <- add`. Butterfly, lane 0 writes `acc0_smem[warp]`, `Sync`, `TreeHalve`, one scalar store. |
| `r02` | same launch, 65536 elements | The same statements. The axis extent in the IR is the only change. |
| `r03` | same launch, 32768 elements | After the tree, `divide` the sum by `mean_keepdim_count[0]`. |
| `r04` | grid 64, block 128, `x` is `(64, 512)` | Two tile axes. The load index is `x[a0, a1]`. The store is `sum_1[a0]`. |

`add` as a reduction is `add.f32` into one register, starting from `0f00000000`. The shuffle offsets are 16, 8, 4, 2, 1. The tree spans are 2 then 1. `divide` is `div.rn.f32`, the same `Assign` path the pointwise ops use. The [architecture](../architecture.md) is this pass.

## How I approached it (step-by-step)

1. Started from the category 1 compiler. New kinds were added to `lower_stmt`: `StridedLoop`, `Accum`, `WarpShuffle`, `Smem`, `Cond`, `Sync`, and `TreeHalve`. No kernel is selected by name.
2. Lowered `StridedLoop` and `Accum` on `r01`. The identity move is emitted while scanning the loop body, before the header. The loop index is a copy of the start, so the tile axis stays intact. The bound and the step are the IR extent and the IR step.
3. Lowered `WarpShuffle`, `Smem`, `Cond`, `Sync`, and `TreeHalve` in the order the IR writes them. Lane 0 stores the warp sum. The tree folds the four shared slots and broadcasts slot 0.
4. Submitted `r01`, then `r02`. The second kernel is the same PTX shape with a larger extent from the IR.
5. Submitted `r03`. One new row in the elementwise table, `divide`, and a load of a one-element pointer argument.
6. Submitted `r04`. `lower_tile` decodes every axis, last axis fastest. A two-index load uses the buffer shape from the graph, so the address stride is 512.
7. Re-submitted `p00` through `p03`. Those kernels never contain the new statements, so they still take the category 1 path.

## Gotchas

- The identity move sits before the loop. Emitting it inside the body would clear the sum on every iteration.
- After the five butterfly steps every lane holds the warp sum. The IR still stores only when `lane == 0`.
- Shuffle length 32 and tree length 4 are different numbers. 128 threads make four warps. The tree combines those four partials.
- `meta.json` reports `smem_bytes` 0. The buffer is still static: `.shared .align 4 .b8 acc0_smem[16]`, four floats.
- The tile guard wraps the whole body, and the body contains `bar.sync`. These four launches have one thread per tile cell (the product of the axis extents), so the guard does not skip the barrier.
- `r04` loads `x[a0, a1]` with shape `(64, 512)`. The stride is 512, the row length. The tile axis `a1_co` has extent 128, which is the thread count along the row, not the address stride.
- `lane` and `warp` are computed from `%tid.x` on first use. Pointwise kernels never name them, so their PTX does not gain `and.b32` or `shr.u32`.
- This pass lowers one `add` accumulator.
- The sum order is the IR's order. The runner compares with `rtol` and `atol` from `meta.json`.

## References used for this step

- [Architecture](../architecture.md)
- [References](../references.md)

**Next:** [03 — Matmul](03-matmul.md)

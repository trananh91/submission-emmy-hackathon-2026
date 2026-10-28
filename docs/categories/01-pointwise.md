# Step 1 — Pointwise

**Open problems:** [`p00_scalar_add`](../../problems/p00_scalar_add), [`p01_vector_add`](../../problems/p01_vector_add), [`p02_scale_shift`](../../problems/p02_scale_shift), [`p03_relu_mul`](../../problems/p03_relu_mul)

## What this step establishes

One 1-D `Tile`: `Load`, elementwise `Assign`, `Write`. Shapes, the grid, and the op names come from the IR and the sibling `meta.json`. The four kernels share that walker. Each one adds a fact the previous kernel did not force.

| Problem | Launch | Tile | What it adds |
|---|---|---|---|
| `p00` | grid 1, block 256 | 64 | Scalar loads, `add`, one `v2` store. 256 threads, 64 workers, 128 elements. |
| `p01` | grid 2, block 256 | 512 | Same body on two blocks. The axis is `ctaid * ntid + tid`. |
| `p02` | grid 4, block 256 | 1024 | `multiply` then `add`. `2.0` and `1.0` are the pointer args `mul_c1` and `add_c1`. One `v4` store. |
| `p03` | grid 2, block 256 | 512 | `relu` then `multiply`. One `v4` store. |

`add` → `add.f32`, `multiply` → `mul.f32`, `relu` → `max.f32` against zero. A `Load` or `Write` of 2 or 4 names is one `v2` or `v4` instruction. These four IRs only vectorize the store. The load path is in the compiler anyway.

The open set passed. Another elementwise name is refused until a later category's IR uses it. The [architecture](../architecture.md) is this pass.

## How I approached it (step-by-step)

1. Read [TASK.md](../../TASK.md), [`problems/p00_scalar_add`](../../problems/p00_scalar_add), and [`examples/`](../../examples) to split the two inputs. `meta.json` supplies the entry match, `arg_order`, and the grid and block. The IR supplies the tile, the axis, and the body: loads, ops, and how stores are grouped. Op names come from the IR, not the launch file.
2. Learned the six parts of the PTX file, then implemented them on `p00` alone. Header, entry, register banks, prologue, body, `ret`.
3. Took the entry name from `KernelOp.name` and the parameter order from `meta.json` `arg_order`. Each parameter is a `.u64` pointer. `ld.param.u64` plus `cvta.to.global.u64` stores the address.
4. Computed `gid` and bound the single tile axis to it. A thread with `gid` past the axis extent branches to `DONE`. That predicate is not in the IR.
5. Walked the body in order. Each SSA name gets a fresh register. An index is `+` and `*` on that axis. The address is `index * sizeof(element) + base`.
6. Submitted `p00`, then `p01`. The second block is what made raw `%tid.x` wrong.
7. Submitted `p02` and `p03`. Same walker. New rows in the op table, constant buffers loaded at index 0, and a store width of 4.

## Gotchas

- Three counts stay different. Launch threads, tile cells, and array length. In `p00` those are 256, 64, and 128. Each worker owns two elements because the IR indexes `(a0 * 2) + lane`.
- The arrays are not declared in the PTX. The kernel only receives their addresses.
- `p02` reloads `mul_c1[0]` and `add_c1[0]` on every unrolled lane. The IR says so, so the compiler does too.
- `relu` is a unary `max` against zero. A binary `maximum` is a later reduction.
- `.reg .f32 %f<n>` covers `%f0` through `%f{n-1}`. The printed bound is the number of registers handed out, plus one. The count is known only after the walk.
- This pass lowers one axis.

## References used for this step

- [Architecture](../architecture.md)
- [References](../references.md#pointwise)

**Next:** [02 — Reductions](02-reductions.md)

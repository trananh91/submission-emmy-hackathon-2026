# Architecture

`compile.py` lowers one Kernel IR to PTX. `main` checks that it was given a single path, `compile_kernel` builds the text, and stdout receives the PTX. Statement kinds are lowered as they appear. There is no category switch.

The output is six parts. The compiler generates the prologue, the body, and the return while counting registers, then writes the header, any static shared buffers, the entry, and the register declarations in front of that text.

1. **[Module header.](#1-module-header)** Declares the language level and the kind of machine the program is for. Static shared buffers, when the IR has them, are printed here too.
2. **[Kernel entry and parameters.](#2-kernel-entry-and-parameters)** The name from the IR, and one pointer parameter per name in the launch file, in that order.
3. **[Register declarations.](#3-register-declarations)** Storage private to each thread, separated by the kind of value it holds: truth values, floats, indexes, and addresses. Counted while generating the body, then written here.
4. **[Prologue.](#4-prologue)** Bring the pointers into thread-local storage, turn them into addresses the body can use, and compute this thread's id. Lane and warp are computed later, the first time a statement needs them.
5. **[Body.](#5-body)** The translation of the IR. Walk the statements in order, keep a map from each IR name to the register that holds it, and turn index expressions into address arithmetic.
6. **[Epilogue.](#6-epilogue)** Return.

| Part | [`compile.py`](../compile.py) |
|---|---|
| §1 Module header | `compile_kernel` (`.version`, `.target`, `.address_size`, then any `.shared` lines) |
| §2 Entry and parameters | `kernel_op`, `arg_order`, `compile_kernel` |
| §3 Register declarations | `Emitter.f32` / `b32` / `b64` / `pred`, printed in `compile_kernel` |
| §4 Prologue | `compile_kernel` (pointers), `lower_tile` (thread id, axis decode, tile guard), `ensure_lane_warp` |
| §5 Body | `lower_stmt` and each `lower_*`, `compile_index`, `linear_index`, `compile_pred`, `address_of`, `shared_addr`, `record_shapes`, `Emitter.bind` / `lookup` |
| §6 Epilogue | `compile_kernel` (`ret`); `lower_tile` emits the `DONE` label |

## 1. Module header

`compile_kernel` prints three directives:

- `.version 8.0` — the PTX language level
- `.target sm_89` — the machine the program is for
- `.address_size 64`

`lower_smem` records one `.shared .align 4 .b8` line per `Smem` statement. `compile_kernel` prints those lines after the header and before the entry. A pointwise kernel records none, so nothing is printed between the header and the entry.

## 2. Kernel entry and parameters

`kernel_op` finds the `KernelOp` in the graph. The entry name is `op.name` from the IR.

`arg_order` reads the sibling `meta.json` and uses `arg_order` for the kernel whose `entry` matches that name. Each name becomes one `.param .u64 param_<name>`, in that order. The launch file decides the parameter list. The IR does not.

## 3. Register declarations

`Emitter` counts four kinds of thread-private storage while it generates later parts, then `compile_kernel` prints the counts:

| Counter | Declaration | Holds |
|---|---|---|
| `np` | `.reg .pred %p<…>` | Predicates (truth values) |
| `nf` | `.reg .f32 %f<…>` | Floats, including accumulators |
| `nr` | `.reg .b32 %r<…>` | Indexes, lane, warp, and loop counters |
| `nrd` | `.reg .b64 %rd<…>` | Addresses |

`f32`, `b32`, `b64`, and `pred` allocate the next register. The declaration is a prefix (`%f<n>` means `%f0` … `%f{n-1}`), so the printed bound is the count plus one.

## 4. Prologue

Two steps are appended to `Emitter.body` before any IR statement. Lane and warp are appended later, at the first use.

**Pointers.** For each name in `arg_order`, the compiler loads the parameter with `ld.param.u64` and turns it into a global address with `cvta.to.global.u64`. That address is stored in `Emitter.ptrs`. The arrays already exist on the device. The kernel only receives their addresses.

**Thread id.** `lower_tile` reads `%tid.x`, `%ntid.x`, and `%ctaid.x`, then computes

```text
gid = ctaid.x * ntid.x + tid.x
```

with `mad.lo.s32`. It keeps `%tid.x` so lane and warp can be derived from it.

**Axes.** The last tile axis is the fastest. For each axis after the first, `rem.s32` and `div.s32` peel that extent off `gid`, and the remainder is bound in `Emitter.axes`. One axis skips that loop: the axis register is `gid`.

**Guard.** A thread whose `gid` is past the product of the axis extents is predicated (`setp.ge.s32`) and branches to `DONE`. That check is not in the IR. The product is the bound. For one axis it is that axis extent, so a pointwise kernel guards the same way as before.

**Lane and warp.** `ensure_lane_warp` runs the first time an index or `TreeHalve` names them: `lane = tid & 31` (`and.b32`) and `warp = tid >> 5` (`shr.u32`). A pointwise kernel never asks, so those two instructions are absent there.

## 5. Body

`lower_tile` walks `tile.body` in order and calls `lower_stmt`. `Emitter.ssa` maps each IR name to the register that holds it (`bind` / `lookup`). `lower_stmt` dispatches on the statement kind. An unknown kind raises `CompileError`.

**Shapes.** `record_shapes` stores each buffer shape from the graph. `linear_index` uses it when a load or store has more than one index: row-major, trailing dimension fastest. One index is compiled as itself. An empty index is the base pointer.

**Indexes.** `compile_index` lowers a `Var` to a tile axis, `lane`, or `warp`, a `Literal` to `mov.u32`, and `+` / `*` to `add.s32` / `mul.lo.s32`. `compile_pred` lowers `==` / `!=` to `setp.eq.s32` / `setp.ne.s32`.

**Addresses.** `address_of` scales the index by the element size (`mul.wide.s32`) and adds it to the global pointer. `shared_addr` moves the shared symbol into a 32-bit register and, when there is an index, adds `index * element size` with `mul.lo.s32` and `add.s32`.

**Statements currently lowered:**

| IR | PTX |
|---|---|
| `Load` | `ld.global.f32`, or `ld.global.v2.f32` / `v4.f32` when a global load names 2 or 4 values. A shared buffer uses `ld.shared.f32`. Each name is bound to an `%f` register. |
| `Assign` | `add` → `add.f32`, `multiply` → `mul.f32`, `divide` → `div.rn.f32`, `relu` → `max.f32` against zero. Anything else is refused. |
| `Write` | `st.global.f32`, or a `v2` / `v4` store. A shared buffer is one `st.shared.f32`. An empty index stores through the pointer itself. |
| `StridedLoop` | A header label, `setp.ge.s32` against the axis extent, the body, then `add.s32` of the step and a branch back. The loop index is a copy of `start`, so the tile axis is left alone. |
| `Accum` | `add.f32` into the same register. The identity `0f00000000` is moved once, before the loop header, so the first iteration reads a defined register. Any op other than `add` is refused. |
| `WarpShuffle` | One `add` state. `shfl.sync.bfly.b32` at offsets `length/2`, `length/4`, … `1`, then `add.f32` of the shuffled value. A bit cast (`mov.b32`) moves the float through the shuffle. |
| `Smem` | Float only. One `.shared .align 4 .b8 name[count * 4]` line, and the name is recorded so later loads and stores use the shared address. |
| `Cond` | `@!%p bra` around the body. An else body branches to a join label. |
| `Sync` | `bar.sync` of the IR barrier id. |
| `TreeHalve` | One `add` buffer. `span` runs from `length/2` down to `1`. A thread whose `tid_var` is at least `span` skips the update. Every thread then `bar.sync`. After the last span, every thread loads slot 0 into the accumulator. |

## 6. Epilogue

`lower_tile` emits the `DONE` label, then `compile_kernel` emits `ret`.

## What this pass covers

For now, the architecture proposed above could cover:
- Category 1-pointwise: one `Tile`, one axis, elementwise `add`, `multiply`, and `relu`, and vector loads and stores of width 1, 2, or 4.
- Category 2-reductions: `StridedLoop`, one `add` accumulator, `WarpShuffle`, `Smem`, `Cond`, `Sync`, and `TreeHalve`, plus `divide`. A tile may have more than one axis.

Raises `CompileError` where not cover yet.

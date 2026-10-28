# Architecture

`compile.py` lowers one Kernel IR to PTX. `main` checks that it was given a single path, `compile_kernel` builds the text, and stdout receives the PTX.

The output is six parts. The compiler generates the prologue, the body, and the return while counting registers, then writes the header, the entry, and the register declarations in front of that text.

1. **[Module header.](#1-module-header)** Declares the language level and the kind of machine the program is for.
2. **[Kernel entry and parameters.](#2-kernel-entry-and-parameters)** The name from the IR, and one pointer parameter per name in the launch file, in that order.
3. **[Register declarations.](#3-register-declarations)** Storage private to each thread, separated by the kind of value it holds: truth values, floats, indexes, and addresses. Counted while generating the body, then written here.
4. **[Prologue.](#4-prologue)** Bring the pointers into thread-local storage, turn them into addresses the body can use, and compute this thread's id.
5. **[Body.](#5-body)** The translation of the IR. Walk the statements in order, keep a map from each IR name to the register that holds it, and turn index expressions into address arithmetic.
6. **[Epilogue.](#6-epilogue)** Return.

| Part | [`compile.py`](compile.py) |
|---|---|
| §1 Module header | `compile_kernel` (`.version`, `.target`, `.address_size`) |
| §2 Entry and parameters | `kernel_op`, `arg_order`, `compile_kernel` |
| §3 Register declarations | `Emitter.f32` / `b32` / `b64` / `pred`, printed in `compile_kernel` |
| §4 Prologue | `compile_kernel` (pointers), `lower_tile` (thread id, tile guard) |
| §5 Body | `lower_stmt`, `compile_index`, `address_of`, `Emitter.bind` / `lookup` |
| §6 Epilogue | `compile_kernel` (`ret`); `lower_tile` emits the `DONE` label |

## 1. Module header

`compile_kernel` prints three directives:

- `.version 8.0` — the PTX language level
- `.target sm_89` — the machine the program is for
- `.address_size 64`

## 2. Kernel entry and parameters

`kernel_op` finds the `KernelOp` in the graph. The entry name is `op.name` from the IR.

`arg_order` reads the sibling `meta.json` and uses `arg_order` for the kernel whose `entry` matches that name. Each name becomes one `.param .u64 param_<name>`, in that order. The launch file decides the parameter list. The IR does not.

## 3. Register declarations

`Emitter` counts four kinds of thread-private storage while it generates later parts, then `compile_kernel` prints the counts:

| Counter | Declaration | Holds |
|---|---|---|
| `np` | `.reg .pred %p<…>` | Predicates (truth values) |
| `nf` | `.reg .f32 %f<…>` | Floats |
| `nr` | `.reg .b32 %r<…>` | Indexes |
| `nrd` | `.reg .b64 %rd<…>` | Addresses |

`f32`, `b32`, `b64`, and `pred` allocate the next register. The declaration is a prefix (`%f<n>` means `%f0` … `%f{n-1}`), so the printed bound is the count plus one.

## 4. Prologue

Two steps, both appended to `Emitter.body` before any IR statement.

**Pointers.** For each name in `arg_order`, the compiler loads the parameter with `ld.param.u64` and turns it into a global address with `cvta.to.global.u64`. That address is stored in `Emitter.ptrs`. The arrays already exist on the device. The kernel only receives their addresses.

**Thread id.** `lower_tile` reads `%tid.x`, `%ntid.x`, and `%ctaid.x`, then computes

```text
gid = ctaid.x * ntid.x + tid.x
```

with `mad.lo.s32`. The single tile axis is bound to that register in `Emitter.axes`. A thread whose `gid` is past the tile extent is predicated (`setp.ge.s32`) and branches to `DONE`. That check is not in the IR. Only a 1-D tile is lowered.

## 5. Body

`lower_tile` walks `tile.body` in order and calls `lower_stmt`. `Emitter.ssa` maps each IR name to the register that holds it (`bind` / `lookup`).

**Indexes.** `compile_index` lowers a `Var` to the tile axis, a `Literal` to `mov.u32`, and `+` / `*` to `add.s32` / `mul.lo.s32`. `address_of` scales the index by the element size (`mul.wide.s32`) and adds it to the buffer pointer.

**Statements currently lowered:**

| IR | PTX |
|---|---|
| `Load` | `ld.global.f32`, or `ld.global.v2.f32` / `v4.f32` when the load names 2 or 4 values. Each name is bound to an `%f` register. |
| `Assign` | `add` → `add.f32`, `multiply` → `mul.f32`, `relu` → `max.f32` against zero. Anything else is refused. |
| `Write` | `st.global.f32`, or a `v2` / `v4` store. An empty index stores through the pointer itself. Atomic stores are refused. |

## 6. Epilogue

`lower_tile` emits the `DONE` label, then `compile_kernel` emits `ret`.

## What this pass covers

For now, the architecture proposed above could cover:
- Category 1-pointwise: one `Tile`, one axis, elementwise `add`, `multiply`, and `relu`, and vector loads and stores of width 1, 2, or 4. Raises `CompileError` where not cover yet.

#!/usr/bin/env python3
"""Kernel IR -> PTX for pointwise kernels, block reductions, and matmul.

    python compile.py path/to/<kernel>.ir.json > out.ptx

Parameter order comes from the sibling meta.json arg_order.
Statement kinds are lowered as they appear. There is no category switch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph


class CompileError(Exception):
    pass


class Emitter:
    # architecture.md §3 counts registers here; §5 stores the SSA map in ssa.
    # shared and smem record Smem buffers printed before the entry.
    def __init__(self) -> None:
        self.body: list[str] = []
        self.shared: list[str] = []
        self.nf = self.nr = self.nrd = self.np = 0
        self.ssa: dict[str, str] = {}
        self.ptrs: dict[str, str] = {}
        self.axes: dict[str, str] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.smem: set[str] = set()
        self.smem_bytes = 0
        self.smem_used = 0
        self.smem_off: dict[str, int] = {}
        self.smem_symbol: dict[str, str] = {}
        self.tid: str | None = None
        self._lab = 0

    def emit(self, line: str) -> None:
        self.body.append(line)

    def label(self, prefix: str) -> str:
        self._lab += 1
        return f"{prefix}_{self._lab}"

    def f32(self) -> str:  # §3 floats. b32 indexes, b64 addresses, pred truth values.
        self.nf += 1
        return f"%f{self.nf}"

    def b32(self) -> str:
        self.nr += 1
        return f"%r{self.nr}"

    def b64(self) -> str:
        self.nrd += 1
        return f"%rd{self.nrd}"

    def pred(self) -> str:
        self.np += 1
        return f"%p{self.np}"

    def bind(self, name: str, reg: str) -> None:  # §5 IR name → register. lookup reads it.
        self.ssa[name] = reg

    def lookup(self, name: str) -> str:
        try:
            return self.ssa[name]
        except KeyError as exc:
            raise CompileError(f"unknown value {name}") from exc

    def ensure_lane_warp(self) -> None:  # §4 lane = tid & 31, warp = tid >> 5, on first use.
        if "lane" in self.axes:
            return
        if self.tid is None:
            raise CompileError("lane/warp used before the thread id exists")
        lane = self.b32()
        warp = self.b32()
        self.emit(f"\tand.b32 {lane}, {self.tid}, 31;")
        self.emit(f"\tshr.u32 {warp}, {self.tid}, 5;")
        self.axes["lane"] = lane
        self.axes["warp"] = warp


def as_int(dim) -> int:
    if isinstance(dim, int):
        return dim
    if hasattr(dim, "value"):
        return int(dim.value)
    return int(dim)


def kernel_op(graph):  # §2 entry name is KernelOp.name from the IR.
    for node in graph.nodes.values():
        if type(node.op).__name__ == "KernelOp":
            return node.op
    raise CompileError("no KernelOp in the IR")


def kernel_launch(ir_path: Path, entry: str) -> dict:  # §1–§2 one meta.json record: arg_order and smem_bytes.
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        raise CompileError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    for kernel in meta["kernels"]:
        if kernel["entry"] == entry:
            return kernel
    raise CompileError(f"meta.json has no kernel entry {entry}")


def arg_order(ir_path: Path, entry: str) -> list[str]:  # §2 parameter order from meta.json.
    return list(kernel_launch(ir_path, entry)["arg_order"])


def record_shapes(em: Emitter, graph) -> None:  # §5 buffer shapes for row-major indexes.
    for node in graph.nodes.values():
        for out in getattr(node, "outputs", ()) or ():
            em.shapes[out.name] = tuple(as_int(dim) for dim in out.shape)


def compile_index(em: Emitter, expr) -> str:  # §5 index expressions → b32. lane and warp come from §4.
    kind = type(expr).__name__
    if kind == "Var":
        if expr.name in ("lane", "warp"):
            em.ensure_lane_warp()
        if expr.name in em.axes:
            return em.axes[expr.name]
        raise CompileError(f"index var {expr.name} is not a tile axis")
    if kind == "Literal":
        reg = em.b32()
        em.emit(f"\tmov.u32 {reg}, {int(expr.value)};")
        return reg
    if kind == "BinaryExpr":
        left = compile_index(em, expr.left)
        right = compile_index(em, expr.right)
        dest = em.b32()
        if expr.op == "+":
            em.emit(f"\tadd.s32 {dest}, {left}, {right};")
        elif expr.op == "-":
            em.emit(f"\tsub.s32 {dest}, {left}, {right};")
        elif expr.op == "*":
            em.emit(f"\tmul.lo.s32 {dest}, {left}, {right};")
        elif expr.op == "/":
            em.emit(f"\tdiv.s32 {dest}, {left}, {right};")
        elif expr.op == "%":
            em.emit(f"\trem.s32 {dest}, {left}, {right};")
        else:
            raise CompileError(f"unsupported index op {expr.op}")
        return dest
    if kind == "TernaryExpr":
        pred = compile_pred(em, expr.cond)
        if_true = compile_index(em, expr.if_true)
        if_false = compile_index(em, expr.if_false)
        dest = em.b32()
        em.emit(f"\tselp.b32 {dest}, {if_true}, {if_false}, {pred};")
        return dest
    raise CompileError(f"unsupported index expr {kind}")


def compile_pred(em: Emitter, expr) -> str:  # §5 ==, !=, and < become setp.
    if type(expr).__name__ != "BinaryExpr" or expr.op not in ("==", "!=", "<"):
        raise CompileError(f"unsupported predicate {expr}")
    left = compile_index(em, expr.left)
    right = compile_index(em, expr.right)
    pred = em.pred()
    op = {"==": "eq", "!=": "ne", "<": "lt"}[expr.op]
    em.emit(f"\tsetp.{op}.s32 {pred}, {left}, {right};")
    return pred


def linear_index(em: Emitter, buf: str, index) -> str | None:  # §5 one index, or row-major from the buffer shape.
    if not index:
        return None
    if len(index) == 1:
        return compile_index(em, index[0])
    shape = em.shapes.get(buf)
    if shape is None or len(shape) < len(index):
        raise CompileError(f"cannot linearize {buf}{list(index)}")
    acc = None
    for dim, expr in enumerate(index):
        idx = compile_index(em, expr)
        stride = 1
        for size in shape[dim + 1 :]:
            stride *= size
        if stride != 1:
            scaled = em.b32()
            em.emit(f"\tmul.lo.s32 {scaled}, {idx}, {stride};")
            idx = scaled
        if acc is None:
            acc = idx
        else:
            dest = em.b32()
            em.emit(f"\tadd.s32 {dest}, {acc}, {idx};")
            acc = dest
    return acc


def address_of(em: Emitter, buf: str, index_reg: str, nbytes: int) -> str:  # §5 index → global address.
    off = em.b64()
    em.emit(f"\tmul.wide.s32 {off}, {index_reg}, {nbytes};")
    addr = em.b64()
    em.emit(f"\tadd.s64 {addr}, {em.ptrs[buf]}, {off};")
    return addr


def shared_addr(em: Emitter, name: str, index_reg: str | None, nbytes: int) -> str:  # §5 index → shared address.
    base = em.b32()
    em.emit(f"\tmov.u32 {base}, {em.smem_symbol.get(name, name)};")
    byte_off = em.smem_off.get(name, 0)
    if byte_off:
        em.emit(f"\tadd.s32 {base}, {base}, {byte_off};")
    if index_reg is None:
        return base
    off = em.b32()
    em.emit(f"\tmul.lo.s32 {off}, {index_reg}, {nbytes};")
    addr = em.b32()
    em.emit(f"\tadd.s32 {addr}, {base}, {off};")
    return addr


BIN_F32 = {
    "add": "add.f32",
    "multiply": "mul.f32",
    "divide": "div.rn.f32",
}


def lower_load(em: Emitter, stmt) -> None:  # §5 global or shared load. Width 2 or 4 is a vector global load.
    index = linear_index(em, stmt.input, stmt.index)
    nbytes = 4 if stmt.dtype is None else stmt.dtype.nbytes
    width = len(stmt.names)
    dests = [em.f32() for _ in stmt.names]
    if stmt.input in em.smem:
        addr = shared_addr(em, stmt.input, index, nbytes)
        space = "shared"
    else:
        if index is None:
            addr = em.ptrs[stmt.input]
        else:
            addr = address_of(em, stmt.input, index, nbytes)
        space = "global"
    if width == 1:
        em.emit(f"\tld.{space}.f32 {dests[0]}, [{addr}];")
    elif width in (2, 4) and space == "global":
        packed = ", ".join(dests)
        em.emit(f"\tld.global.v{width}.f32 {{{packed}}}, [{addr}];")
    else:
        raise CompileError(f"unsupported load width {width}")
    for name, reg in zip(stmt.names, dests):
        em.bind(name, reg)


def lower_assign(em: Emitter, stmt) -> None:  # §5 add, multiply, divide, and relu.
    dest = em.f32()
    op = stmt.op.name
    if op in BIN_F32 and len(stmt.args) == 2:
        a, b = (em.lookup(n) for n in stmt.args)
        em.emit(f"\t{BIN_F32[op]} {dest}, {a}, {b};")
    elif op == "relu" and len(stmt.args) == 1:
        x = em.lookup(stmt.args[0])
        em.emit(f"\tmax.f32 {dest}, {x}, 0f00000000;")
    else:
        raise CompileError(f"unsupported elementwise op {op}")
    em.bind(stmt.name, dest)


def lower_write(em: Emitter, stmt) -> None:  # §5 global or shared store. Width 2 or 4 is a vector global store.
    index = linear_index(em, stmt.output, stmt.index)
    nbytes = stmt.value_dtype.nbytes
    regs = [em.lookup(n) for n in stmt.values]
    width = len(regs)
    if stmt.atomic:
        if stmt.output in em.smem or width != 1:
            raise CompileError("only a scalar global atomic add is lowered")
        if index is None:
            addr = em.ptrs[stmt.output]
        else:
            addr = address_of(em, stmt.output, index, nbytes)
        em.emit(f"\tred.global.add.f32 [{addr}], {regs[0]};")
        return
    if stmt.output in em.smem:
        if width != 1:
            raise CompileError("vector shared stores are not lowered")
        addr = shared_addr(em, stmt.output, index, nbytes)
        em.emit(f"\tst.shared.f32 [{addr}], {regs[0]};")
        return
    if index is None:
        addr = em.ptrs[stmt.output]
    else:
        addr = address_of(em, stmt.output, index, nbytes)
    if width == 1:
        em.emit(f"\tst.global.f32 [{addr}], {regs[0]};")
    elif width in (2, 4):
        packed = ", ".join(regs)
        em.emit(f"\tst.global.v{width}.f32 [{addr}], {{{packed}}};")
    else:
        raise CompileError(f"unsupported store width {width}")


def init_accum(em: Emitter, stmt) -> None:  # §5 add identity, before the loop that folds it.
    if stmt.name in em.ssa:
        return
    if stmt.op.name != "add":
        raise CompileError(f"unsupported accumulator {stmt.op.name}")
    reg = em.f32()
    em.emit(f"\tmov.f32 {reg}, 0f00000000;")
    em.bind(stmt.name, reg)


def lower_accum(em: Emitter, stmt) -> None:  # §5 fold one value into the add accumulator.
    init_accum(em, stmt)
    if stmt.op.name != "add":
        raise CompileError(f"unsupported accumulator {stmt.op.name}")
    acc = em.lookup(stmt.name)
    val = em.lookup(stmt.value)
    em.emit(f"\tadd.f32 {acc}, {acc}, {val};")


def lower_init(em: Emitter, stmt) -> None:  # §5 one zero before the loops. A later Accum folds into it.
    if stmt.name in em.ssa:
        raise CompileError(f"duplicate init {stmt.name}")
    if float(stmt.identity) != 0.0:
        raise CompileError(f"unsupported init identity {stmt.identity}")
    reg = em.f32()
    em.emit(f"\tmov.f32 {reg}, 0f00000000;")
    em.bind(stmt.name, reg)


def lower_loop(em: Emitter, loop) -> None:  # §5 counted loop from 0, step 1. Init is the seed.
    i = em.b32()
    em.emit(f"\tmov.u32 {i}, 0;")
    em.axes[loop.axis.name] = i
    head = em.label("LOOP")
    end = em.label("LOOP_END")
    bound = as_int(loop.axis.extent)
    em.emit(f"{head}:")
    pred = em.pred()
    em.emit(f"\tsetp.ge.s32 {pred}, {i}, {bound};")
    em.emit(f"\t@{pred} bra {end};")
    for stmt in loop.body:
        lower_stmt(em, stmt)
    em.emit(f"\tadd.s32 {i}, {i}, 1;")
    em.emit(f"\tbra {head};")
    em.emit(f"{end}:")


def lower_strided_loop(em: Emitter, loop) -> None:  # §5 stride the axis. Accumulators start before the loop.
    for stmt in loop.body:
        if type(stmt).__name__ == "Accum":
            init_accum(em, stmt)
    start = compile_index(em, loop.start)
    step = compile_index(em, loop.step)
    i = em.b32()
    em.emit(f"\tmov.u32 {i}, {start};")
    em.axes[loop.axis.name] = i
    head = em.label("LOOP")
    end = em.label("LOOP_END")
    bound = as_int(loop.axis.extent)
    em.emit(f"{head}:")
    pred = em.pred()
    em.emit(f"\tsetp.ge.s32 {pred}, {i}, {bound};")
    em.emit(f"\t@{pred} bra {end};")
    for stmt in loop.body:
        lower_stmt(em, stmt)
    em.emit(f"\tadd.s32 {i}, {i}, {step};")
    em.emit(f"\tbra {head};")
    em.emit(f"{end}:")


def lower_warp_shuffle(em: Emitter, stmt) -> None:  # §5 one add butterfly across the warp.
    if len(stmt.state) != 1 or stmt.combine_states[0].op.name != "add":
        raise CompileError("only a single add warp shuffle is lowered")
    acc = em.lookup(stmt.state[0])
    offset = int(stmt.length) // 2
    while offset >= 1:
        bits = em.b32()
        shuffled = em.b32()
        other = em.f32()
        em.emit(f"\tmov.b32 {bits}, {acc};")
        em.emit(f"\tshfl.sync.bfly.b32 {shuffled}, {bits}, {offset}, 0x1f, 0xffffffff;")
        em.emit(f"\tmov.b32 {other}, {shuffled};")
        em.emit(f"\tadd.f32 {acc}, {acc}, {other};")
        offset //= 2


def smem_align(stmt) -> int:  # §1 align 0 is the static default of 4. Any other IR value is used as written.
    align = int(stmt.align)
    return 4 if align == 0 else align


def lower_smem(em: Emitter, stmt) -> None:  # §1 static buffer, or a byte offset into one dynamic pool.
    if stmt.dtype != "float":
        raise CompileError(f"unsupported shared dtype {stmt.dtype}")
    extents = tuple(as_int(e) for e in stmt.extents)
    count = 1
    for extent in extents:
        count *= extent
    nbytes = count * 4
    align = smem_align(stmt)
    if em.smem_bytes == 0:
        em.shared.append(f".shared .align {align} .b8 {stmt.name}[{nbytes}];")
        em.smem_off[stmt.name] = 0
    else:
        if not any(line.startswith(".extern .shared") for line in em.shared):
            em.shared.append(".extern .shared .align 16 .b8 _smem_pool[];")
        start = em.smem_used
        if start % align:
            start += align - (start % align)
        em.smem_off[stmt.name] = start
        em.smem_used = start + nbytes
        em.smem_symbol[stmt.name] = "_smem_pool"
    em.smem.add(stmt.name)
    em.shapes[stmt.name] = extents


def lower_cond(em: Emitter, stmt) -> None:  # §5 predicated body, with a join when an else body exists.
    pred = compile_pred(em, stmt.cond)
    skip = em.label("COND_SKIP")
    em.emit(f"\t@!{pred} bra {skip};")
    for inner in stmt.body:
        lower_stmt(em, inner)
    else_body = getattr(stmt, "else_body", ()) or ()
    if else_body:
        join = em.label("COND_JOIN")
        em.emit(f"\tbra {join};")
        em.emit(f"{skip}:")
        for inner in else_body:
            lower_stmt(em, inner)
        em.emit(f"{join}:")
    else:
        em.emit(f"{skip}:")


def lower_tree_halve(em: Emitter, stmt) -> None:  # §5 shared-memory tree of one add, then every thread loads slot 0.
    if len(stmt.state) != 1 or len(stmt.bufs) != 1:
        raise CompileError("only a single-value TreeHalve is lowered")
    if stmt.combine_states[0].op.name != "add":
        raise CompileError("only an add TreeHalve is lowered")
    buf = stmt.bufs[0]
    acc = em.lookup(stmt.state[0])
    em.ensure_lane_warp()
    tid = em.axes[stmt.tid_var]
    barrier = int(stmt.barrier_id)
    span = int(stmt.length) // 2
    while span > 0:
        pred = em.pred()
        skip = em.label("HALVE_SKIP")
        em.emit(f"\tsetp.ge.s32 {pred}, {tid}, {span};")
        em.emit(f"\t@{pred} bra {skip};")
        other_i = em.b32()
        em.emit(f"\tadd.s32 {other_i}, {tid}, {span};")
        mine = shared_addr(em, buf, tid, 4)
        other = shared_addr(em, buf, other_i, 4)
        left = em.f32()
        right = em.f32()
        em.emit(f"\tld.shared.f32 {left}, [{mine}];")
        em.emit(f"\tld.shared.f32 {right}, [{other}];")
        em.emit(f"\tadd.f32 {left}, {left}, {right};")
        em.emit(f"\tst.shared.f32 [{mine}], {left};")
        em.emit(f"{skip}:")
        em.emit(f"\tbar.sync {barrier};")
        span //= 2
    base = shared_addr(em, buf, None, 4)
    em.emit(f"\tld.shared.f32 {acc}, [{base}];")


def lower_cp_async_copy(em: Emitter, stmt) -> None:  # §5 16-byte cg copy, or ca for 4 or 8. lane_rows is not the stride.
    if stmt.swizzle != "NONE":
        raise CompileError(f"unsupported cp.async swizzle {stmt.swizzle}")
    nbytes = int(stmt.nbytes)
    if nbytes not in (4, 8, 16):
        raise CompileError(f"unsupported cp.async size {nbytes}")
    index = linear_index(em, stmt.smem, stmt.smem_index)
    if stmt.lane_index:
        lane = linear_index(em, stmt.smem, stmt.lane_index)
        if index is None:
            index = lane
        elif lane is not None:
            both = em.b32()
            em.emit(f"\tadd.s32 {both}, {index}, {lane};")
            index = both
    saddr = shared_addr(em, stmt.smem, index, 4)
    gindex = linear_index(em, stmt.src, stmt.src_index)
    if gindex is None:
        gaddr = em.ptrs[stmt.src]
    else:
        gaddr = address_of(em, stmt.src, gindex, 4)
    if nbytes == 16:
        em.emit(f"\tcp.async.cg.shared.global [{saddr}], [{gaddr}], 16;")
    else:
        em.emit(f"\tcp.async.ca.shared.global [{saddr}], [{gaddr}], {nbytes};")


def lower_stmt(em: Emitter, stmt) -> None:  # §5 one translation per statement kind, including Sync.
    kind = type(stmt).__name__
    if kind == "Load":
        lower_load(em, stmt)
    elif kind == "Assign":
        lower_assign(em, stmt)
    elif kind == "Write":
        lower_write(em, stmt)
    elif kind == "StridedLoop":
        lower_strided_loop(em, stmt)
    elif kind == "Accum":
        lower_accum(em, stmt)
    elif kind == "WarpShuffle":
        lower_warp_shuffle(em, stmt)
    elif kind == "Smem":
        lower_smem(em, stmt)
    elif kind == "Cond":
        lower_cond(em, stmt)
    elif kind == "Sync":
        em.emit(f"\tbar.sync {int(stmt.barrier_id)};")
    elif kind == "TreeHalve":
        lower_tree_halve(em, stmt)
    elif kind == "Init":
        lower_init(em, stmt)
    elif kind == "Loop":
        lower_loop(em, stmt)
    elif kind == "CpAsyncCopy":
        lower_cp_async_copy(em, stmt)
    elif kind == "CpAsyncCommit":
        em.emit("\tcp.async.commit_group;")
    elif kind == "CpAsyncWait":
        em.emit(f"\tcp.async.wait_group {int(stmt.group)};")
    else:
        raise CompileError(f"unsupported statement {kind}")


def lower_tile(em: Emitter, tile) -> None:  # §4 thread id, axis decode, and tile guard, then §5 body. DONE closes the guard.
    tid = em.b32()
    ntid = em.b32()
    cta = em.b32()
    gid = em.b32()
    em.tid = tid
    em.emit(f"\tmov.u32 {tid}, %tid.x;")
    em.emit(f"\tmov.u32 {ntid}, %ntid.x;")
    em.emit(f"\tmov.u32 {cta}, %ctaid.x;")
    em.emit(f"\tmad.lo.s32 {gid}, {cta}, {ntid}, {tid};")

    extents = [as_int(axis.extent) for axis in tile.axes]
    rest = gid
    for i in range(len(tile.axes) - 1, 0, -1):
        rem = em.b32()
        quot = em.b32()
        em.emit(f"\trem.s32 {rem}, {rest}, {extents[i]};")
        em.emit(f"\tdiv.s32 {quot}, {rest}, {extents[i]};")
        em.axes[tile.axes[i].name] = rem
        rest = quot
    em.axes[tile.axes[0].name] = rest

    cells = 1
    for extent in extents:
        cells *= extent
    skip = em.pred()
    done = em.label("DONE")
    em.emit(f"\tsetp.ge.s32 {skip}, {gid}, {cells};")
    em.emit(f"\t@{skip} bra {done};")
    for stmt in tile.body:
        lower_stmt(em, stmt)
    em.emit(f"{done}:")


def compile_kernel(ir_path: Path) -> str:
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    op = kernel_op(graph)
    launch = kernel_launch(ir_path, op.name)
    order = list(launch["arg_order"])
    em = Emitter()
    em.smem_bytes = int(launch["smem_bytes"])
    record_shapes(em, graph)

    # §4 Prologue: pointer parameters → global addresses.
    for name in order:
        raw = em.b64()
        em.emit(f"\tld.param.u64 {raw}, [param_{name}];")
        ptr = em.b64()
        em.emit(f"\tcvta.to.global.u64 {ptr}, {raw};")
        em.ptrs[name] = ptr

    body = op.body
    if len(body) != 1 or type(body[0]).__name__ != "Tile":
        raise CompileError("expected a single Tile as the kernel body")
    lower_tile(em, body[0])
    if em.smem_bytes and em.smem_used != em.smem_bytes:
        raise CompileError(
            f"shared pool is {em.smem_used} bytes, meta.json smem_bytes is {em.smem_bytes}"
        )
    em.emit("\tret;")  # §6 Epilogue.

    params = ",\n".join(f"\t.param .u64 param_{name}" for name in order)
    lines = [
        # §1 Module header. Static .shared lines follow when the walk declared any.
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
        *em.shared,
        *( [""] if em.shared else [] ),
        # §2 Kernel entry and parameters.
        f".visible .entry {op.name}(",
        params,
        ")",
        "{",
        # §3 Register declarations. Counts were taken while emitting the body.
        f"\t.reg .pred %p<{em.np + 1}>;",
        f"\t.reg .f32 %f<{em.nf + 1}>;",
        f"\t.reg .b32 %r<{em.nr + 1}>;",
        f"\t.reg .b64 %rd<{em.nrd + 1}>;",
        "",
        *em.body,
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/<kernel>.ir.json", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(compile_kernel(Path(sys.argv[1])))
    except CompileError as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

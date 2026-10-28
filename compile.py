from __future__ import annotations

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph


class CompileError(Exception):
    pass


class Emitter:
    # architecture.md §3 counts registers here; §5 stores the SSA map in ssa.
    def __init__(self) -> None:
        self.body: list[str] = []
        self.nf = self.nr = self.nrd = self.np = 0
        self.ssa: dict[str, str] = {}
        self.ptrs: dict[str, str] = {}
        self.axes: dict[str, str] = {}
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


def kernel_op(graph):  # §2 entry name is KernelOp.name from the IR.
    for node in graph.nodes.values():
        if type(node.op).__name__ == "KernelOp":
            return node.op
    raise CompileError("no KernelOp in the IR")


def arg_order(ir_path: Path, entry: str) -> list[str]:  # §2 parameter order from meta.json.
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        raise CompileError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    for kernel in meta["kernels"]:
        if kernel["entry"] == entry:
            return list(kernel["arg_order"])
    raise CompileError(f"meta.json has no kernel entry {entry}")


def compile_index(em: Emitter, expr) -> str:  # §5 index expressions → b32.
    kind = type(expr).__name__
    if kind == "Var":
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
        elif expr.op == "*":
            em.emit(f"\tmul.lo.s32 {dest}, {left}, {right};")
        else:
            raise CompileError(f"unsupported index op {expr.op}")
        return dest
    raise CompileError(f"unsupported index expr {kind}")


def address_of(em: Emitter, buf: str, index_reg: str, nbytes: int) -> str:  # §5 index → global address.
    off = em.b64()
    em.emit(f"\tmul.wide.s32 {off}, {index_reg}, {nbytes};")
    addr = em.b64()
    em.emit(f"\tadd.s64 {addr}, {em.ptrs[buf]}, {off};")
    return addr


BIN_F32 = {
    "add": "add.f32",
    "multiply": "mul.f32",
}


def lower_stmt(em: Emitter, stmt) -> None:  # §5 Load, Assign, Write.
    kind = type(stmt).__name__
    if kind == "Load":
        index = compile_index(em, stmt.index[0])
        addr = address_of(em, stmt.input, index, stmt.dtype.nbytes)
        width = len(stmt.names)
        dests = [em.f32() for _ in stmt.names]
        if width == 1:
            em.emit(f"\tld.global.f32 {dests[0]}, [{addr}];")
        elif width in (2, 4):
            packed = ", ".join(dests)
            em.emit(f"\tld.global.v{width}.f32 {{{packed}}}, [{addr}];")
        else:
            raise CompileError(f"unsupported load width {width}")
        for name, reg in zip(stmt.names, dests):
            em.bind(name, reg)
        return
    if kind == "Assign":
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
        return
    if kind == "Write":
        if stmt.atomic:
            raise CompileError("atomic stores are not lowered yet")
        index = compile_index(em, stmt.index[0]) if stmt.index else None
        nbytes = stmt.value_dtype.nbytes
        if index is None:
            addr = em.ptrs[stmt.output]
        else:
            addr = address_of(em, stmt.output, index, nbytes)
        regs = [em.lookup(n) for n in stmt.values]
        width = len(regs)
        if width == 1:
            em.emit(f"\tst.global.f32 [{addr}], {regs[0]};")
        elif width in (2, 4):
            packed = ", ".join(regs)
            em.emit(f"\tst.global.v{width}.f32 [{addr}], {{{packed}}};")
        else:
            raise CompileError(f"unsupported store width {width}")
        return
    raise CompileError(f"unsupported statement {kind}")


def lower_tile(em: Emitter, tile) -> None:  # §4 thread id and tile guard, then §5 body. DONE closes the guard.
    if len(tile.axes) != 1:
        raise CompileError("only a 1-D tile is lowered")
    axis = tile.axes[0]
    n = int(axis.extent.value)
    tid = em.b32()
    ntid = em.b32()
    cta = em.b32()
    gid = em.b32()
    em.emit(f"\tmov.u32 {tid}, %tid.x;")
    em.emit(f"\tmov.u32 {ntid}, %ntid.x;")
    em.emit(f"\tmov.u32 {cta}, %ctaid.x;")
    em.emit(f"\tmad.lo.s32 {gid}, {cta}, {ntid}, {tid};")
    em.axes[axis.name] = gid
    skip = em.pred()
    done = em.label("DONE")
    em.emit(f"\tsetp.ge.s32 {skip}, {gid}, {n};")
    em.emit(f"\t@{skip} bra {done};")
    for stmt in tile.body:
        lower_stmt(em, stmt)
    em.emit(f"{done}:")


def compile_kernel(ir_path: Path) -> str:
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    op = kernel_op(graph)
    order = arg_order(ir_path, op.name)
    em = Emitter()

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
    em.emit("\tret;")  # §6 Epilogue.

    params = ",\n".join(f"\t.param .u64 param_{name}" for name in order)
    lines = [
        # §1 Module header.
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
        "",
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

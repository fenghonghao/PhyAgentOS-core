"""``robot.run(code)``: execute a model-written Python block against the primitives.

* The namespace persists across blocks within a scene; ``reset()`` clears it at scene start.
* Only what the code prints is returned (plus traceback, ledger, elapsed time), and the image
  files of the block's ``show`` calls.
* No time limit is applied to a block. Cancellation only happens through an explicit cancel.

Guard rails (same process, so they stop misuse by the model, not a determined attacker;
an OS-level boundary would need a separate sandboxed process):
* restricted builtins (no open/eval/exec/compile/globals/vars...), guarded getattr;
* imports limited to modules whose public surface exposes no other modules
  (math, time, itertools, functools, collections, random); ``numpy`` is replaced by a view
  without submodules (except safe views of linalg/random/fft), file I/O or ctypes;
* an AST check rejecting dunder names and any attribute starting with ``_`` (e.g.
  ``random._os``) plus array file/pickle methods (``tofile``, ``dump``, ``dumps``).
"""

from __future__ import annotations

import ast
import builtins
import linecache
import types
import math
import random as _random_module
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np

from .primitives import Robot
from .types import Cancelled, RobotFault

MAX_OUTPUT_CHARS = 8000
_FILENAME = "<robot.run>"

ALLOWED_IMPORTS = {"numpy", "math", "time", "itertools", "functools", "collections", "random"}
_NUMPY_DENY = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt",
               "fromfile", "fromregex", "memmap", "DataSource", "show_config", "show_runtime",
               "get_include", "info", "ctypeslib", "lib", "testing", "f2py", "typing", "core"}
_DENIED_ATTRS = {"tofile", "dump", "dumps",
                 # frame / code introspection: a generator's frame leads to the executor's globals
                 "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "ag_frame", "ag_code",
                 "ag_await", "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "tb_frame", "tb_next",
                 "func_globals", "co_code", "co_consts",
                 # numpy's ctypes bridge hands out raw pointers
                 "ctypes", "from_address", "from_buffer", "from_buffer_copy", "data_as", "shape_as", "strides_as"}


def _safe_view(mod: types.ModuleType, deny: set[str] = frozenset()) -> types.SimpleNamespace:
    """Public, non-module attributes of ``mod`` (no route to os/sys/ctypes through it)."""
    return types.SimpleNamespace(**{
        a: getattr(mod, a) for a in dir(mod)
        if not a.startswith("_") and a not in deny and not isinstance(getattr(mod, a, None), types.ModuleType)
    })


def _numpy_view() -> types.SimpleNamespace:
    view = _safe_view(np, _NUMPY_DENY)
    view.linalg = _safe_view(np.linalg)
    view.random = _safe_view(np.random)
    view.fft = _safe_view(np.fft)
    return view


NP = _numpy_view()
RANDOM = _safe_view(_random_module)  # functions and classes only, no route to os
_SAFE_BUILTIN_NAMES = (
    "abs all any bool bytes callable chr dict divmod enumerate filter float format frozenset hasattr "
    "hash int isinstance issubclass iter len list map max min next ord pow print range repr "
    "reversed round set slice sorted str sum tuple zip __build_class__ "
    "Exception ArithmeticError AssertionError AttributeError IndexError KeyError LookupError "
    "NameError NotImplementedError RuntimeError StopIteration TypeError ValueError ZeroDivisionError"
).split()


class CodeRejected(Exception):
    """The block uses something the robot.run sandbox does not allow."""


def _warm_numpy() -> None:
    """Load the numpy submodules that array methods import lazily from C (``a.mean()`` ->
    ``numpy._core._methods``), so those imports find them already loaded."""
    a = np.arange(4.0)
    for f in (a.mean, a.std, a.var, a.sum, a.prod, a.max, a.min, a.any, a.all, a.cumsum, a.argmax, a.argmin):
        f()
    a.round(1), a.clip(0, 1), np.float64(1.5).round(1), a.reshape(2, 2).mean(axis=0)


_warm_numpy()


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.startswith("numpy.") and name in sys.modules:
        # numpy's C code imports its own (already loaded) submodules through the caller's
        # __import__ and then takes them from sys.modules: hand back only the safe view.
        # Model code cannot name such modules itself (check_code rejects "_" components).
        return NP
    if level != 0 or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in robot.run "
                          f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
    if name == "numpy":
        return NP
    if name == "random":
        return RANDOM
    return builtins.__import__(name, globals, locals, fromlist, level)


def _checked_name(name):
    """The attribute name as an exact str, refused if private or denied (a str subclass could
    otherwise answer startswith() however it likes)."""
    if not isinstance(name, str):
        raise TypeError("attribute name must be a string")
    name = str.__str__(name)
    if str.startswith(name, "_") or name in _DENIED_ATTRS:
        raise AttributeError(f"access to {name!r} is not allowed in robot.run")
    return name


def _guarded_getattr(obj, name, *default):
    return getattr(obj, _checked_name(name), *default)


def _guarded_hasattr(obj, name):
    return hasattr(obj, _checked_name(name))


def _safe_builtins() -> dict:
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES}
    safe["__import__"] = _guarded_import
    safe["getattr"] = _guarded_getattr
    safe["hasattr"] = _guarded_hasattr
    return safe


def check_code(tree: ast.AST) -> None:
    """Reject private/dunder attributes, dunder names and array file methods."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in _DENIED_ATTRS):
            raise CodeRejected(f"line {node.lineno}: attribute {node.attr!r} is not allowed in robot.run")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] + ([node.module] if getattr(node, "module", None) else [])
            for name in names:
                if any(part.startswith("_") for part in name.split(".")):
                    raise CodeRejected(f"line {node.lineno}: import of {name!r} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            raise CodeRejected(f"line {node.lineno}: name {node.id!r} is not allowed in robot.run")


class Executor:
    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        self._lock = threading.Lock()
        self.ns: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        """Fresh namespace for a new scene."""
        ns: dict[str, Any] = {"__name__": "robot_run", "__builtins__": _safe_builtins(),
                              "np": NP, "numpy": NP, "math": math}
        ns.update(self.robot.namespace())
        self.ns = ns
        self.robot.cancel_event.clear()

    def cancel(self) -> None:
        self.robot.cancel_event.set()
        self.robot.arms_backend.stop()

    def run(self, code: str) -> dict[str, Any]:
        """Execute ``code``; returns output, error, fault flag, ledger and elapsed seconds."""
        with self._lock:  # one block at a time
            self.robot.cancel_event.clear()
            out: list[str] = []
            size = 0

            def _print(*args: Any, sep: str = " ", end: str = "\n", **_: Any) -> None:
                nonlocal size
                text = sep.join(str(a) for a in args) + end
                size += len(text)
                out.append(text)

            self.ns["print"] = _print
            self.robot.ledger.drain()
            self.robot.take_shown()
            t0 = time.time()
            error = None
            fault = False
            cancelled = False
            linecache.cache[_FILENAME] = (len(code), None, code.splitlines(True), _FILENAME)
            try:
                tree = ast.parse(code, _FILENAME, "exec")
                check_code(tree)
                compiled = compile(tree, _FILENAME, "exec")
                # Executing model code is the purpose of this tool; see the module docstring
                # for the guard rails and their limits.
                exec(compiled, self.ns)  # noqa: S102
            except Cancelled as exc:
                cancelled = True
                error = f"Cancelled: {exc}"
            except RobotFault as exc:
                if self.robot.cancel_event.is_set():
                    # a cancelled motion / gripper goal fails at its controller: that is the
                    # cancel taking effect, not a hardware fault
                    cancelled = True
                    error = f"Cancelled: {exc}"
                else:
                    fault = True
                    error = _user_traceback(exc)
            except SyntaxError as exc:
                error = "".join(traceback.format_exception_only(type(exc), exc))
            except CodeRejected as exc:
                error = f"CodeRejected: {exc}"
            except Exception as exc:
                error = _user_traceback(exc)
            if not cancelled and not fault and self.robot.cancel_event.is_set():
                cancelled = True                      # cancelled while the last statement ran
                error = error or "Cancelled: robot.run cancelled"
            output = "".join(out)
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[: MAX_OUTPUT_CHARS // 2] + "\n...[output truncated]...\n" + output[-MAX_OUTPUT_CHARS // 2:]
            return {
                "ok": error is None,
                "output": output,
                "error": error,
                "fault": fault,
                "cancelled": cancelled,
                "elapsed_s": round(time.time() - t0, 2),
                "ledger": self.robot.ledger.drain(),
                "images": self.robot.take_shown(),
            }


def _user_traceback(exc: BaseException) -> str:
    """Traceback limited to frames of the model's code plus the final error line."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == _FILENAME]
    lines = ["Traceback (model code):\n"]
    lines += traceback.format_list(frames)
    lines += traceback.format_exception_only(type(exc), exc)
    return "".join(lines)

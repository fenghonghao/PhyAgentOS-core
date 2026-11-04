import numpy as np

from tabletop.types import RobotFault


def test_print_capture_and_persistent_namespace(executor):
    r1 = executor.run("x = 41\nprint('hello', x)")
    assert r1["ok"] and r1["output"] == "hello 41\n"
    r2 = executor.run("print(x + 1)")
    assert r2["output"] == "42\n"
    executor.reset()
    r3 = executor.run("print(x)")
    assert not r3["ok"] and "NameError" in r3["error"]


def test_traceback_shows_model_line(executor):
    r = executor.run("a = 1\nb = a / 0\n")
    assert not r["ok"] and "line 2" in r["error"] and "b = a / 0" in r["error"] and "ZeroDivisionError" in r["error"]


def test_syntax_error(executor):
    r = executor.run("def f(:\n  pass")
    assert not r["ok"] and "SyntaxError" in r["error"]


def test_robot_fault_flag(executor, world):
    def boom(*a, **k):
        raise RobotFault("left: CAN disconnected")
    world.grip = boom
    r = executor.run("set_gripper(0)")
    assert not r["ok"] and r["fault"] and "CAN disconnected" in r["error"]


def test_ledger_in_result(executor):
    r = executor.run("s = locate('red block')[0]\nprint(s.score > 0)")
    assert r["output"] == "True\n"
    assert [e["call"] for e in r["ledger"] if not e["nested"]] == ["locate"]


def test_model_code_can_pick_and_stack(executor, world):
    r = executor.run("""
red, blue = locate("red block")[0], locate("blue block")[0]
print(pick(red).ok)
print(place(blue).ok)
print(state().holding)
""")
    assert r["ok"], r["error"]
    assert r["output"].split() == ["True", "True", "False"]
    b, c = world.boxes["red block"], world.boxes["blue block"]
    assert abs(b.z_bottom - c.top) < 1e-6 and np.allclose(b.center_xy, c.center_xy, atol=0.01)


def test_sandbox_blocks_os_files_and_dunders(executor):
    for code, needle in [
        ("import os\nos.listdir('/')", "not allowed"),
        ("import subprocess", "not allowed"),
        ("open('/etc/passwd').read()", "NameError"),
        ("eval('1+1')", "NameError"),
        ("x = ().__class__", "CodeRejected"),
        ("f = getattr(np, '__loader__')", "not allowed"),
        ("__import__('os')", "CodeRejected"),
    ]:
        r = executor.run(code)
        assert not r["ok"] and needle in r["error"], (code, r["error"])


def test_sandbox_allows_normal_model_code(executor):
    code = """
import numpy as np
import math, time
from collections import defaultdict
class Plan:
    def __init__(self, items):
        self.items = list(items)
d = defaultdict(int)
p = Plan(["red block"])
s = locate(p.items[0])[0]
print(round(math.hypot(*s.center_xyz[:2]), 2) > 0, isinstance(s.size, list), getattr(s, "score") > 0)
"""
    r = executor.run(code)
    assert r["ok"], r["error"]
    assert r["output"] == "True True True\n"


def test_sandbox_blocks_known_escapes(executor):
    for code, needle in [
        ("import random\nrandom._os.system('true')", "CodeRejected"),
        ("import statistics", "not allowed"),
        ("import dataclasses", "not allowed"),
        ("import json", "not allowed"),
        ("import os.path", "not allowed"),
        ("np.ctypeslib", "AttributeError"),
        ("np.load('/etc/passwd')", "AttributeError"),
        ("np.lib", "AttributeError"),
        ("a = np.zeros(3)\na.tofile('/tmp/x')", "CodeRejected"),
        ("import numpy\nnumpy.save('/tmp/x', [1])", "AttributeError"),
    ]:
        r = executor.run(code)
        assert not r["ok"] and needle in r["error"], (code, r["error"])


def test_numpy_view_keeps_everyday_math(executor):
    code = """
import numpy as np, math
v = np.array([3.0, 4.0])
print(float(np.linalg.norm(v)), round(float(np.deg2rad(180)), 4), np.add([1, 2], [3, 4]).tolist(),
      isinstance(v, np.ndarray), np.random.uniform(0, 1) < 1, round(math.hypot(3, 4)))
"""
    r = executor.run(code)
    assert r["ok"], r["error"]
    assert r["output"] == "5.0 3.1416 [4, 6] True True 5\n"


def test_frame_introspection_is_rejected(executor):
    r = executor.run("def g():\n    yield 1\nx = g()\nprint(x.gi_frame.f_back.f_globals)")
    assert not r["ok"] and "not allowed" in r["error"]


def test_array_methods_work_and_numpy_internals_stay_out_of_reach(executor):
    r = executor.run("xy = np.array([[0.1, 0.2], [0.3, 0.4]])\n"
                     "print(np.round(xy.mean(axis=0), 3), round(float(xy.std()), 4), xy.max(axis=0) - xy.min(axis=0))")
    assert r["ok"], r["error"]
    for code in ("import numpy._core._methods", "from numpy._core import _methods", "import numpy.core as c"):
        r = executor.run(code)
        assert not r["ok"], code
    r = executor.run("import numpy.linalg\nprint(numpy.linalg.norm([3, 4]))")   # gets the safe view
    assert r["ok"] and r["output"].strip() == "5.0", r


def test_sandbox_closes_known_escapes(executor):
    escapes = [
        # a str subclass that lies in startswith() must not get past getattr
        "class S(str):\n    def startswith(self, *a):\n        return False\nprint(getattr(len, S('__self__')))",
        "print(type(1))",                                   # no type(): its 3-argument form builds classes
        "print(object)",
        "a = np.zeros(2)\nprint(a.ctypes)",                 # raw pointers
        "a = np.zeros(2)\nprint(getattr(a, 'ctypes'))",
        "print(getattr(len, '__self__'))",
        "print(hasattr(len, '__self__'))",
        "import random\nprint(random.Random)\nprint(getattr(random, '_os'))",
    ]
    for code in escapes:
        r = executor.run(code)
        assert not r["ok"], code
    r = executor.run("import random\nrandom.seed(1)\nprint(0 <= random.random() < 1, getattr(state(), 'holding') is not None)")
    assert r["ok"] and r["output"].strip() == "True True", r


def test_fault_during_a_cancel_is_reported_as_cancelled(executor, robot):
    """Cancelling a running gripper / motion goal makes its controller answer with a failure
    code; that is the cancel taking effect, not a hardware fault."""
    def interrupted():
        robot.cancel_event.set()                       # what Executor.cancel() does
        raise RobotFault("left: gripper CANCELED: goal cancelled")

    executor.ns["interrupted"] = interrupted
    r = executor.run("print('start')\ninterrupted()")
    assert r["cancelled"] and not r["fault"] and not r["ok"]


def test_block_that_ends_after_a_cancel_is_marked_cancelled(executor, robot):
    executor.ns["cancel_now"] = robot.cancel_event.set
    r = executor.run("cancel_now()\nx = 1")
    assert r["cancelled"] and not r["fault"]


def test_a_real_fault_without_cancel_is_still_a_fault(executor, robot):
    def broken():
        raise RobotFault("left: NO_FRESH_ROBOT_STATE: driver silent")

    executor.ns["broken"] = broken
    r = executor.run("broken()")
    assert r["fault"] and not r["cancelled"]


def test_gripper_cancel_code_is_not_a_fault():
    from tabletop.dora_backend import gripper_fault

    assert gripper_fault("succeeded", "SUCCESS", "") is None
    assert gripper_fault("aborted", "STALLED", "") is None
    assert gripper_fault("canceled", "CANCELED", "goal cancelled") is None
    assert gripper_fault("cancelled", "CANCELLED", "") is None
    assert gripper_fault("aborted", "HARDWARE_ERROR", "motor") == "gripper HARDWARE_ERROR: motor"

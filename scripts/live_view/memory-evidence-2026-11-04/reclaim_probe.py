import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('reclaim', '/home/bsb/PhyAgentOS-core/example/piper-dual-tabletop/nodes/dora_shm_guard.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import dora
original = dora.Node
dora.Node = lambda *args, **kwargs: module.ReclaimingNode(original(*args, **kwargs))
exec(compile(Path('/tmp/dora-shm-investigation/probe.py').read_text(), '/tmp/dora-shm-investigation/probe.py', 'exec'))

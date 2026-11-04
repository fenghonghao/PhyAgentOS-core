"""Run separate local dataflows, recording only synthetic frame ownership metrics."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
python = '/home/bsb/.PhyAgentOS/skills/.venvs/piper-vla/bin/python'
prefix = os.environ.get('PROBE_PREFIX', 'baseline')
cases = [('fast', 0)] if os.environ.get('PROBE_FAST_ONLY') else [('fast', 0), ('slow', 0.1), ('paused', 2)]
for label, delay in cases:
    label = f'{prefix}-{label}'
    config = root / f'{label}.yaml'
    config.write_text(f'''nodes:
- id: synthetic_sender
  path: {root}/probe.sh
  args: sender
  inputs:
    tick: dora/timer/millis/33
  outputs: [frame, _shm_reclaim]
- id: synthetic_receiver
  path: {root}/probe.sh
  args: receiver {delay}
  inputs:
    frame: {{source: synthetic_sender/frame, queue_size: 1}}
''')
    with (root / f'{label}.log').open('w') as log:
        proc = subprocess.Popen([os.environ.get('PROBE_DORA', 'dora'), 'run', str(config)], cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                rc = proc.wait(timeout=5)
        print(json.dumps({'case': label, 'exit_code': rc, 'log': str(root / f'{label}.log')}), flush=True)

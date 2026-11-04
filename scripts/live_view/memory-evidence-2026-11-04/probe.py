"""Bounded synthetic Dora 0.4.1 shared-memory reproduction; no hardware access."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from dora import Node


def snapshot():
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if '/dev/shm/shmem_' in line and not line.endswith('(deleted)')}
    sizes = [Path(path).stat().st_size for path in paths if Path(path).exists()]
    return {'segments': len(sizes), 'mapped_mib': round(sum(sizes) / 2**20, 3)}


role = sys.argv[1]
node = Node()
start = time.monotonic()
count = 0
last_report = -1
if role == 'sender':
    payload = pa.array(np.zeros(2**20, dtype=np.uint8))
    for event in node:
        if event['type'] == 'STOP':
            break
        if event['type'] != 'INPUT':
            continue
        elapsed = time.monotonic() - start
        if elapsed >= 14:
            break
        for _ in range(int(os.environ.get('PROBE_BURST', '1'))):
            node.send_output('frame', payload)
            count += 1
        if int(elapsed) != last_report:
            sample = snapshot()
            print(json.dumps({'role': role, 't': round(elapsed, 3), 'count': count, **sample}), flush=True)
            last_report = int(elapsed)
            if sample['mapped_mib'] > 512:
                print('memory safety ceiling reached', flush=True)
                break
else:
    delay = float(sys.argv[2])
    for event in node:
        if event['type'] == 'STOP':
            break
        if event['type'] == 'INPUT':
            # Deliberately retain only the current event, never an application queue.
            count += 1
            if delay:
                time.sleep(delay)
    print(json.dumps({'role': role, 'received': count, 'elapsed': round(time.monotonic() - start, 3)}), flush=True)
del event
del node

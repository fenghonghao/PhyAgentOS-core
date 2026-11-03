"""Machine self-test helpers (used by nodes/vla/selftest.sh): synthetic 14-D dataset, load + infer.

  python -m piper_vla.selftest record <dataset_root>     # 3 episodes, then resume + 1 more
  python -m piper_vla.selftest infer <pretrained_model_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np

from piper_vla.spec import ArmSpec, CameraSpec, VlaSpec

SPEC = VlaSpec(arms=[ArmSpec("left"), ArmSpec("right")],
               cameras=[CameraSpec("head"), CameraSpec("wrist_left"), CameraSpec("wrist_right")], fps=30)


def episode(rec, n=60, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(n):
        q = np.sin(np.linspace(0, 1, SPEC.dim) * 3 + t / 10.0).astype(np.float32) * 0.5
        imgs = {c.name: (rng.integers(0, 255, (240, 320, 3))).astype(np.uint8) for c in SPEC.cameras}
        rec.add(q, q + 0.01, imgs, "pick up the cube")
    return rec.save()


def record(root):
    from piper_vla.recorder import Recorder
    root = Path(root).expanduser()
    t0 = time.time()
    rec = Recorder(SPEC, root, "local/" + root.name)
    for i in range(3):
        print("saved frames", episode(rec, seed=i), "t=%.1f" % (time.time() - t0))
    rec.close()
    rec = Recorder(SPEC, root, "local/" + root.name)
    print("resumed with", rec.episodes, "episodes")
    episode(rec, seed=9)
    rec.close()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/" + root.name, root=root)
    print("episodes", ds.meta.total_episodes, "frames", ds.meta.total_frames)
    st = ds.meta.stats["action"]
    print("action stat keys", sorted(st.keys()))
    item = ds[0]
    print({k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")})


def infer(path):
    from piper_vla.policy_node import PolicyEntry, load_policy
    import torch
    e = PolicyEntry(name="t", path=path, n_action_steps=10)
    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    t0 = time.time()
    load_policy(e)
    print("loaded in %.1fs" % (time.time() - t0), type(e.policy).__name__,
          "gpu mem %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0))
    rng = np.random.default_rng(0)
    e.policy.reset()
    dev = e.device_used
    for k in range(25):
        batch = {"observation.state": torch.zeros(1, SPEC.dim, device=dev), "task": ["pick up the cube"]}
        for c in SPEC.cameras:
            batch[f"observation.images.{c.name}"] = torch.from_numpy(
                rng.integers(0, 255, (240, 320, 3)).astype(np.uint8)).to(dev).permute(2, 0, 1).float().div(255).unsqueeze(0)
        t1 = time.time()
        with torch.inference_mode():
            a = e.post(e.policy.select_action(e.pre(batch)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if k in (0, 1, 10, 24):
            print("step", k, tuple(a.shape), "%.1f ms" % ((time.time() - t1) * 1000), a.squeeze(0)[:3].tolist())


if __name__ == "__main__":
    {"record": record, "infer": infer}[sys.argv[1]](sys.argv[2])

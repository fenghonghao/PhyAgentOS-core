# Dora 0.4.1 shared-memory growth — 2026-11-04

The failure was reproduced without a camera, browser, or dynamic node. The producer's
release-notification channel uses `flume::bounded(0)`: sending new large messages can
outpace reclaiming old shared-memory allocations. Upstream fixed this in
[PR #1417](https://github.com/dora-rs/dora/pull/1417), merged March 17, 2026. The fix is
present in the v0.5.0 source. Merely upgrading the CLI or `live_view` does not patch
the Dora library embedded in the frozen Orbbec adapter. A coordinated upgrade also
needs to account for the change from message protocol 0.7 to 0.8.

## Deployed compatibility fix

Plan A now launches the official Orbbec adapter v1.1.0 Python source using its existing
skill-server Python environment, with adapter dependencies isolated under
`~/.PhyAgentOS/camera-nodes/orbbec-camera-1.1.0-shmfix`. Dora remains 0.4.1.

`example/piper-dual-tabletop/nodes/dora_shm_guard.py` services release notifications
after each image by sending four one-byte messages on the unsubscribed `_shm_reclaim`
output. These inline messages do not allocate shared-memory images. If allocations
exceed 24 mapped segments, it stops sending images and services acknowledgements
until usage drops; after two seconds without recovery it raises an error and the
adapter closes the camera. This is a bounded workaround, not a backport of #1417.

Both camera outputs declare `_shm_reclaim` in the real-piper dataflow. The launchers,
guard, and declarations were synchronized to the installed Plan A skill without
changing its camera resolution or starting the robot controller. Other profiles
using this launcher must also declare this private output.

The start hook installs the isolated dependencies when absent. To install manually,
run `bash example/piper-dual-tabletop/scripts/setup_guarded_camera.sh` with
`SKILL_SERVER_VENV` pointing to the skill-server environment if needed. The installer
verifies the adapter source archive's SHA-256 before installing it.

## Validation

Synthetic producer: one 1 MiB Arrow payload sent in a burst every 33 ms;
receiver immediately consumes messages, with `queue_size: 1`.

| Case | Observation |
| --- | --- |
| Original, 1 output per tick | 4 MiB after warm-up, 14 s run |
| Original, 2 outputs per tick | 193 MiB at 13 s, increasing |
| Original, 4 outputs per tick | 413 MiB at 13 s, increasing |
| Guarded, 4 outputs per tick | 17 MiB at 13 s, 1,692 messages received; exit 0 |
| Installed guarded head camera, RGB + depth | 856 messages in 14.37 s; 10.25 MiB steady shared-memory segments; clean stop, exit 0 |

The first guarded synthetic run was interrupted by a shared-memory segment disappearing
(`failed to map shared memory output`, OS error 2). Its result was discarded; the
repeat above completed successfully. The cause of that disappearance was not proven.

The real-camera smoke test finished at 17:05:16 +08. These are short reproductions
and smoke tests, not an hours-long soak test or a full robot-task acceptance test.
Source and logs are preserved in the sibling `memory-evidence-2026-11-04` directory.
No global `/dev/shm` cleanup is part of this fix.

## Follow-up recovery and shutdown fix (18:23 +08)

The head camera later lost USB connectivity (17:43:23) and subsequent starts
intermittently produced color but no depth. Changing only the head depth profile
from 640x400/15 FPS to 640x400/30 FPS restored paired frames in the tested startup;
color remains 1920x1080/15 FPS and the Dora publish timer remains 100 ms. This is a
validated operating configuration, not proof of the original USB disconnect's cause.

A separate reproducible adapter shutdown bug was fixed: the capture child tried
to flush its multiprocessing image queue after the parent stopped reading. The
parent consequently killed it after three seconds (exit code -15). The packaged
`scripts/patch_orbbec_shutdown.py` calls `frame_queue.cancel_join_thread()` inside
the capture worker, allowing its existing `backend.close()` and normal process
exit to complete. The patch is applied by setup and on each start, including
existing dependency installations. No capture retry loop was added.

The installed skill also now includes the live view and launches it from the
start hook. The ready bundle was built from the installed skill plus these camera
and live view changes, preserving the installed robot action implementation:
`dist/piper-dual-tabletop-live-view-fix-20261104.tar.gz`.
SHA-256: `feb6afd85407b39f8f5963bd4f0c17f92fef05e6b3dc1b2ddafefdcfa3e79d65`.

Fresh validation:

- Actual exposure measurement → isolated camera: 72 paired frames in 5 seconds;
  before the shutdown patch, close took 3.01 s and exit code was -15.
- Patched isolated camera: 74 paired frames in 5 seconds; close took 1.00 s,
  exit code 0.
- Hardware-free regression with a full image pipe and no reader: unpatched worker
  hangs, patched worker exits normally; 2 tests passed on Python 3.12.
- Full installed `paos skill start`: preflight passed, gateway and `robot.run` ready.
- WebSocket smoke test decoded all four streams: head RGB/depth 1920x1080,
  wrist RGB/depth 640x480; 101/101/243/243 frames received in 10 seconds.
- 120 s, 25 samples: head ~10 FPS; wrist 28.3–30.3 FPS. Sender shared memory
  remained 41.54 MiB (head) and 14.41 MiB (wrist). Live view RSS warmed from
  226.7 MiB to approximately 480–491 MiB in the final 30 seconds. This short
  observation does not prove hours-long stability.

Evidence is under `memory-evidence-2026-11-04`, including
`live-view-followup-120s.jsonl`, `acceptance30.log`, and `fixed-shutdown30.log`.

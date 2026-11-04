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

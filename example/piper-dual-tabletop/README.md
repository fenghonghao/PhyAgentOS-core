# piper-dual-tabletop (paos skill)

Single-arm AgileX PIPER tabletop manipulation for PhyAgentOS (the competition booth: one arm,
a pole-mounted head camera, a wrist camera): move target objects into containers, sort objects
into groups, arrange number blocks into an equation. The model (DeepSeek) writes Python blocks
that run through one Forge Action tool, `robot.run`; a block calls perception (SAM3
segmentation, a vision-language model through `look` / `inspect` / `orient`), motion and
gripper primitives and `pick` / `place` / `drag` (API: `SKILL.md`). The name
is historical; the code still supports two arms (profiles kept in `dev/legacy/`).

## Layout

| Path | Content |
|---|---|
| `skill.yaml` | Manifest: profiles `real-piper`, `sim-piper`; locked registry nodes |
| `SKILL.md` | Workflow and robot API for the model |
| `profiles/real-piper/` | PIPER driver, motion_server, trajectory and gripper controllers, head + wrist Orbbec cameras, gateway, skill_server (with `vision:` for `look`) |
| `profiles/sim-piper/` | Single-arm MuJoCo profile to check the whole chain without hardware |
| `nodes/skill_server/` | `tabletop` package: primitives, skills, executor, Dora node, IK, perception (SAM3), vision model client, tests |
| `nodes/run_skill_server.sh`, `nodes/run_camera.sh` | Node launchers (venv / camera release binaries) |
| `start.sh` | Builds the node venv outside the bundle, installs pinned deps, pre-flight checks |
| `run_scene.sh` | Competition entry: one scene per call, writes the 180 s deadline, logs start/end |
| `scripts/` | `make_bundle.sh`, `fetch_camera_nodes.sh`, `calib_compose.py`, `calibrate/` (camera calibration tool and printable board) |

## Install and run

```bash
./start.sh                       # venv + SAM3 deps + checks (CAN, video group, GPU, weights)
nodes/skill_server/download_sam3.sh   # SAM3 weights (ModelScope mirror of facebook/sam3), if missing
scripts/fetch_camera_nodes.sh    # camera adapter binaries (sha256-verified), if missing
paos skill install --local <bundle.tar.gz> -y
paos skill start piper-dual-tabletop --profile real-piper
./run_scene.sh <scene_id> "<task text>"
```

Calibrate the cameras with `scripts/calibrate/calibrate.py` (own venv from
`scripts/calibrate/requirements.txt`). Print `charuco_a4_7x5_38mm.pdf`, measure a printed square with
a ruler (printers scale pages: ours came out at 36 mm, not 38), tape the sheet flat on the table, then:

```bash
calibrate.py check RUN                  # both cameras see the sheet (no motion)
calibrate.py collect RUN --square-mm 36 # about 5 min; --head-only if the wrist camera was not touched
calibrate.py solve RUN                  # prints the cameras: block for profiles/real-piper/skill_server.yaml
                                        # (--merge OTHER_RUN combines runs with the same camera and sheet)
calibrate.py point RUN                  # closed fingertips hover over corners the head camera located
```

Recalibrate the head camera if it or its pole moved (0.5 degree moves points on the table by about
7 mm). To tell, compare the sheet's corners in the head image, not the background: the booth panels and
boxes do move. Fill the booth's panels into `robot.workspace`, and the judges' reading direction into `robot.read_up_deg` /
`read_right_deg`. `look` uses the PAOS DeepSeek key (`providers.deepseek` in
`~/.PhyAgentOS/config.json`, or `DEEPSEEK_API_KEY`) and saves every image it sends to
`~/.PhyAgentOS/logs/piper-dual-tabletop/look/`.

Offline check of SKILL.md with the real model on the three task types (fake world, oracle vision):
`nodes/skill_server/.venv/bin/python dev/llm_trial_booth.py --runs 2`.

## Design rules

- Scenes end only when the model decides (`forge_task_finalize`) or the referee stops the robot:
  no timeouts or time-based stopping anywhere in this skill. Every model request and the
  `time_left()` primitive show the seconds left, as information only.
- Only what a program cannot compose from other primitives is a primitive. Perception is four
  functions (`locate`, `show`, `look`, `capture`) that never move the arm; task logic is the
  model's code.
- Inputs whose loss would wedge an Action (tool messages, motion/gripper goals and results) use
  deep Dora queues (`queue_size: 1000`); joint states and images keep only the latest message.

## Known risks

- **Model code runs inside the skill_server process.** Guard rails: restricted builtins, an
  import allow-list (numpy view without file I/O or ctypes, math, time, itertools, functools,
  collections, random) and an AST check rejecting private/dunder attribute access. This stops
  accidental misuse by the model, but it is not a security boundary: an OS-level sandbox (separate
  process, no network, read-only filesystem, RPC to the primitives) is not implemented. The code
  comes from the configured LLM prompted with the referee's task and this skill's own outputs.
- A top-down gripper pose reaches at most about 0.116 m above the arm base (0.25 m out; PIPER
  joint5 limit), less farther out, and nothing beyond about 0.46 m; higher or farther poses need a
  lean. `pick`, `place` and `drag` choose the smallest lean that reaches, separately for the hover
  and the point below it.
- SAM3 costs about 0.55 s per camera frame on an RTX 5060 laptop GPU.
- pi0.5 needs about 8 GB of GPU memory for inference: it runs only on the RTX 4090 workstation,
  not next to SAM3 on the 8 GB laptop. Demonstrations come from Cartesian teleoperation (no
  leader arms), so they are slower than leader-follower demonstrations.

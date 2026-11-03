# piper-vla (paos skill)

Learned visuomotor policies on one AgileX PIPER arm, as its own paos skill: teleop data
collection into LeRobot datasets, ACT / pi0.5 training and export, and on the robot four
Forge tools (`vla.policies`, `vla.state`, `vla.home`, `vla.run`; API for the model: `SKILL.md`).
It does not depend on any other skill bundle.

## Layout

| Path | Content |
|---|---|
| `skill.yaml` | Manifest: profiles `real-piper` (booth: driver, head + wrist Orbbec cameras) and `sim-piper` (MuJoCo); locked registry nodes |
| `profiles/*/dataflow.yaml` | gateway <-> `vla_server` (tools) -> `vla_policy` (policy, homing) -> driver / simulator |
| `profiles/*/vla_policy.yaml` | Data layout (fps, arm, cameras) and the trained policies (name, path, sha256, description, task) |
| `nodes/vla/src/piper_vla/` | `tool_server` (Forge endpoint), `policy_node`, `collect_node` + `recorder` + `teleop` + `ui`, `spec` (shared data layout), `kinematics` (teleop IK) |
| `nodes/vla/src/forge_tool/` | Vendored Forge tool endpoint library (see its VENDORED.md) |
| `nodes/vla/*.sh` | `setup_venv.sh`, `collect.sh`, `train.sh`, `export_model.sh`, `selftest.sh` |
| `start.sh` | Start hook: VLA venv, policy weight checks, CAN / camera checks on the real robot |
| `scripts/` | `make_bundle.sh`, `fetch_camera_nodes.sh` |
| `dev/gw_call.py` | Call one tool straight through the Gateway (bench test without the agent) |

## Collect, train, deploy

```bash
nodes/vla/setup_venv.sh [--pi05]                      # VLA venv, PaliGemma tokenizer cache, ACT backbone, pi05_base
nodes/vla/collect.sh real --dataset cup --task "..."  # teleop (keyboard / gamepad) + LeRobot recording; nothing else may drive the arm
nodes/vla/train.sh act cup                            # or: train.sh pi05 cup   (RTX 4090 workstation)
nodes/vla/export_model.sh <run_dir> cup "<task, scene, demo length>" "<instruction>"
# paste the printed entry into profiles/real-piper/vla_policy.yaml, then
paos skill install --local <bundle.tar.gz> -y
paos skill start piper-vla --profile real-piper
```

`vla.run` sends the policy's joint and gripper targets straight to the driver for the
`seconds` the model asks for; the arm then holds the last target. `vla.home` moves along a
smooth joint path to the demonstrations' start pose. Nothing in the skill stops on time.
Settings marked `HARDWARE-LIMITED` were chosen for the RTX 5060 laptop; raise them on the
workstation (`grep -rn HARDWARE-LIMITED .`).

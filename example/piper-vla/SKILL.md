# piper-vla

One AgileX PIPER arm driven by learned visuomotor policies (ACT or pi0.5, trained with
LeRobot on demonstrations recorded on this robot). A policy looks at the head and wrist
cameras and moves the arm and gripper on its own; you choose which policy runs and for how
long.

## Tools

| Tool | Kind | Arguments | Result |
|---|---|---|---|
| `vla.policies` | Query | none | `policies`: `name`, `state` (`ready` / `loading` / `error`), `description` (task, scene and length of the demonstrations), `task` (instruction text) |
| `vla.state` | Query | none | `joints` (rad) and `gripper_width_m` per arm, the `run` or `home` in progress |
| `vla.home` | Action | none | `reached`, `error`, `elapsed_s`: arm at the demonstrations' start pose, gripper open |
| `vla.run` | Action | `policy`, `seconds`, optional `task` | `steps`, `rate_hz`, `infer_ms`, `error`, `cancelled`, `gripper_width_m` |

## Workflow

1. Create the task with `forge_task_create` (criteria = what the scene must look like).
2. `forge_tool_query("vla.policies")`: pick the policy whose `description` matches the task.
   Use a policy only for the task it was trained on.
3. `forge_tool_start_action(task_id, "vla.home", {}, wait=true)`: every demonstration started
   at this pose with the gripper open; start the policy the same way.
4. `forge_tool_start_action(task_id, "vla.run", {"policy": ..., "seconds": ..., "task": ...},
   wait=true)`. Do not pass `timeout_ms`.
   - `seconds`: about the length of one demonstration (in the policy's `description`). The
     policy does not know when the task is done; after `seconds` the arm holds still.
   - `task`: the instruction text for language-conditioned policies (pi0.5); use the policy's
     own `task` text or the task wording. ACT ignores it.
   - `steps == 0` or `error` set: the policy did not act (still loading, failed to load, or no
     camera images); read `vla.policies` / `vla.state`.
5. `forge_tool_query("vla.state")` to see where the arm and gripper ended (a gripper narrower
   than fully open but not closed usually means it holds something). Run the same policy
   again for more time, another policy for the next step, or `vla.home`.
6. When the scene is done (or you decide to stop), call `forge_task_finalize`.

Every request shows `[time_left_s=…]`, the seconds left in the scene, as information only.
Deciding when to finish is up to you.

## Rules

- There is no tool for raw joint commands; the arm moves only through `vla.home` and `vla.run`.
- A cancelled `vla.run` or `vla.home` stops at once and the arm holds where it is.
- If a tool reports `POLICY_NODE_DOWN`, the policy process stopped: finalize with that in
  the summary.

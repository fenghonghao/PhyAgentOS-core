# Official PyRUA-Lean source review

Inspected official main at commit `e8f01dca744421d70684adbbc1713fd4d92f8aca` on 2026-11-04; local read-only clone `/tmp/pyrua-upstream-review`. Sources below pin this commit. These are code findings, not evidence of real Piper performance.

## Is PyRUA-RSI a handbook, and what transfers to handbook RSI?

### Takeaway
Official released offline PyRUA-RSI is fixed Python per task, but that does not require the user's RSI to use the same artifact. A handbook is a reasonable alternative improvement artifact; upstream success rates do not measure it.

### Cited Findings
- RSI README explicitly states fixed program per task and no evaluation-time language model; candidate programs were written by a code-writing agent loop and retained by simulator performance. [README lines 5, 51–57](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/integrations/xpolicylab/PyRUA_RSI/README.md#L5-L57).
- This is implemented, not just wording: `program_for` loads `programs/<task>.py`; the first chat response emits the whole file as a `python` tool call, subsequent responses emit `finish` with a hardcoded summary. This finish status is not the benchmark verdict. [offline_programs.py 25–64](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/integrations/xpolicylab/PyRUA_RSI/offline_programs.py#L25-L64).
- The adapter points the OpenAI runtime to this local fake chat endpoint. [model.py 65–70](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/integrations/xpolicylab/PyRUA_RSI/model.py#L65-L70).
- pack v20 source retains version history, bounded per-object repairs, box remeasurement, explicit move-result checking, and home retry. [pack_objects_into_box.py 1–22](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/integrations/xpolicylab/PyRUA_RSI/programs/pack_objects_into_box.py#L1-L22), [55–98](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/integrations/xpolicylab/PyRUA_RSI/programs/pack_objects_into_box.py#L55-L98).
- Main Lean RoboDojo prompt has one knowledge text for all tasks and no task recipes. Thus a per-task handbook adds task-conditioned experience beyond this baseline. [RoboDojo docs 25–26](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/docs/robots/robodojo.md#L25-L26).

### Inferences
- Transferable mechanism: run → diagnose → record reusable condition/action/recovery → test revised artifact on held-out layouts. The artifact can be a handbook; define success by measured improvements, not matching released offline programs.
- Strong handbook entries should encode when a step is applicable, preconditions, observable postconditions, stop/replan rules, and failure-specific repairs, rather than prescribe exactly three calls or a fixed target list.
- Distinguish `our handbook RSI` and `released PyRUA-RSI offline`; do not present 82.82% as a predicted handbook outcome or generic proof that task templates close the weak-model gap.

### Gaps
- Did not find an upstream controlled evaluation of task handbook optimization versus fixed programs. RSI separate technical report remains forthcoming in README.

## What do locate, look, capture, show and geometry actually do?

### Takeaway
Locate maps well to the proposed geometry measurement, with important caveats. Look semantic subcalls and capture bundles are local design choices, not upstream APIs; upstream shows images directly to the planner.

### Cited Findings
- RoboDojo depth-capable `locate` returns one best-matching `ObjectPose`, not all instances. It is read-only and SAM3 match is not proof of identity. [robodojo.py 794–857](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/robodojo.py#L794-L857).
- Actual mask handler projects mask pixels using the depth and calibration of the same observation, then computes geometry. Missing measurement can return found=true but `geometry_error`; not-found is separate. [robodojo_tools.py 1185–1266](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/hosts/robodojo_tools.py#L1185-L1266).
- Geometry removes points within 4 mm of table, rejects fewer than 12 points, trims distant points, takes top at z 97th percentile, estimates z midpoint from table and top, derives footprint orientation and gripper-across-short-side yaw; visible surface is not complete shape. [geometry implementation 832–860](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/hosts/robodojo_tools.py#L832-L860), [footprint 755–787](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/hosts/robodojo_tools.py#L755-L787).
- Official leaderboard/default RoboDojo adapter is RGB-only: geometry uses masks from other cameras plus calibrated camera poses; height with one view is a guess; cached previous good height may be reused. It explicitly recommends wrist remeasurement for this RGB setup. Do not confuse this with the depth setup. [robodojo_xpl_rgb.py 228–267](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/robodojo_xpl_rgb.py#L228-L267).
- `image()` loads an RGB numpy array. RoboDojo explicitly has NO recorded world maps: `world_map` raises and directs to per-pixel `back_project`; some other benchmark adapters support world_map. [robodojo.py 667–692](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/robodojo.py#L667-L692), [base world_map 197–218](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/_robot.py#L197-L218).
- `show` requests an image attachment for the next planner turn, no semantic model call inside the primitive. [robodojo.py 706–715](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/robodojo.py#L706-L715). Runtime appends tool text, then user content with images. [openai_runtime.py 554–585](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/openai_runtime.py#L554-L585).
- Framework also permits `display` on numpy/PIL images to attach custom crops/overlays. [arms.py 197–199](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/arms.py#L197-L199), [355–381](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/arms.py#L355-L381).

### Inferences
- Proposed locate-all list is sensible for classification and numbered semantic queries, but it adds deduplication, consistent ordering and identity association responsibilities not tested by upstream locate-one.
- Proposed `capture()->Frame(rgb, world)` is a good ergonomic adaptation for depth-enabled local robot, provided RGB, depth, wrist extrinsics and frame identity are captured atomically. A common image grid does not alone prove temporal registration.
- `look` as a stateless visual worker can reduce planner turns but loses planner-visible raw visual evidence and long-context grounding. Treat it as a design tradeoff; it is not what upstream show implements. Require structured outputs with unknown/ambiguous states and references to object indices.
- Physical long-axis angle has 180-degree ambiguity and is not semantic character-up direction; equation orientation needs separate semantic verification.

### Gaps
- Search found no `look` or `capture` primitive definitions in src; cannot cite such features as implemented upstream.
- No source evidence proves a mask's low-z points give a box opening center or that all box floor points are inside the opening; requires actual mask/depth data.

## Persistence, feedback boundaries and primitive failures

### Takeaway
Upstream supports persistent cells and ordinary code-level checks. It does not require exactly three phases, and existing result objects distinguish command completion from task success.

### Cited Findings
- Cell namespace holds robo/math/time/display; exec runs in shared `_globals`; exceptions are returned as tracebacks rather than silently handled. [arms.py 177–190](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/arms.py#L177-L190), [393–443](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/arms.py#L393-L443).
- Read-only/backend errors are logged and re-raised; motion results are typed. [base _readonly/_stateful 294–349](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/_robot.py#L294-L349).
- `move_to` says test `reached`, reports blocked and ik_error; `holding` is explicitly not proof the intended object is held. [robodojo.py 915–990](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/robodojo.py#L915-L990).
- Upstream observed old RoboDojo images one observation behind robot state and uses image_state to recover matching wrist camera pose. This is a concrete example of synchronization hazards. [robodojo_tools.py 885–897](https://github.com/DAGroup-PKU/PyRUA-Lean/blob/e8f01dca744421d70684adbbc1713fd4d92f8aca/src/pyrualean/hosts/robodojo_tools.py#L885-L897).

### Inferences
- “Observe/act/verify” is useful handbook organization, not a fixed three-call protocol. Yield where information is uncertain, an action changes scene geometry, or a failure cannot be safely classified in code; batch routine steps that have explicit checks.
- Pasted `grasp`/`drop_into` helpers must check every movement before lowering/closing/releasing; printing holding after a whole loop is not enough.
- Do not reuse stale locate results after nearby pushes, box motion, regrasp or camera movement. An explicit `frame_id`/timestamp, `depth_valid` and measurement-quality state would make handbook rules enforceable.

### Gaps
- Local PAOS persistence, image transport and vision backend claims must be checked separately; upstream facts cannot validate them.

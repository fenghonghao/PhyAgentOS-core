# piper-dual-tabletop

One AgileX PIPER arm in a tabletop booth (head camera on a pole behind the arm, a camera on the
wrist). Tasks: move target objects into containers, sort objects into groups, arrange number
blocks into an equation. You control the robot by writing Python that runs on the robot
computer through one tool, `robot.run(code)`, and you see what its cameras see with `show()`.

## Workflow

1. Create the task with `forge_task_create` (criteria = checkable sub-goals, e.g.
   "the peach is inside the cardboard box").
2. Run code with `forge_tool_start_action(task_id, "robot.run", {"code": ...}, wait=true)`:
   it returns the block's result in the same call. Do not pass `timeout_ms`.
   `robot.run` is the only tool of this skill: an Action whose arguments are exactly
   `{"code": "<python source>"}` and whose result has `ok`, `output`, `error`, `elapsed_s`,
   `images`. No need to call `forge_tool_context` for it. These three tools are all a scene
   needs: this document replaces reading files or other skills.
3. Work in short blocks: **see, act, check**. Variables and functions stay defined from block
   to block.
   - End a block where you need to look before deciding. The first block of a task only looks
     and touches nothing, so that a wrong count or a missing object shows before anything moves.
   - Inside a block, decide from returned values in code (skip a missing object, try another
     word, stop when `ok` is False) instead of coming back for every step.
4. When the scene is done (or you decide to stop), call `forge_task_finalize`.

**What comes back from a block:** only what it printed, and the images of its `show()` calls,
which you see right after the result, in call order. Nothing else reaches you: results stay in
variables until you print them.

Every request shows `[time_left_s=…]`, the seconds left in the scene. `time_left()` returns
the same number inside code. Deciding when to finish is up to you.

**The task is a sentence**, such as 把猕猴桃放进盒子里, with no target image: it names what to
move and where. Match those names to what you see in the head image.

<!-- BEGIN API (also used verbatim by the offline evaluation harness) -->
## Robot API inside `robot.run`

Available names: the functions below, `np` (numpy), `math`, `print`.
Imports are limited to numpy (array math, `np.linalg`, `np.random`; no file I/O), math, time,
itertools, functools, collections and random. There is no file, process or network access;
attributes starting with `_` and names starting with `__` are rejected.

**Conventions.** World frame = the arm's base frame: metres, x forward (away from the arm),
y to the arm's left, z up; the table surface is at `TABLE_Z`. Angles in degrees.
`yaw_deg` is the heading of the gripper's closing line seen from above (from +x toward +y); half a
turn is the same line. `tilt_deg` is how far the fingers lean away from the arm's base: 0 = straight
down, 30 = pointing 30 degrees outward. The two are independent.
Domain outcomes come back as values; misuse raises `ValueError`; hardware faults raise
`RobotFault` and end the block. Any exception ends the block with its traceback in `error`;
what was printed and shown before it still comes back.

### Rules that shape every task

- Pushing any object that is not being moved costs points; pushing a container costs as much
  as missing it. Move above other objects and go down and up straight (`pick` and `place` do
  this).
- Nobody changes the scene during a task. If an object or a container is not where it was, your arm
  moved it: nothing senses contact, so a collision shows only in the next image. Say so in your report.
- Positions come from `locate`, identity from your own eyes (`show`): do not estimate
  coordinates from an image, and do not trust a label you have not looked at.
- Perception never moves the arm, and the arm hides what is behind it: `home()` before using
  the head camera.
- Time on the real arm: `pick` about 13 s, `place` about 11 s, `home` about 5 s. Plan with
  `time_left()`; do not redo what already scores.

### Seeing

`locate(text, camera="head") -> list[Sight]` — every instance of `text` in one image of
`"head"` or `"wrist"`, best score first; `[]` when nothing matches. About 0.4 s.
- `text`: one short **English** noun phrase. The segmentation model (SAM3) only understands
  English: translate the task's object names (桃子 → "peach"). For another word, call again:
  `locate("fruit") or locate("toy fruit")`.
- `Sight` fields: `text`, `score`, `camera`, `pixel` [row, col] on that camera's full image,
  `center_xyz` (middle of the footprint rectangle, half its height above the table), `top_z`, `size` [length, width,
  height], `long_axis_deg` (direction of the long side, -90 to 90; nothing about front and
  back), `grasp_yaw_deg`, `grasp_xyz`, `open_width_m`, `color` (typical [r, g, b]), `note`,
  `points` (the object's world points as an N x 3 array; not printed), and for the equation
  task's pieces `symbol`, `top_deg`, `top_every_deg` (see Playbook C; you may set them).
  `print(s)` gives one compact line.
- `note` appears only when the measurement is limited: `cut off at the image edge`;
  `too_wide for the gripper`; `no usable depth: position on the table plane, height unknown`
  (black or glossy objects: then `size` height is 0 and `grasp_xyz` is None).
- The mask is the best match for the words, not proof of identity: check it with
  `show(marks=...)`. A low `score` or a position far from the last one is a sign of the wrong
  object. `locate` takes the object to stand on the table: something held, stacked or inside a
  container is measured wrongly; judge those from the image.

`show(camera="head", marks=None, image=None)` — you see this image right after the block ends.
The picture is taken at the call, so calls in a loop show each moment, in order. Each image stays
in the conversation, so show what the next decision needs. Read the images yourself and do not
send them to the `image` tool: it knows nothing about the task and costs time. In a test the main
model read a starfruit and a kiwi correctly, then asked that tool and took over its wrong answers
(mango, mangosteen). Which number is the target decides the task, not the exact name of a fruit.
A camera image is the camera's whole view; the head image reaches you reduced to 1024 x 576.
- `marks`: Sights or `[x, y, z]` points, drawn as numbered rings 1, 2, … on the camera image,
  in list order. Use it to tell which number is which object.
- `image`: a file path or an array to show instead of a camera image (a crop such as
  `f.rgb[100:200, 300:400]`, in the camera's full pixels). An array up to 1024 pixels
  on its longest side reaches you at full detail, so crop the head image to see something small.

`look(question, camera="head", marks=None, image=None, json=False)` — asks a vision model one
question from inside a block and returns its answer as text (`json=True`: the parsed JSON; ask
for JSON in the question; other text raises `ValueError`). About 1 s. It knows nothing about
the task or earlier questions, so write the whole question. Use it only when code must branch
on an image without waiting for your next turn; for decisions, `show` and look yourself.
`image` (path or array) is sent first; `camera=None` sends no camera image.

`capture(camera="head")` — the current frame as arrays for your own calculations, on one pixel
grid; only what you print comes back. Use it when `locate` does not give what you need.
`f.rgb`: H x W x 3 colours. `f.world`: H x W x 3 world coordinates of each pixel, NaN where
there is no depth, so `f.world[..., 2]` is the height of each pixel. `s.pixel` indexes both.

```text
f = capture("head")                                        # is this spot free?
near = np.linalg.norm(f.world[..., :2] - [0.30, 0.10], axis=2) < 0.05
print(int((near & (f.world[..., 2] > TABLE_Z + 0.01)).sum()))

yellow = (f.rgb[..., 0] > 180) & (f.rgb[..., 1] > 160) & (f.rgb[..., 2] < 100)   # no word finds it
print(f.world[yellow & (f.world[..., 2] > TABLE_Z + 0.004)].mean(axis=0))

n = np.linalg.svd(s.points - s.points.mean(axis=0), full_matrices=False)[2][-1]  # how a flat top leans
```

**Cameras (head 1920 x 1080, wrist 640 x 480, depth aligned to colour).** The head camera sees the
whole table and past it: `show` and `capture` also see the panels and what is beyond them, such as
numbers on a poster. `locate` searches only the middle 1280 x 960 of the head image, so it does not
find those, and they do not matter. Small things are small in this camera, and it sees an object's
top and near side only. On the real table its
measurements repeat within 1 mm from frame to frame. The wrist camera's axis leans 20 degrees
toward the fingers. With a top-down gripper about 0.09 m high straight above a low object, the
object is in the middle of the wrist image, about 20 cm from the camera. A taller object (8 cm
fruit) then comes out low in the image, where depth is unreliable, and too close for depth at
all. Leaning the gripper outward 20 to 30 degrees (`tilt_deg` 20 to 30) lets the camera look
straight down from farther away (calculated, not yet tried on the arm).

A camera can stop (a USB connection dropped on the real rig). Then `locate`, `show`, `look` and
`capture` on it raise `RobotFault` saying so, and that ends the block. Do not look for what is wrong
with the scene: act on the Sights you already have, and say so in your report.

### Handling

`pick`, `place` and `drag` plan their whole route before anything moves. When it cannot be done they
return `ok=False`, `reason="unreachable"`, and the arm and the fingers have not moved. Each makes one
attempt and does no looking: it acts on the `Sight` you give it, so `locate` again after something
moved. The gripper leans only as far as it must: straight down wherever it can, and the hover above
may lean more than the grasp below it. They cross at the higher of the two heights, where the arm
starts and where it hovers: first up on the spot, or first across and then down, so a move out of a
container does not sweep through it. Other objects are not known to them.

`pick(target, z=None, yaw_deg=None, width_m=None) -> Pick` (`ok`, `reason`, `detail`, `width_m`,
`tilt_deg`, `yaw_deg`) — grasps a Sight and lifts it: opens to fit, moves above, straight down,
closes, straight back up. `z`: TCP height of the grasp (default `grasp_xyz`). `yaw_deg`: closing line
(default `grasp_yaw_deg`, the way round the wrist turns least; given, it also fixes which way round,
so leave it out unless you need that line). `width_m`: how far to open before coming down (default
the object's width + 3 cm, up to 10 cm). Reasons:
- `empty`: the fingers closed on nothing. The object may have been pushed away: look before trying
  again. A thin stem pinched flat reads the same: if in doubt, `show("wrist")`, and if you see something
  hanging from the gripper, `place(box)` all the same.
- `height_unknown`: no depth under it. Give `z`; for a flat piece `z=TCP_FLOOR_Z`.
- `unreachable`, `too_wide`, `already_holding`.

`place(target, bottom_z=None, yaw_deg=None) -> Place` (`ok`, `reason`, `detail`, `release_xyz`,
`tilt_deg`, `yaw_deg`) — puts the held object down: moves above, straight down, opens a little,
straight back up. It does not ask whether the fingers hold anything. Reason: `unreachable` (it stays
held).
- `target`: `[x, y]` (its bottom ends 2 mm above the table, so it does not bounce), or a container's Sight (over its centre,
  its bottom 3 cm above `top_z`; the gripper stays above the rim).
- `bottom_z`: the height of the object's bottom when let go, instead of those. Over a container keep
  it at least 3 cm above the container's `top_z`, as the default does: a leaning finger can reach 1 to
  2 cm below a small object's bottom, and `top_z` varies by a few mm.
- `yaw_deg`: turn the object to this closing line. Without it the line stays if the wrist can keep
  it, else it turns as little as it must; the result's `yaw_deg` says which.

`drag(target, to, z=None, yaw_deg=None, width_m=None) -> Drag` (`ok`, `reason`, `detail`, `width_m`,
`final_xyz`, `tilt_deg`, `yaw_deg`) — slides a Sight along the table to `to` = `[x, y]` without
lifting it: opens, moves above, straight down, closes, straight line at that height, opens a little,
back up. `z`, `yaw_deg`, `width_m` and the reasons as in `pick`; on `empty` it does not slide. Nothing
senses contact: keep the line clear of other objects.

**Learned on the real arm.**
- A shape with no parallel sides (the heart) can be squeezed out sideways as the fingers close, and
  it flew 6 cm. Close across the sides that are most nearly parallel (`yaw_deg`).
- The fingers open to the object's width + 3 cm, up to 10 cm, and a wide opening can brush a
  neighbour: a peach moved about 2 cm when the starfruit beside it was picked. Between close
  neighbours choose a closing line that keeps the fingers away from them (`yaw_deg`) and open less
  (`width_m`, at least the object's width across that line + 1 cm).
- Beyond about 0.46 m a flat piece can only be grasped leaning. Then close across the lean,
  `yaw_deg` = heading from the base + 90, or one finger sits higher than the other and misses it.
- While holding, the closing line can be at most 100 degrees from that same direction (the wrist's
  limit). For a big turn or a long sideways drag, grasp the way round that leaves room for it.
- Passing `yaw_deg=s.grasp_yaw_deg` is not the same as leaving it out. At 0.45 m it made the wrist
  turn 162 degrees, the grasp lean 40 and the hover lean 70, only 3 cm above the grasp and beside a
  fruit. Left out, the same line was straight down with a 15 degree turn.
- Two objects can be one: a grasp of one cherry lifted the other on the same stem too (3 times out of
  4, `width_m` 2.6 to 2.7 cm), and a `pick` of the second at its old position closed on nothing. Before
  each next `pick`, `home()` and `locate` again. A long thin find, such as a cherry with its stem (`size`
  5.8 x 2.2 cm, not 2.4 cm), is closed on where most of its points are, not at `center_xyz`. After a
  `pick`, compare `width_m` with the object.
- If `locate` no longer finds an object that should not have moved, `pick` its earlier Sight: `pick`
  says `empty` if it has gone. That worked for a strawberry. Do not move the arm around to look: after
  `home()` it is out of the head camera's way.
- Two small round things side by side can be found as one, and closing across both squeezes them out.
  If `size` is much larger than one of them, open less (`width_m` about 3.5 cm for a cherry). Aim at
  one, not at the middle.
- Two words can find the same objects: `locate("cherry")` and `locate("strawberry")` returned the same
  red spots. Compare positions and look at the image, or you will go to one object twice.

### Motion, gripper, state

`move_to(xyz, yaw_deg=None, tilt_deg=None, path="auto") -> Move` (`reached`, `final_xyz`, `err_m`,
`yaw_deg`, `tilt_deg`, `reason`, `detail`) — moves the TCP; the fingertips reach 12.5 mm below it.
`None` keeps the current angle. `path`: `"line"` is the straight line or nothing (near objects),
`"joint"` is fast but curved (away from objects): between ends of one height it stays about at that
height, and between two heights it sweeps down or up through whatever stands in between, which once
turned a box over. `"auto"` is a line up to 12 cm and a joint move beyond. `reason` `ik_failed`, `below_floor` or `outside_workspace` means
nothing moved, so try another lean or height: it costs nothing. An empty gripper also tries half a
turn (the result's `yaw_deg` says which).

Highest TCP z straight ahead, calculated from the arm's geometry (other directions alike):

| distance from base | `tilt_deg` 0 | 15 | 30 | 45 and more |
|---|---|---|---|---|
| 0.20 to 0.30 m | 0.10 | 0.15 | 0.25 | 0.25 and up |
| 0.35 m | 0.08 | 0.15 | 0.25 | 0.25 and up |
| 0.40 m | 0.06 | 0.15 | 0.25 | 0.25 and up |
| 0.45 m | 0.02 | 0.15 | 0.25 | 0.25 and up |
| 0.50 m | out of reach | 0.10 | 0.20 | 0.25 and up |
| 0.55 m | out of reach | 0.06 | 0.20 | 0.25 and up |

Leaning 70 degrees it released a fruit at 0.71 m from the base, 0.23 m high, into a real box.

`set_gripper(width_m, speed_m_s=None) -> Grip` (`width_m`, `holding`) — moves the fingers to an opening,
0 = closed, `MAX_OPENING_M` = open (10 cm). `speed_m_s`: how fast the opening may change, default full
speed; closing on a light object by hand, use 0.02 (`pick` and `drag` already close that way).
`holding`: they stopped wider than asked, so something is between
them; it does not say what. `home()` — back to the start pose, out of the head camera's way; the
gripper keeps its opening. `state()` — `tcp_xyz`, `yaw_deg`, `tilt_deg`, `joints`, `gripper_width_m`,
`holding`. `time_left()` — seconds left in the scene (information only).
`TCP_FLOOR_Z` — the lowest TCP height with the gripper straight down; a flat piece is grasped there.
`READ_UP_DEG` / `READ_RIGHT_DEG` — headings in which judges read an arrangement: a symbol's top
should point to `READ_UP_DEG`; a row reads left to right along `READ_RIGHT_DEG`.

## Task playbooks

The competition has three task types. Each playbook says what scores, the blocks to run, the
decisions and pitfalls. The reference code ran in a fake world only. The words given to `locate`
worked on the organizers' photos, and "fruit" and "cardboard box" also on the real table. The code
is a starting point, not a fixed procedure: read
the task and the scene, then write your own blocks, changing the code where the task differs
(other objects, a different number of items, a different layout, an unexpected result).

### Playbook A: move target objects into their containers

**Scores.** Per target object: completely inside its container and stable 100; outside but
within 10 cm of the container's edge 50; farther 0; the task score is the average. Every
moved non-target object −10, every moved container −50.

**Blocks.**
1. See: `home()`, `locate` the objects and the container, `show` the numbered head image. From
   the image and the task's words decide which numbers go in. Check each target's `note` and how
   close its neighbours are.
2. Act: for each target, nearest first, `pick` then `place(container)`. Before each later target,
   `home()` and `locate` again: the last move may have changed things. If `place` cannot reach,
   `place` right next to the container (within 10 cm of its edge): half the score beats leaving
   the object where it was.
3. Check: `home()`, `show("head")`: is each target inside? Compare the container with where it
   was: if it moved, your arm moved it, and you report that. A dropped object can be located and
   picked again.

**Pitfalls.** Never lower the gripper into a container or push one (−50): with its `bottom_z` at 0.10 m
under a rim at about 0.13 m the gripper went in and pushed a real box 11 cm. `place(box)` releases
above the rim and never enters. Do not move anything else. Use the whole box: `locate("cardboard box")`
is steady (score 0.95 and up). Open flaps belong to it: they raise its `top_z`, so the object falls a
little further, and its centre was about 1 cm from the middle of the opening. Do not use
`locate("empty space inside the box")`: on one box its `top_z` jumped between 0.06, 0.13 and 0.17 m, and
with the arm in view its centre moved 3 cm. `too_wide` in a container's `note` means nothing.

Act only on what you are sure of. If you cannot find an object, `home()` and look again, and watch for
whether it has already gone into the box.

**Reference code.**

```python
# Block 1: see
home()
objects = locate("fruit")
box = locate("cardboard box")[0]
print("box", box)
for i, s in enumerate(objects, 1):
    print(i, s)
show("head", marks=objects + [box])
```

```python
# Block 2: act, after looking at the image
TARGETS = [1]                      # the numbers you read from the image
for k, i in enumerate(TARGETS):
    s = objects[i - 1]
    if k:                          # look again, from out of the way: the last move may have changed things
        home()
        s = next((n for n in locate(s.text) if np.linalg.norm(np.array(n.center_xyz[:2]) - s.center_xyz[:2]) < 0.03), None)
        if s is None:
            print("target", i, "is no longer where it was")
            continue
    r = pick(s)
    print("pick", i, r.ok, r.reason)
    if not r.ok:
        continue
    p = place(box)
    print("place", i, p.ok, p.reason, p.detail)
    if not p.ok:                   # half score: right next to the container
        print("beside", place([box.center_xyz[0], box.center_xyz[1] - (box.size[0] / 2 + 0.05)]))
```

```python
# Block 3: check
home()
now = locate("cardboard box")[0]
print("box moved by", round(float(np.linalg.norm(np.array(now.center_xyz) - box.center_xyz)), 3))
show("head")
```

### Playbook B: sort blocks and fruits into groups

**Scores.** Instruction "将桌上积木和水果分类摆放好": blocks together, fruits together, two
clear areas. Score = share of the initially out-of-place objects (usually 1–2) that end up
in their group; all blocks and fruits are checked at the end, so an object knocked out of
its group counts against you.

**Blocks.**
1. See: `home()`, `locate` with a word for each kind, `show` the numbered head image. Read which
   number is a block and which a fruit.
2. Act: for each kind find where most of its members already are; the others are the strays.
   Put each stray on a free spot next to its own group, on the side away from the other group.
3. Check: `home()`, `show("head")`: two clear groups? Move only what is still out of place.

**Pitfalls.** The word "fruit" also finds flat blocks, at a score around 0.5 (fruits score 0.75
and up), and "flat colored block" can find something beside the table around 0.4: the numbered
image decides, not the word. "toy block" finds nothing. Blocks here are flat (about 1 cm) and sit
close to the table: `pick` keeps the fingertips above it, so check `ok` after the grasp. A flat
block that will not lift can be moved with `drag` if the line to its spot is clear.
Leave grouped objects where they are:
every move risks a push. If the groups are not separated at the start, pick two areas (e.g. one
per side of the table) and move the minority of each area.

**Reference code.**

```python
# Block 1: see
home()
items = locate("fruit")
for s in locate("flat colored block"):        # two words can find the same object
    if all(np.linalg.norm(np.array(s.center_xyz[:2]) - o.center_xyz[:2]) > 0.03 for o in items):
        items.append(s)
for i, s in enumerate(items, 1):
    print(i, np.round(s.center_xyz[:2], 3).tolist(), np.round(s.size, 3).tolist(), s.note)
show("head", marks=items)
```

```python
# Block 2: act, after looking at the image
KINDS = {1: "fruit", 2: "block"}              # what you read for each number
groups = {"block": [], "fruit": []}
for i, s in enumerate(items, 1):
    groups[KINDS[i]].append(s)

def xy(s):
    return np.array(s.center_xyz[:2])

def core(members, radius=0.12):               # where most of a group already is
    best = []
    for m in members:
        near = [o for o in members if np.linalg.norm(xy(o) - xy(m)) <= radius]
        if len(near) > len(best):
            best = near
    return best

def free_spot(s, home_xy, away_xy, others):
    # nearest point to home_xy that keeps s clear of every object and on its group's side
    for r in np.arange(0.0, 0.25, 0.01):
        for a in range(0, 360, 20):
            p = home_xy + r * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            if all(np.linalg.norm(p - xy(o)) >= (max(s.size[:2]) + max(o.size[:2])) / 2 + 0.02 for o in others) and \
               np.linalg.norm(p - away_xy) > np.linalg.norm(p - home_xy) + 0.05:
                return p
    return None

cores = {k: core(v) for k, v in groups.items()}
centre = {k: np.median([xy(o) for o in c], axis=0) for k, c in cores.items() if c}
for kind, members in groups.items():
    other = "fruit" if kind == "block" else "block"
    for m in [m for m in members if m not in cores[kind]]:
        spot = free_spot(m, centre[kind], centre.get(other, centre[kind] + 1.0), [o for o in items if o is not m])
        r = pick(m, z=TCP_FLOOR_Z if m.grasp_xyz is None else None)   # no depth under a flat block
        p = place(spot) if (r.ok and spot is not None) else None
        print(kind, "stray at", np.round(xy(m), 3), "->", None if spot is None else np.round(spot, 3), r.ok, r.reason, p)
        if p is not None and p.ok:
            m.center_xyz = [float(spot[0]), float(spot[1]), m.center_xyz[2]]
```

```python
# Block 3: check
home()
show("head")
```

### Playbook C: arrange the number blocks into an equation

**Scores.** Three number pieces and one piece each of + − × ÷ =. Use all three numbers, one
operator and the equals sign in one row: "number op number = number" or "number = number op
number". 50 for a complete, clearly readable row, another 50 if it is mathematically correct.

**Time decides this task.** Moving one piece takes about 21 s, five take about 105 s, so start moving
within 60 s of the start. Do not work the geometry out in your head: the reference code finds the
equation, lays out the row and turns the pieces. If it prints `equation: None`, a reading is wrong or
missing. A complete row still earns 50, so do not stop: set `row` yourself to three numbers, one operator
and `=`, for example `row = ["2", "+", "3", "=", "6"]`, naming through `READ` a piece that has no `symbol`.

**The pieces** are cut in the shape of the number or symbol: nothing is printed on them, so read
the shape. Numbers go from 1 to 20. `locate` compares each piece with stored outlines of the real
pieces and fills in `symbol` (`+`, `×`, `÷`, `-`, `=` or a number as text), `top_deg` (world heading
of the symbol's top) and `top_every_deg` (it looks the same turned by this much). Trust what it reads.
- `symbol` None: it could not tell. Block 1 reads the piece again from close up.
- `top_deg` None: it knows the piece but not which way up. Block 1 reads this one from close up too.

`show(image=PIECE_CHART)` shows a clear picture of every piece, upright and named, sorted by colour.
`show(image=PIECE_IMAGES["6"])` shows one piece larger, by what it shows (`"1"` to `"20"`, `"+"`, `"-"`,
`"×"`, `"÷"`, `"="`). Use them only to choose between two you are unsure of: one long look at enlarged
pictures of the pieces cost 77 s in a real run. A piece can lie on its back, and then it shows the mirror image. Colours narrow
it down, and a colour is shared by several pieces:

| Colour | Pieces |
|---|---|
| orange | × 4 11 |
| yellow | 1 5 15 |
| green | + 2 18 |
| teal | 10 12 20 |
| blue | − 3 7 19 |
| purple | = 9 17 |
| pink to dark red | ÷ 6 8 13 14 16 |

`+` and `×` have the same shape: only how the piece lies makes it one or the other. `÷` is a cross with
two arms shorter.

**Where the row goes: you choose.** Pick a convenient free place from the head image and the pieces'
positions, and lay the five pieces out one after another, about 6 cm between centres. The row runs along
`READ_RIGHT_DEG`, tops to `READ_UP_DEG`. The arm leans by itself as far as a place needs, and that is fine.
Calculated on the real arm, a piece can be laid down with any turn for x from 0.18 to 0.50 and y up to
0.48; for x up to 0.46 and y from 0.16 to 0.40 it leans 25 degrees at most. Keep y above 0.08, or the arm
at home hides the row from the head camera. A slot needs 5.5 cm to any piece that stays
where it is. With no convenient free place, let the row lie over pieces the equation does not use and
give each of them a free place in `AWAY`: the code moves them first, about 21 s each.

**Blocks.**
1. See and read, all in one block: `home()`, `locate`, print, `show` the numbered head image. Then, for
   each piece that did not read, move over it and `locate` with the wrist camera, the same matching with
   more to see (about 6 s a piece). It prints what is still unread.
2. Only for a piece still unread: ask `look` from close up, with the pieces of its colour as the choices
   (Block 1b).
3. Act: fill in `CENTRE`, `READ` for what `look` told you, `AWAY` if the row lies over unused pieces. The
   code finds a true equation, checks the slots and moves the pieces without going home in between.
4. Check: `home()`, `show("head")`, read the row as the judges do.

**Pitfalls.** "−" turned 90 degrees looks like "1". A complete row of the right pieces already earns
half, so a piece that cannot be turned is still put in its slot. Grasp the way the code chooses: it
leaves the wrist room for the turn. A turn the wrist cannot make at once goes in two grasps, about
21 s more.

**Reference code.** Only the functions of the API exist from the start. `way_round`, `grasp`, `put` and
the others below exist once a block has defined them, so copy their definitions into your block.

```python
# Block 1: see and read; touches nothing
home()
pieces = locate("colored number or math symbol piece")
for i, s in enumerate(pieces, 1):
    print(i, s.symbol, s.top_deg, s.color, np.round(s.center_xyz[:2], 3).tolist(), np.round(s.size[:2], 3).tolist(), s.note)
show("head", marks=pieces)
f = capture("head")

def up_heading(frame):                        # world heading of "up" in this image
    d = frame.world[:-40, :, :2] - frame.world[40:, :, :2]
    dx, dy = np.nanmedian(d, axis=(0, 1))
    return math.degrees(math.atan2(dy, dx))

up_deg = {i: up_heading(f) for i in range(1, len(pieces) + 1)}   # of the image each piece is read in

def over(s):                                  # leaning 25 degrees the wrist camera looks straight down at the piece, wherever it lies
    h = math.atan2(s.center_xyz[1], s.center_xyz[0])
    return move_to([s.center_xyz[0] - 0.04 * math.cos(h), s.center_xyz[1] - 0.04 * math.sin(h), 0.08],
                   yaw_deg=math.degrees(h) + 90, tilt_deg=25, path="joint")

def unread():
    return [i for i, s in enumerate(pieces, 1) if s.symbol is None or s.top_deg is None]

for i in unread():                            # the same matching as from the head camera, with more to see
    s = pieces[i - 1]
    r = over(s)
    if not r.reached:
        print(i, r.reason)
        continue
    near = [w for w in locate("colored number or math symbol piece", camera="wrist")
            if np.linalg.norm(np.array(w.center_xyz[:2]) - s.center_xyz[:2]) < 0.03]
    print(i, "from close up:", near[0].symbol if near else None, near[0].top_deg if near else None)
    if near and near[0].symbol and near[0].top_deg is not None:
        s.symbol, s.top_deg, s.top_every_deg = near[0].symbol, near[0].top_deg, near[0].top_every_deg
print("still unread:", unread())
```

```python
# Block 1b, only for a piece still unread after Block 1: ask look from close up; touches nothing
LOOK = {}                                     # piece -> the pieces of its colour, from the table above: {1: ["=", "9", "17"]}
for i, same_colour in LOOK.items():
    r = over(pieces[i - 1])
    if not r.reached:
        print(i, r.reason)
        continue
    up_deg[i] = up_heading(capture("wrist"))  # READ below takes its top in this image
    print(i, "look:", look(f"A flat plastic piece lies in the middle of this image, cut in the shape of a number or a sign, "
                           f"turned any way. It is one of: {', '.join(same_colour)}. Which one, and which side of the image "
                           f"does its top point to: up, down, left or right?", camera="wrist"))
```

```python
# Block 2: act, after reading
READ = {}                                     # only what look told you: {1: ("7", "up")}: what it shows, where its top points in the image
CENTRE = [0.28, 0.20]                         # middle of the row: the free place you chose
STEP = 0.06                                   # between centres
SPIN = {"up": 0, "left": 90, "down": 180, "right": -90}
for i, (symbol, top) in READ.items():         # also corrects a piece you saw labelled wrongly
    s = pieces[i - 1]
    coarse = up_deg[i] + SPIN[top]
    s.symbol = symbol
    s.top_deg = min((s.long_axis_deg + k * 90 for k in range(4)), key=lambda a: abs((a - coarse + 180) % 360 - 180))
    s.top_every_deg = 360.0

import itertools
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "×": lambda a, b: a * b,
       "÷": lambda a, b: a / b if b else None}
by_symbol = {s.symbol: s for s in pieces if s.symbol}
row = None
for a, b, c in itertools.permutations([k for k in by_symbol if k.isdigit()], 3):
    for op, fn in OPS.items():
        v = fn(int(a), int(b))
        if op in by_symbol and "=" in by_symbol and v is not None and abs(v - int(c)) < 1e-9:
            row = [a, op, b, "=", c]
            break
    if row:
        break
print("equation:", " ".join(row) if row else None)

right = np.array([math.cos(math.radians(READ_RIGHT_DEG)), math.sin(math.radians(READ_RIGHT_DEG))])
slots = {symbol: np.array(CENTRE) + (k - 2) * STEP * right for k, symbol in enumerate(row or [])}

def distance_to_slots(s):
    return min(np.linalg.norm(p - s.center_xyz[:2]) for p in slots.values())

in_the_way = [s for s in pieces if s.symbol not in slots and distance_to_slots(s) < 0.055] if slots else []
print("slots", {k: np.round(v, 3).tolist() for k, v in slots.items()},
      "in the way:", [(s.symbol, np.round(s.center_xyz[:2], 3).tolist()) for s in in_the_way])

def way_round(s, turn, slot):                 # the closing line to grasp with: the wrist stops 100 degrees from the tangent
    def off(yaw, xy):
        return abs((yaw - math.degrees(math.atan2(xy[1], xy[0])) - 90 + 180) % 360 - 180)

    lines = [s.grasp_yaw_deg, s.grasp_yaw_deg + 180]
    if s.size[0] < 1.4 * s.size[1]:           # about square: it can also be held across the other way
        lines += [s.grasp_yaw_deg + 90, s.grasp_yaw_deg - 90]
    return min(lines, key=lambda g: max(off(g, s.center_xyz[:2]), off(g + turn, slot)))

def grasp(s, turn, slot):
    return pick(s, z=TCP_FLOOR_Z if s.grasp_xyz is None else None, yaw_deg=way_round(s, turn, slot))

def put(s, slot, turn):
    r = grasp(s, turn, slot)
    if not r.ok:
        return r
    p = place(slot, yaw_deg=r.yaw_deg + turn)
    if p.ok or not turn:
        return p
    p = place(slot, yaw_deg=r.yaw_deg + turn / 2)   # too far for one go: half now, the rest after grasping again
    if not p.ok:
        return place(slot)                    # still in its slot, unturned
    s.center_xyz[:2] = [float(v) for v in slot]
    s.points = None
    s.grasp_yaw_deg += turn / 2
    r = grasp(s, turn / 2, slot)
    return place(slot, yaw_deg=r.yaw_deg + turn / 2) if r.ok else r

AWAY = []                                     # a free place for each piece in the way, in that order: [[x, y]]
if len(in_the_way) > len(AWAY):
    print("nothing moved: give each piece in the way a free place in AWAY, or choose another CENTRE")
else:
    for s, spot in zip(in_the_way, AWAY):
        print("out of the way:", s.symbol, "->", spot, put(s, spot, 0.0))
    for symbol in sorted(slots, key=lambda k: distance_to_slots(by_symbol[k])):   # first those lying on a slot
        s, slot = by_symbol[symbol], slots[symbol]
        every = s.top_every_deg or 360.0
        turn = 0.0 if s.top_deg is None else (READ_UP_DEG - s.top_deg + every / 2) % every - every / 2   # the smallest
        print(symbol, "->", np.round(slot, 3).tolist(), "turn", round(turn), put(s, slot, turn))
```

```python
# Block 3: check
home()
show("head")
```

### Practical notes

- The arm reaches about 0.7 m from its base leaning far, a straight-down gripper about 0.46 m and
  only low (the table above).
- Print one line per step so a failure is easy to see in the output.
- `locate(...)[0]` raises `IndexError` when nothing was found. That ends the block, which is fine
  in a block that only looks; use `if not hits:` where the block should go on.
- `Sight.grasp_xyz` can be `None`, and so can `Move.err_m` after `home()`: check before rounding or
  indexing them.
<!-- END API -->

## Rules

- There is no API for raw joint commands; use the functions above.
- Do not `place` before a `pick`, unless you can see the object hanging from the gripper.
- On `RobotFault` from the arm or the gripper, stop issuing motions and finalize with the fault in the
  summary. A camera that stopped is different: act on the Sights you already have (see the cameras above).

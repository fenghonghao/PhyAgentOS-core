"""run_scene.sh gives each scene a clean PAOS workspace: a task left open by a stopped scene (the
referee's, or ours) must not block the next one, and what one scene noted must not reach the next."""

import os
import stat
import subprocess
from pathlib import Path

RUN_SCENE = Path(__file__).resolve().parents[3] / "run_scene.sh"


def scene(tmp_path, name):
    stub = tmp_path / "paos"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$HOME/args.txt"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "HOME": str(tmp_path), "PAOS_BIN": str(stub),
           "PAOS_SCENE_DEADLINE_FILE": str(tmp_path / "deadline")}
    subprocess.run(["bash", str(RUN_SCENE), name, "把樱桃放进箱子里"], env=env, check=True, capture_output=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    return Path(args[args.index("--workspace") + 1])


def base_workspace(tmp_path):
    base = tmp_path / ".PhyAgentOS" / "workspace"
    (base / "memory").mkdir(parents=True)
    (base / "memory" / "MEMORY.md").write_text("notes of an earlier scene")
    (base / ".paos" / "agent_tasks").mkdir(parents=True)
    (base / ".paos" / "agent_tasks" / "tasks.sqlite3").write_text("a task left open")
    (base / "AGENTS.md").write_text("how to work")


def test_each_scene_runs_in_its_own_workspace(tmp_path):
    base_workspace(tmp_path)
    first, second = scene(tmp_path, "a"), scene(tmp_path, "b")
    assert first != second and first.is_dir() and second.is_dir()


def test_a_scene_keeps_the_prompt_files_but_not_memory_or_tasks(tmp_path):
    base_workspace(tmp_path)
    ws = scene(tmp_path, "a")
    assert (ws / "AGENTS.md").read_text() == "how to work"
    assert not (ws / "memory" / "MEMORY.md").exists()
    assert not (ws / ".paos").exists()

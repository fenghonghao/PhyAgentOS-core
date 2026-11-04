"""Remaining scene time is appended to every model request when PAOS_SCENE_DEADLINE is set."""

import time

import pytest

from PhyAgentOS.agent.context import ContextBuilder
from PhyAgentOS.utils.scene_deadline import SCENE_DEADLINE_ENV, scene_time_left, time_left_note


@pytest.fixture
def builder(tmp_path):
    return ContextBuilder(tmp_path)


def test_unset_means_no_note(monkeypatch, builder):
    monkeypatch.delenv(SCENE_DEADLINE_ENV, raising=False)
    assert scene_time_left() is None and time_left_note() is None
    assert "time_left_s" not in builder._build_message_context(None, None)
    msgs = builder.add_tool_result([], "c1", "robot.run", "done")
    assert msgs[-1]["content"] == "done"


def test_invalid_value_is_ignored(monkeypatch):
    monkeypatch.setenv(SCENE_DEADLINE_ENV, "soon")
    assert scene_time_left() is None


def test_seconds_left_and_negative_after_deadline(monkeypatch):
    monkeypatch.setenv(SCENE_DEADLINE_ENV, "1000.0")
    assert scene_time_left(now=880.0) == 120.0
    assert time_left_note(now=1003.25) == "[time_left_s=-3.2]"


def test_first_request_and_tool_results_carry_time_left(monkeypatch, builder):
    monkeypatch.setenv(SCENE_DEADLINE_ENV, str(time.time() + 180))
    ctx = builder._build_message_context(None, None)
    assert "[time_left_s=1" in ctx  # ~180 s
    msgs = builder.add_tool_result([], "c1", "robot.run", '{"ok": true}')
    content = msgs[-1]["content"]
    assert content.startswith('{"ok": true}\n[time_left_s=') and content.endswith("]")


def test_earlier_messages_are_not_rewritten(monkeypatch, builder):
    """Only the newest message gets the note, so the prompt prefix stays cacheable."""
    monkeypatch.setenv(SCENE_DEADLINE_ENV, str(time.time() + 180))
    msgs = builder.add_tool_result([], "c1", "robot.run", "first")
    snapshot = [dict(m) for m in msgs]
    builder.add_tool_result(msgs, "c2", "robot.run", "second")
    assert msgs[0] == snapshot[0]

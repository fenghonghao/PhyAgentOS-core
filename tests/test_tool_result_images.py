"""Images a Forge Action returns in ``outputs.images`` reach the next model request.

A tool message cannot carry images and must stay attached to its assistant ``tool_calls``, so
after the LAST tool result of an assistant turn the loop appends exactly one ``user`` message
holding every image of that turn, in call order.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import httpx
import pytest

from PhyAgentOS.agent.context import ContextBuilder
from PhyAgentOS.agent.loop import AgentLoop
from PhyAgentOS.agent.tools import forge_tool_api
from PhyAgentOS.agent.tools.base import Tool
from PhyAgentOS.agent.tools.forge_tool_api import build_forge_tool_api_tools
from PhyAgentOS.config.schema import ForgeConfig
from PhyAgentOS.forge.task import AgentTaskCoordinator
from PhyAgentOS.forge.tool_client import ForgeToolClient
from PhyAgentOS.providers.base import LLMResponse, ToolCallRequest
from PhyAgentOS.verification.contracts import TaskVerificationContract, VerificationEvidencePolicy


class _RecordingProvider:
    """Hands out scripted responses and keeps a snapshot of the messages of every call."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.requests: list[list[dict]] = []

    def get_default_model(self) -> str:
        return "stub-model"

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        assert self._responses, "unexpected extra chat_with_retry call"
        return self._responses.pop(0)


class _Bus:
    async def publish_outbound(self, message) -> None:
        pass


class _Static(Tool):
    """A tool that returns a fixed string, whatever it is."""

    def __init__(self, name: str, result: str):
        self._name, self._result = name, result

    name = property(lambda self: self._name)
    description = property(lambda self: "static")
    parameters = property(lambda self: {"type": "object", "properties": {}})

    async def execute(self, **kwargs) -> str:
        return self._result


def _calls(*calls: tuple[str, str, dict]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id=i, name=n, arguments=a) for i, n, a in calls],
        finish_reason="tool_calls",
    )


def _jpeg(directory: Path, name: str) -> tuple[str, str]:
    """Write a tiny JPEG-magic file; return (path, the data URI the model should get)."""
    raw = b"\xff\xd8\xff\xe0" + name.encode()
    path = directory / f"{name}.jpg"
    path.write_bytes(raw)
    return str(path), "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def _loop(tmp_path: Path, responses: list[LLMResponse]) -> tuple[AgentLoop, _RecordingProvider]:
    provider = _RecordingProvider(responses)
    loop = AgentLoop(bus=_Bus(), provider=provider, workspace=tmp_path, context_window_tokens=10**9)
    return loop, provider


def _image_urls(message: dict) -> list[str]:
    return [p["image_url"]["url"] for p in message["content"] if p["type"] == "image_url"]


def _static_outputs(images) -> str:
    return json.dumps({"ok": True, "data": {"status": "available",
                       "result": {"status": "succeeded", "outputs": {"ok": True, "images": images}}}})


# --- the real Forge tools, so the nesting under test is the nesting they really return ----------

def _ok(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"ok": True, "data": data})


def _done(images: list[str]) -> httpx.Response:
    outputs = {"ok": True, "output": "", "error": "", "elapsed_s": 1.0, "primitive_calls": 2,
               "images": images}
    return _ok({"status": "available", "result": {"status": "succeeded", "outputs": outputs}})


def _forge(tmp_path: Path, monkeypatch, results: list[httpx.Response]):
    monkeypatch.setattr(forge_tool_api, "WAIT_POLL_S", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tools/robot.run:invoke":
            return _ok({"invocation_id": "inv_1", "attempt_id": "att_1"}, status=202)
        if request.url.path == "/tools/robot.run":
            return _ok({"tool_id": "robot.run", "endpoint_id": "robot.skill_server",
                        "operation": "run", "semantics": "action"})
        if request.url.path == "/invocations/inv_1/result":
            return results.pop(0)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    client = ForgeToolClient("http://gateway.test", transport=httpx.MockTransport(handler))
    client.base_url = ""
    coordinator = AgentTaskCoordinator(workspace=tmp_path, config=ForgeConfig(), client=client)
    task = coordinator.create_task(
        task_description="show images",
        verification=TaskVerificationContract(evidence_policy=VerificationEvidencePolicy(required_kinds=[])),
    )
    return client, coordinator, task, build_forge_tool_api_tools(client, coordinator=coordinator)


async def test_images_of_all_tool_calls_follow_the_last_tool_message_in_one_user_message(
    tmp_path: Path, monkeypatch,
) -> None:
    a_path, a_uri = _jpeg(tmp_path, "a")
    b_path, b_uri = _jpeg(tmp_path, "b")
    c_path, c_uri = _jpeg(tmp_path, "c")
    pending = _ok({"status": "pending"}, status=202)
    client, coordinator, task, tools = _forge(
        tmp_path, monkeypatch, [pending, _done([a_path]), _done([b_path, c_path])])
    loop, provider = _loop(tmp_path, [
        _calls(
            ("c1", "forge_tool_start_action", {
                "task_id": task.task_id, "tool_id": "robot.run",
                "arguments": {"code": "show(a)"}, "wait": True}),
            ("c2", "forge_tool_action_result", {"task_id": task.task_id, "invocation_id": "inv_1"}),
        ),
        LLMResponse(content="done"),
    ])
    for tool in tools:
        loop.tools.register(tool)
    try:
        _, _, messages = await loop._run_agent_loop(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])
    finally:
        await client.close()

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool", "user", "assistant"]
    images = messages[5]
    assert _image_urls(images) == [a_uri, b_uri, c_uri]  # call order, one message, all three
    texts = [p["text"] for p in images["content"] if p["type"] == "text"]
    assert len(texts) == 1 and "show()" in texts[0] and "call order" in texts[0]


async def test_next_provider_call_receives_the_images(tmp_path: Path) -> None:
    a_path, a_uri = _jpeg(tmp_path, "a")
    loop, provider = _loop(tmp_path, [
        _calls(("c1", "robot_run", {})),
        LLMResponse(content="saw it"),
    ])
    loop.tools.register(_Static("robot_run", _static_outputs([a_path])))

    await loop._run_agent_loop([{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])

    first, second = provider.requests
    assert [m["role"] for m in first] == ["system", "user"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "user"]
    assert _image_urls(second[-1]) == [a_uri]


@pytest.mark.parametrize("result", [
    "plain text, not json",
    json.dumps(["images"]),
    json.dumps({"ok": False, "error": {"type": "runtime", "message": "boom"}}),
    json.dumps({"ok": True, "data": {"result": {"outputs": {"ok": True, "output": "x"}}}}),  # no images
    json.dumps({"ok": True, "data": {"result": {"outputs": {"images": "not-a-list"}}}}),
    json.dumps({"ok": True, "data": {"result": {"outputs": {"images": [42, None]}}}}),
    _static_outputs([]),
], ids=["not-json", "json-list", "error", "no-images", "images-not-list", "non-string-paths", "empty"])
async def test_no_user_message_without_images(tmp_path: Path, result: str) -> None:
    loop, provider = _loop(tmp_path, [_calls(("c1", "robot_run", {})), LLMResponse(content="ok")])
    loop.tools.register(_Static("robot_run", result))

    _, _, messages = await loop._run_agent_loop(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]


async def test_no_user_message_when_no_file_loads_as_an_image(tmp_path: Path) -> None:
    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")
    loop, provider = _loop(tmp_path, [_calls(("c1", "robot_run", {})), LLMResponse(content="ok")])
    loop.tools.register(_Static("robot_run", _static_outputs(
        [str(tmp_path / "missing.jpg"), str(not_an_image), str(tmp_path)])))

    _, _, messages = await loop._run_agent_loop(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]


async def test_loadable_images_are_kept_when_a_sibling_path_is_missing(tmp_path: Path) -> None:
    a_path, a_uri = _jpeg(tmp_path, "a")
    loop, provider = _loop(tmp_path, [_calls(("c1", "robot_run", {})), LLMResponse(content="ok")])
    loop.tools.register(_Static("robot_run", _static_outputs([str(tmp_path / "gone.jpg"), a_path])))

    _, _, messages = await loop._run_agent_loop(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])

    assert [m["role"] for m in messages][-3:] == ["tool", "user", "assistant"]
    assert _image_urls(messages[4]) == [a_uri]


async def test_each_assistant_turn_gets_its_own_image_message(tmp_path: Path) -> None:
    a_path, a_uri = _jpeg(tmp_path, "a")
    b_path, b_uri = _jpeg(tmp_path, "b")
    loop, provider = _loop(tmp_path, [
        _calls(("c1", "show_a", {})),
        _calls(("c2", "show_b", {})),
        LLMResponse(content="ok"),
    ])
    loop.tools.register(_Static("show_a", _static_outputs([a_path])))
    loop.tools.register(_Static("show_b", _static_outputs([b_path])))

    _, _, messages = await loop._run_agent_loop(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}])

    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "tool", "user", "assistant", "tool", "user", "assistant"]
    assert _image_urls(messages[4]) == [a_uri] and _image_urls(messages[7]) == [b_uri]


async def test_saved_session_keeps_no_base64(tmp_path: Path) -> None:
    a_path, _ = _jpeg(tmp_path, "a")
    loop, provider = _loop(tmp_path, [_calls(("c1", "robot_run", {})), LLMResponse(content="saw it")])
    loop.tools.register(_Static("robot_run", _static_outputs([a_path])))

    await loop.process_direct("go", session_key="cli:test")

    assert "base64," in json.dumps(provider.requests[1])  # the model did get it
    session = loop.sessions.get_or_create("cli:test")
    saved = [m for m in session.messages if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(saved) == 1
    assert {"type": "text", "text": "[image]"} in saved[0]["content"]
    on_disk = loop.sessions._get_session_path("cli:test").read_text(encoding="utf-8")
    assert "base64," not in on_disk and "data:image" not in on_disk


def test_tool_result_images_reads_both_real_nestings() -> None:
    read = ContextBuilder.tool_result_images
    result = {"ok": True, "data": {"status": "available",
              "result": {"status": "succeeded", "outputs": {"images": ["/x/1.jpg", "/x/2.jpg"]}}}}
    assert read(json.dumps(result)) == ["/x/1.jpg", "/x/2.jpg"]  # forge_tool_action_result
    assert read(json.dumps({"start": {"ok": True, "data": {}}, "result": result})) == [
        "/x/1.jpg", "/x/2.jpg"]  # forge_tool_start_action with wait=true

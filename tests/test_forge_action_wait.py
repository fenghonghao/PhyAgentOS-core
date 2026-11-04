"""wait=true on Action start/result reads until a terminal state, persisting every read."""

from __future__ import annotations

import json

import httpx

from PhyAgentOS.agent.tools import forge_tool_api
from PhyAgentOS.agent.tools.forge_tool_api import build_forge_tool_api_tools
from PhyAgentOS.config.schema import ForgeConfig
from PhyAgentOS.forge.task import AgentTaskCoordinator
from PhyAgentOS.forge.tool_client import ForgeToolClient
from PhyAgentOS.verification.contracts import TaskVerificationContract, VerificationEvidencePolicy


def _ok(data: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"ok": True, "data": data})


def _setup(tmp_path, results):
    reads = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tools/robot.run:invoke":
            return _ok({"invocation_id": "inv_1", "attempt_id": "att_1"}, status=202)
        if request.url.path == "/tools/robot.run":
            return _ok({"tool_id": "robot.run", "endpoint_id": "robot.skill_server",
                        "operation": "run", "semantics": "action"})
        if request.url.path == "/invocations/inv_1/result":
            reads["n"] += 1
            return results.pop(0)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    client = ForgeToolClient("http://gateway.test", transport=httpx.MockTransport(handler))
    client.base_url = ""
    coordinator = AgentTaskCoordinator(workspace=tmp_path, config=ForgeConfig(), client=client)
    task = coordinator.create_task(
        task_description="wait contract",
        verification=TaskVerificationContract(evidence_policy=VerificationEvidencePolicy(required_kinds=[])),
    )
    tools = {t.name: t for t in build_forge_tool_api_tools(client, coordinator=coordinator)}
    return client, coordinator, task, tools, reads


def _done(status="succeeded"):
    return _ok({"status": "available", "result": {"status": status, "outputs": {"output": "done\n"}}})


async def test_start_action_wait_returns_terminal_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forge_tool_api, "WAIT_POLL_S", 0)
    pending = _ok({"status": "pending"}, status=202)
    client, coordinator, task, tools, reads = _setup(tmp_path, [pending, pending, _done()])
    try:
        assert "wait" in tools["forge_tool_start_action"].parameters["properties"]
        out = json.loads(await tools["forge_tool_start_action"].execute(
            task.task_id, "robot.run", {"code": "print('x')"}, wait=True))
    finally:
        await client.close()
    assert reads["n"] == 3
    assert out["start"]["data"]["invocation_id"] == "inv_1"
    assert out["result"]["data"]["result"]["status"] == "succeeded"
    record = coordinator.store.get(task.task_id).execution_records[-1]
    assert record.status == "succeeded"


async def test_start_action_without_wait_is_unchanged(tmp_path) -> None:
    client, coordinator, task, tools, reads = _setup(tmp_path, [])
    try:
        out = json.loads(await tools["forge_tool_start_action"].execute(
            task.task_id, "robot.run", {"code": "print('x')"}))
    finally:
        await client.close()
    assert reads["n"] == 0 and out["data"]["invocation_id"] == "inv_1"


async def test_action_result_wait_stops_on_failed_and_unknown(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(forge_tool_api, "WAIT_POLL_S", 0)
    for terminal in (_done("failed"), _ok({"phase": "unknown"})):
        pending = _ok({"status": "pending"}, status=202)
        client, coordinator, task, tools, reads = _setup(tmp_path / terminal.json()["data"].get("phase", "f"),
                                                         [pending, terminal])
        try:
            await tools["forge_tool_start_action"].execute(task.task_id, "robot.run", {"code": "1"})
            out = json.loads(await tools["forge_tool_action_result"].execute(task.task_id, "inv_1", wait=True))
        finally:
            await client.close()
        assert reads["n"] == 2
        assert out["ok"] is True

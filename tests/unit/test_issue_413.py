"""#413 — a reasoning/agent turn that narrates an action but never calls a tool.

The investigation showed JARVIS executes whatever tool calls the model emits, so
"narrated but didn't act" means the model returned prose with no tool call. That
was invisible in the log (just a gap between "LLM needed" and the reply). The fix
is observability: when an agent turn ends with no tool call at all, log which
model served it and how many tools it had. These cover the message helper.
"""
import pytest


@pytest.fixture
def agent(load):
    return load("agent")


def test_note_names_model_and_tool_count(agent):
    note = agent._no_action_note("ollama/wolfe-command", 47)
    assert "no tool call" in note
    assert "ollama/wolfe-command" in note
    assert "47" in note


def test_note_reports_reasoning_fallback_label(agent):
    note = agent._no_action_note("anthropic/claude-opus-5-5 (reasoning-tier fallback)", 12)
    assert "reasoning-tier fallback" in note
    assert "anthropic/claude-opus-5-5" in note
    assert "12 tool(s) available" in note


def test_note_handles_zero_tools(agent):
    note = agent._no_action_note("groq/llama", 0)
    assert "0 tool(s) available" in note

"""``tools.tool_search.eager``: names pinned eager override every deferral rule, including
the blanket one that defers all MCP tools."""

from unittest import mock

import tools.tool_search as ts
import tools.tool_search_validation as validation


def _defs(*names):
    return [{"type": "function",
             "function": {"name": n, "description": "x", "parameters": {}}} for n in names]


QMD = "mcp__qmd__query"
DATAGOV = "mcp__datagov__run_pandas"


def _assemble(raw):
    """Assemble with both MCP tools registered under an ``mcp-`` toolset (the blanket rule)."""
    with mock.patch.object(ts, "_registry_toolset", lambda n: "mcp-" + n.split("__")[1]):
        return ts.assemble_tool_defs(_defs(QMD, DATAGOV), context_length=200_000,
                                     config=ts.ToolSearchConfig.from_raw(raw))


def _names(result):
    return [d["function"]["name"] for d in result.tool_defs]


def test_default_defers_every_mcp_tool():
    """Baseline: without ``eager`` both servers go behind the bridge."""
    names = _names(_assemble(None))
    assert QMD not in names and DATAGOV not in names
    assert set(ts.BRIDGE_TOOL_NAMES).issubset(names)


def test_eager_prefix_pins_one_server_and_defers_the_rest():
    names = _names(_assemble({"eager": ["mcp__qmd__*"]}))
    assert QMD in names, "the pinned server must stay ambient"
    assert DATAGOV not in names, "an unpinned server must still defer"
    assert set(ts.BRIDGE_TOOL_NAMES).issubset(names), "the bridge still carries the deferred ones"


def test_eager_exact_name_also_pins():
    assert QMD in _names(_assemble({"eager": [QMD]}))


def test_eager_beats_an_explicit_defer_entry():
    """A name in both lists is eager: the pin is the more specific instruction."""
    assert QMD in _names(_assemble({"eager": [QMD], "defer": [QMD]}))


def test_eager_tool_is_rejected_at_the_bridge():
    """An eager tool is callable directly, so ``tool_call`` must send the model to the right
    door rather than accepting a name the bridge no longer owns."""
    with mock.patch.object(ts, "_registry_toolset", lambda n: "mcp-" + n.split("__")[1]), \
         mock.patch.object(ts, "load_config_readonly",
                           lambda: ts.ToolSearchConfig.from_raw({"eager": ["mcp__qmd__*"]})):
        assert ts.is_deferrable_tool_name(QMD, frozenset(), frozenset(["mcp__qmd__*"])) is False
        assert ts.scoped_deferrable_names(_defs(QMD, DATAGOV)) == frozenset([DATAGOV])


def test_eager_does_not_expose_an_unavailable_mcp_writer():
    """Pinning a name must never grant access to a tool missing from this session."""
    writer = "mcp__intelligence__intelligence_feedback_publish"
    cfg = ts.ToolSearchConfig.from_raw({"eager": [writer]})
    with mock.patch.object(ts, "_registry_toolset", lambda n: "mcp-" + n.split("__")[1]), \
         mock.patch.object(ts, "load_config_readonly", lambda: cfg):
        names = _names(ts.assemble_tool_defs(_defs(QMD), context_length=200_000, config=cfg))
        assert writer not in names
        assert QMD not in names, "the unpinned reader is still deferred"
        target, arguments, error = ts.resolve_underlying_call(
            {"calls": [{"name": writer, "arguments": {}}]},
            session_tool_names=[QMD, "tool_call"])
        assert target is None and arguments == {} and error
        assert "not a known tool name" in error.lower()


def test_eager_large_payload_cannot_be_routed_through_tool_call():
    """The exact long-JSON MCP name stays direct, avoiding the fragile calls wrapper."""
    writer = "mcp__commercial_scouting__commercial_validate_report"
    cfg = ts.ToolSearchConfig.from_raw({"eager": [writer]})
    with mock.patch.object(ts, "_registry_toolset", lambda n: "mcp-" + n.split("__")[1]), \
         mock.patch.object(validation, "_registry_entry", lambda n: object() if n == writer else None), \
         mock.patch.object(ts, "load_config_readonly", lambda: cfg):
        names = _names(ts.assemble_tool_defs(_defs(writer, DATAGOV),
                                             context_length=200_000, config=cfg))
        assert writer in names and DATAGOV not in names
        target, arguments, error = ts.resolve_underlying_call(
            {"calls": [{"name": writer, "arguments": {"report_json": "x" * 9000}}]},
            session_tool_names=[writer, DATAGOV, "tool_call"])
        assert target is None and arguments == {} and error
        assert "direct" in error.lower()


def test_scalar_eager_warns_and_is_ignored():
    """Fail loud, same shape as a scalar ``defer``: never silently reshape the tool surface."""
    with mock.patch.object(ts.logger, "warning") as warn:
        config = ts.ToolSearchConfig.from_raw({"eager": "mcp__qmd__*"})
    assert config.eager_tools == frozenset()
    assert warn.called


def test_absent_eager_is_empty_not_none():
    assert ts.ToolSearchConfig.from_raw({}).eager_tools == frozenset()


def test_bare_star_pins_everything():
    names = _names(_assemble({"eager": ["*"]}))
    assert QMD in names and DATAGOV in names


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

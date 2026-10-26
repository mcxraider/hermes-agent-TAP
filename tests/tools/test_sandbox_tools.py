"""``code_execution.sandbox_tools``: the operator's ceiling on what model-written sandbox code
may reach over the RPC bridge.

Nothing passes ``enabled_tools`` in production, so ``_sandbox_tools_for`` always takes its
fall-back branch — without this key, disabling the ``terminal``/``file`` toolsets does not stop
sandbox code calling those tools.
"""

from unittest import mock

import tools.code_execution_tool as ce


def _with_config(value):
    """Patch the module's ``code_execution`` section to carry ``sandbox_tools: value``."""
    return mock.patch.object(ce, "_load_config", lambda: {"sandbox_tools": value})


WEB = {"web_search", "web_extract"}
SHELL = {"terminal", "read_file", "write_file", "patch", "search_files"}


def test_absent_key_keeps_the_documented_fallback():
    """null = every sandbox tool. The upstream contract must not shift under existing deploys."""
    with _with_config(None):
        assert ce._configured_sandbox_tools() == ce.SANDBOX_ALLOWED_TOOLS
        assert ce._sandbox_tools_for(None) == ce.SANDBOX_ALLOWED_TOOLS


def test_explicit_list_narrows_the_ceiling():
    with _with_config(["web_search", "web_extract"]):
        assert ce._configured_sandbox_tools() == WEB


def test_shell_and_file_tools_are_unreachable_when_narrowed():
    """The whole point: enabled_tools is always None in production, so the fall-back branch is
    what actually governs, and it must not hand back terminal/read_file/write_file."""
    with _with_config(["web_search", "web_extract"]):
        for enabled in (None, [], ["terminal"], ["read_file", "vision_analyze"]):
            got = ce._sandbox_tools_for(enabled)
            assert not (got & SHELL), f"{sorted(got & SHELL)} reachable with enabled_tools={enabled}"


def test_nonoverlapping_still_falls_back_but_only_to_the_ceiling():
    """``test_nonoverlapping_tools_fallback`` keeps passing (non-overlap -> fall back), while the
    set it falls back to is now the operator's, not everything allowed."""
    with _with_config(["web_search", "web_extract"]):
        assert ce._sandbox_tools_for(["vision_analyze", "browser_snapshot"]) == WEB


def test_intersection_wins_when_it_is_non_empty():
    with _with_config(["web_search", "web_extract"]):
        assert ce._sandbox_tools_for(["web_search", "terminal"]) == {"web_search"}


def test_unknown_name_is_dropped_not_injected():
    """The key narrows; it can never widen the sandbox beyond SANDBOX_ALLOWED_TOOLS."""
    with _with_config(["web_search", "kanban_create", "not_a_tool"]), \
         mock.patch.object(ce.logger, "warning") as warn:
        assert ce._configured_sandbox_tools() == {"web_search"}
    assert warn.called, "an unknown name should be reported, not silently dropped"


def test_empty_list_means_no_tool_rpc():
    """[] is a legitimate 'no RPC at all', not a typo to fall back from."""
    with _with_config([]):
        assert ce._configured_sandbox_tools() == frozenset()
        assert ce._sandbox_tools_for(None) == frozenset()


def test_scalar_warns_and_keeps_every_tool():
    with _with_config("web_search"), mock.patch.object(ce.logger, "warning") as warn:
        assert ce._configured_sandbox_tools() == ce.SANDBOX_ALLOWED_TOOLS
    assert warn.called


def test_schema_advertises_only_the_ceiling():
    """The model must not read about a tool the bridge will refuse."""
    with _with_config(["web_search", "web_extract"]):
        schema = ce.build_execute_code_schema(ce._configured_sandbox_tools())
    desc = schema["description"]
    assert "web_search" in desc
    # read_file/write_file/patch/search_files are doc-line driven and must vanish. `terminal`
    # survives in two prose asides (the interpreter note, a helper's docstring) that are not
    # import stubs, so it is excluded from this assertion by design.
    for name in ("read_file", "write_file", "patch", "search_files"):
        assert name not in desc, f"{name} still advertised in the execute_code schema"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

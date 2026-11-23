"""Ported from hermes-antigravity-acp tests (test_runtime_safety_isolation.py env-key
enumeration, test_acp_stderr_no_echo.py diagnostic helper) against the vendored module."""

import hashlib
import re

import pytest

from antigravity_acp_mcp import _upstream as up

# Credential-shaped fragments assembled at runtime, as upstream does.
TOKEN = "SYNTH" + ".EARLYEXIT" + ".TOKEN" + ".VALUE" + ".AAAA"
BEARER = "SYNTHETICBearerValue" + "BBBB"
LINES = ["routine chatter", "login dump: access_token=" + TOKEN, "Bea" + "rer " + BEARER]
# Real 429 line sample captured 2026-11-23 (PlanC component 8).
REAL_429 = (
    "harness stderr: ERROR: logging before google.Init: E1123 errorreport.go:273] "
    "Error 429, Message: Resource has been exhausted (e.g. check quota)., "
    "Status: RESOURCE_EXHAUSTED"
)


def test_env_keys_are_enumerated_never_prefixed():
    for name in ("XDG_ACCESS_TOKEN", "XDG_RUNTIME_DIR", "LC_API_TOKEN", "LC_PAPER",
                 "xdg_private_key", "OPENAI_API_KEY", "DISCORD_BOT_TOKEN", "ACP_MUX_AFFINITY"):
        assert up._is_acp_child_env_key(name) is False, name
    for name in ("PATH", "HOME", "LC_ALL", "XDG_CACHE_HOME", "SYSTEMROOT"):
        assert up._is_acp_child_env_key(name) is True, name
        assert up._is_acp_child_env_key(name.lower()) is True, name
    narrowed = up._acp_child_env_filter(
        {"PATH": "/bin", "XDG_ACCESS_TOKEN": "x", "LC_API_TOKEN": "y", "xdg_private_key": "z"}
    )
    assert set(narrowed) == {"PATH"}


def test_xdg_base_admitted_only_inside_profile_home(tmp_path):
    inside, outside = str(tmp_path / "home" / ".cache"), str(tmp_path / "other" / ".cache")
    (tmp_path / "home").mkdir()
    narrowed = up._acp_child_env_filter(
        {"XDG_CACHE_HOME": inside, "XDG_CONFIG_HOME": outside}, profile_home=tmp_path / "home"
    )
    assert narrowed == {"XDG_CACHE_HOME": inside}


def test_build_subprocess_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "planted")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "planted")
    env = up._build_subprocess_env()
    assert "OPENAI_API_KEY" not in env and "DISCORD_BOT_TOKEN" not in env
    assert env["HOME"] and "PATH" in env


def test_stderr_diagnostic_never_echoes_and_is_deterministic():
    summary = up._acp_stderr_diagnostic(LINES)
    for text in (*LINES, TOKEN, BEARER, "routine chatter"):
        assert text not in summary
    assert "3 line(s)" in summary
    assert "token-assignment" in summary and "bearer-token" in summary
    assert summary == up._acp_stderr_diagnostic(LINES)
    assert re.search(r"sha256=[0-9a-f]{16}", summary) and len(summary) <= 400
    assert hashlib.sha256("\n".join(LINES).encode()).hexdigest() not in summary


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([REAL_429], "rate-limit"),
        (["upstream: 429 too many requests"], "rate-limit"),
        (["login: authentication failed"], "authentication"),
        (["nothing to see"], None),
    ],
)
def test_stderr_retry_class_is_a_fixed_name(lines, expected):
    assert up._acp_stderr_retry_class(lines) == expected

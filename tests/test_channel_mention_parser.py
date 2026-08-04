"""Markdown-safe bare @mention extraction for channel posts."""

from __future__ import annotations

import pytest

from mcp_agent_mail.utils import extract_channel_mentions


def test_basic_order_case_dedup_and_valid_explicit_ids():
    body = "@BlueLake @opencode-main @blueLake @alpha.one @9-worker"
    assert extract_channel_mentions(body) == [
        "BlueLake",
        "opencode-main",
        "alpha.one",
        "9-worker",
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("contact a@b.com, then @BlueLake", ["BlueLake"]),
        ("@@BlueLake and @BlueLake", ["BlueLake"]),
        ("(@BlueLake), @RedRiver!", ["BlueLake", "RedRiver"]),
        ("@foo.bar next", ["foo.bar"]),
        ("sentence @BlueLake.", ["BlueLake"]),
        ("@BlueLake@other-project @RedRiver", ["RedRiver"]),
        ("@other-project:slug#BlueLake @RedRiver", ["RedRiver"]),
    ],
)
def test_boundaries_and_unsupported_explicit_styles(body, expected):
    assert extract_channel_mentions(body) == expected


def test_length_boundary_rejects_whole_overlong_token():
    accepted = "a" + "1" * 127
    rejected = accepted + "2"
    assert extract_channel_mentions(f"@{accepted}") == [accepted]
    assert extract_channel_mentions(f"@{rejected}") == []
    assert extract_channel_mentions(f"@{accepted}-") == []


@pytest.mark.parametrize(
    "body",
    [
        "`@hidden` @real-agent",
        "``@hidden`` @real-agent",
        "``````@hidden`````` @real-agent",
        "before `@hidden without a closer",
        "``@hidden ` nested`` @real-agent",
    ],
)
def test_inline_code_spans_do_not_leak(body):
    expected = [] if "without a closer" in body else ["real-agent"]
    assert extract_channel_mentions(body) == expected


@pytest.mark.parametrize("fence", ["```", "~~~", "````", "~~~~"])
def test_closed_fenced_code_does_not_leak(fence):
    body = f"@before-agent\n{fence}python\n@hidden\n{fence}\n@after-agent"
    assert extract_channel_mentions(body) == ["before-agent", "after-agent"]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_unclosed_fenced_code_is_hidden_to_eof(fence):
    assert extract_channel_mentions(f"@before-agent\n{fence}\n@hidden") == ["before-agent"]


def test_indented_code_lines_do_not_leak():
    body = "    @hidden\n\t@also-hidden\n@real-agent"
    assert extract_channel_mentions(body) == ["real-agent"]


@pytest.mark.parametrize(
    "body",
    [
        "https://example.test/@hidden @real-agent",
        "/path/to/@hidden @real-agent",
        "[profile](https://example.test/@hidden) @real-agent",
        "[relative](@hidden) @real-agent",
        "@scope/package @real-agent",
    ],
)
def test_url_and_path_contexts_do_not_leak(body):
    assert extract_channel_mentions(body) == ["real-agent"]


def test_empty_and_unknown_are_pure_syntax():
    assert extract_channel_mentions("") == []
    assert extract_channel_mentions("@NotRegistered") == ["NotRegistered"]


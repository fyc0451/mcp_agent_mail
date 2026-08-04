"""M1b-3c-b: markdown-aware @-mention extractor (pure helper, no DB/delivery).

Covers ``extract_channel_mentions``: pull bare @-mention names out of a channel
message body while ignoring code spans, emails, @@, decorator false positives,
and unsupported explicit cross-project prose addressing. Pure *syntax* extraction
only — it does not resolve whether a name is a registered agent.
"""

from __future__ import annotations

import pytest

from mcp_agent_mail.utils import extract_channel_mentions


class TestBasicExtraction:
    def test_single_mention(self):
        assert extract_channel_mentions("hi @BlueLake") == ["BlueLake"]

    def test_explicit_id_with_separator(self):
        assert extract_channel_mentions("@opencode-main please") == ["opencode-main"]
        assert extract_channel_mentions("@worker_42 report") == ["worker_42"]
        assert extract_channel_mentions("@cc-0 status") == ["cc-0"]

    def test_dot_bearing_and_digit_led_explicit_ids(self):
        assert extract_channel_mentions("@alpha.one review") == ["alpha.one"]
        assert extract_channel_mentions("@9-worker here") == ["9-worker"]

    def test_multiple_mentions_stable_first_occurrence_order(self):
        body = "@opencode-main then @BlueLake then @opencode-main again"
        assert extract_channel_mentions(body) == ["opencode-main", "BlueLake"]

    def test_multiline(self):
        body = "line1 @BlueLake\nline2 @opencode-main\nline3 @worker_42"
        assert extract_channel_mentions(body) == ["BlueLake", "opencode-main", "worker_42"]

    def test_empty_and_no_mention(self):
        assert extract_channel_mentions("") == []
        assert extract_channel_mentions("no mentions here, just prose") == []


class TestDedup:
    def test_case_insensitive_dedup_keeps_first_spelling(self):
        # Three spellings of the SAME name collapse to one entry, first kept.
        assert extract_channel_mentions("@BlueLake @bluelake @BLUELAKE") == ["BlueLake"]
        assert extract_channel_mentions("@bluelake @BlueLake") == ["bluelake"]

    def test_distinct_names_all_kept_in_order(self):
        body = "@BlueLake @opencode-main @worker_42"
        assert extract_channel_mentions(body) == ["BlueLake", "opencode-main", "worker_42"]


class TestCodeIgnored:
    def test_fenced_code_block_ignored(self):
        body = (
            "ok @BlueLake\n"
            "```\n"
            "@opencode-main\n"
            "@staticmethod\n"
            "```\n"
            "after @worker_42"
        )
        assert extract_channel_mentions(body) == ["BlueLake", "worker_42"]

    def test_unclosed_fence_treated_as_code_to_eof(self):
        # An unterminated ``` fence is code through EOF; @hidden must NOT leak.
        body = "@BlueLake\n```\n@hidden then more @alsohidden"
        assert extract_channel_mentions(body) == ["BlueLake"]

    def test_inline_code_ignored(self):
        body = "see `@BlueLake` in code, but @opencode-main is real"
        assert extract_channel_mentions(body) == ["opencode-main"]


class TestEmailAndDoubleAt:
    def test_email_not_extracted(self):
        body = "contact a@b.com or user@example.com, then @BlueLake"
        assert extract_channel_mentions(body) == ["BlueLake"]

    def test_double_at_prefix_ignored(self):
        body = "@@BlueLake and @BlueLake"
        assert extract_channel_mentions(body) == ["BlueLake"]


class TestPunctuationBoundary:
    @pytest.mark.parametrize("trailing", [",", "!", "?", ";", ")", "]", "}"])
    def test_trailing_punctuation_kept_out(self, trailing):
        assert extract_channel_mentions(f"@BlueLake{trailing} next") == ["BlueLake"]

    def test_sentence_final_dot_trimmed(self):
        assert extract_channel_mentions("@BlueLake. next") == ["BlueLake"]
        assert extract_channel_mentions("end @BlueLake.") == ["BlueLake"]

    def test_colon_before_whitespace_is_plain_boundary(self):
        # "@Name:" followed by whitespace is punctuation, NOT project addressing.
        assert extract_channel_mentions("@BlueLake: please review") == ["BlueLake"]

    def test_surrounding_punctuation(self):
        assert extract_channel_mentions("(@BlueLake)") == ["BlueLake"]


class TestExplicitIdShape:
    def test_mid_name_dot_kept_not_truncated(self):
        # @foo.bar must yield "foo.bar"; it must NEVER degrade to a "foo" prefix.
        assert extract_channel_mentions("@foo.bar next") == ["foo.bar"]

    def test_alpha_one_kept(self):
        assert extract_channel_mentions("@alpha.one cc") == ["alpha.one"]

    def test_digit_led_id_accepted(self):
        assert extract_channel_mentions("@9-worker go") == ["9-worker"]


class TestExplicitAddressingRejected:
    def test_at_name_at_project_rejected(self):
        # Initial 3c has no prose addressing; must not degrade to a bare Name.
        assert extract_channel_mentions("@BlueLake@other-project") == []

    def test_project_slug_hash_name_rejected(self):
        assert extract_channel_mentions("@other-project:slug#BlueLake") == []

    def test_explicit_addressing_does_not_shadow_later_real_mention(self):
        body = "@BlueLake@x and @opencode-main"
        assert extract_channel_mentions(body) == ["opencode-main"]


class TestLengthAndIllegal:
    def test_max_length_128_accepted(self):
        name = "a" + "1" * 127  # exactly 128 chars
        assert extract_channel_mentions(f"@{name}") == [name]

    def test_over_max_length_rejected(self):
        name = "a" + "1" * 128  # 129 chars -> whole token rejected, not truncated
        assert extract_channel_mentions(f"@{name}") == []

    def test_max_length_id_then_id_char_rejected(self):
        # A 128-char id immediately followed by another id-body char (. or -) is a
        # longer token; it must NOT be truncated to the first 128 chars and accepted.
        name128 = "a" + "1" * 127
        assert extract_channel_mentions(f"@{name128}.") == []
        assert extract_channel_mentions(f"@{name128}-") == []

    @pytest.mark.parametrize("lead", ["@", ".", "-", "_"])
    def test_leading_separator_or_at_is_illegal(self, lead):
        # Must start alphanumeric; @@name / @.name / @-name / @_name all rejected.
        assert extract_channel_mentions(f"@{lead}name") == []


class TestPureSyntaxOnly:
    def test_unregistered_name_still_extracted(self):
        assert extract_channel_mentions("@NotARegisteredAgent") == ["NotARegisteredAgent"]

    def test_mixed_realistic_channel_post(self):
        body = (
            "@codex-main @opencode-main please review PR #123.\n"
            "```\n"
            "@pytest.fixture\n"
            "def db(): ...\n"
            "```\n"
            "ping @codex-main again, cc @SwiftFox — not a@b.com."
        )
        assert extract_channel_mentions(body) == ["codex-main", "opencode-main", "SwiftFox"]

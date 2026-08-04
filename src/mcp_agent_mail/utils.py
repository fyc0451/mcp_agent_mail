"""Utility helpers for the MCP Agent Mail service."""

from __future__ import annotations

import random
import re
from typing import Iterable, Optional

# Agent name word lists - used to generate memorable adjective+noun combinations
# These lists are designed to provide a large namespace (62 x 69 = 4278 combinations)
# while keeping names easy to remember, spell, and distinguish.
#
# Design principles:
# - All words are capitalized for consistent CamelCase output (e.g., "GreenLake")
# - Adjectives are colors, weather, materials, and nature-themed descriptors
# - Nouns are nature, geography, animals, and simple objects
# - No offensive, controversial, or confusing words
# - No words that could be easily misspelled or confused with each other

ADJECTIVES: Iterable[str] = (
    # Colors (original + expanded)
    "Red",
    "Orange",
    "Pink",
    "Black",
    "Purple",
    "Blue",
    "Brown",
    "White",
    "Green",
    "Chartreuse",
    "Lilac",
    "Fuchsia",
    "Azure",
    "Amber",
    "Coral",
    "Crimson",
    "Cyan",
    "Gold",
    "Gray",
    "Indigo",
    "Ivory",
    "Jade",
    "Lavender",
    "Magenta",
    "Maroon",
    "Navy",
    "Olive",
    "Pearl",
    "Rose",
    "Ruby",
    "Sage",
    "Scarlet",
    "Silver",
    "Teal",
    "Topaz",
    "Violet",
    "Cobalt",
    "Copper",
    "Bronze",
    "Emerald",
    "Sapphire",
    "Turquoise",
    # Weather and nature
    "Sunny",
    "Misty",
    "Foggy",
    "Stormy",
    "Windy",
    "Frosty",
    "Dusty",
    "Hazy",
    "Cloudy",
    "Rainy",
    # Descriptive
    "Swift",
    "Quiet",
    "Bold",
    "Calm",
    "Bright",
    "Dark",
    "Wild",
    "Silent",
    "Gentle",
    "Rustic",
)

NOUNS: Iterable[str] = (
    # Original nouns
    "Stone",
    "Lake",
    "Dog",
    "Creek",
    "Pond",
    "Cat",
    "Bear",
    "Mountain",
    "Hill",
    "Snow",
    "Castle",
    # Geography and nature
    "River",
    "Forest",
    "Valley",
    "Canyon",
    "Meadow",
    "Prairie",
    "Desert",
    "Island",
    "Cliff",
    "Cave",
    "Glacier",
    "Waterfall",
    "Spring",
    "Stream",
    "Reef",
    "Dune",
    "Ridge",
    "Peak",
    "Gorge",
    "Marsh",
    "Brook",
    "Glen",
    "Grove",
    "Hollow",
    "Basin",
    "Cove",
    "Bay",
    "Harbor",
    # Animals
    "Fox",
    "Wolf",
    "Hawk",
    "Eagle",
    "Owl",
    "Deer",
    "Elk",
    "Moose",
    "Falcon",
    "Raven",
    "Heron",
    "Crane",
    "Otter",
    "Beaver",
    "Badger",
    "Finch",
    "Robin",
    "Sparrow",
    "Lynx",
    "Puma",
    # Objects and structures
    "Tower",
    "Bridge",
    "Forge",
    "Mill",
    "Barn",
    "Gate",
    "Anchor",
    "Lantern",
    "Beacon",
    "Compass",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_AGENT_NAME_RE = re.compile(r"[^A-Za-z0-9]+")
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

# Pre-built frozenset of all valid agent names (lowercase) for O(1) validation lookup.
# This is computed once at module load time rather than O(n*m) per validation call.
_VALID_AGENT_NAMES: frozenset[str] = frozenset(
    f"{adj}{noun}".lower() for adj in ADJECTIVES for noun in NOUNS
)


def slugify(value: str) -> str:
    """Normalize a human-readable value into a slug."""
    normalized = value.strip().lower()
    slug = _SLUG_RE.sub("-", normalized).strip("-")
    return slug or "project"


def generate_agent_name() -> str:
    """Return a random adjective+noun combination."""
    adjective = random.choice(tuple(ADJECTIVES))
    noun = random.choice(tuple(NOUNS))
    return f"{adjective}{noun}"


def validate_agent_name_format(name: str) -> bool:
    """
    Validate that an agent name matches the required adjective+noun format.

    CRITICAL: Agent names MUST be randomly generated two-word combinations
    like "GreenLake" or "BlueDog", NOT descriptive names like "BackendHarmonizer".

    Names should be:
    - Unique and easy to remember
    - NOT descriptive of the agent's role or task
    - One of the predefined adjective+noun combinations

    Note: This validation is case-insensitive to match the database behavior
    where "GreenLake", "greenlake", and "GREENLAKE" are treated as the same.

    Returns True if valid, False otherwise.
    """
    if not name:
        return False

    # O(1) lookup using pre-built frozenset (vs O(n*m) iteration)
    return name.lower() in _VALID_AGENT_NAMES


_EXPLICIT_ID_SEPARATOR_RE = re.compile(r"[._-]")


def validate_explicit_agent_id(name: str) -> bool:
    """Validate that a caller-supplied identity is safe for use as an agent name.

    Explicit IDs allow stable, human-chosen identities like ``cc-0``,
    ``alpha-one``, or ``worker_42`` — useful for swarm workflows where agents
    are relaunched onto the same identity.  The format mirrors thread IDs:
    ASCII alphanumerics plus ``._-``, starting with an alphanumeric, max 128
    characters.

    To distinguish explicit IDs from adjective+noun names, the ID must
    contain at least one separator character (``-``, ``_``, or ``.``).
    Purely alphanumeric strings go through the adjective+noun validation
    path instead.
    """
    if not name:
        return False
    if not _THREAD_ID_RE.fullmatch(name):
        return False
    # Require at least one separator so purely-alphanumeric strings like
    # "BackendHarmonizer" still go through adjective+noun validation.
    return _EXPLICIT_ID_SEPARATOR_RE.search(name) is not None


def sanitize_agent_name(value: str) -> Optional[str]:
    """Normalize user-provided agent name; return None if nothing remains."""
    cleaned = _AGENT_NAME_RE.sub("", value.strip())
    if not cleaned:
        return None
    return cleaned[:128]


# Markdown-aware @-mention extraction for channel message bodies. The accepted
# name shape mirrors valid explicit agent ids (see _THREAD_ID_RE): ASCII
# alphanumeric start, then [A-Za-z0-9._-], max 128 — so dot-bearing ids like
# "alpha.one" and digit-led ids like "9-worker" are accepted. A trailing '.' is
# trimmed so a sentence-final "@Name." still yields "Name", while a mid-name '.'
# is kept ("@foo.bar" -> "foo.bar", never truncated to "foo"). Explicit cross-
# project prose addressing ("@Name@project", "@project:slug#Name") is NOT
# supported by the initial 3c and is rejected wholesale to avoid misdelivery.
_CHANNEL_MENTION_NAME_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9][A-Za-z0-9._-]{0,127})(?![A-Za-z0-9._-])")
# A fenced block is either a closed ```...``` pair or, if never closed, code
# through end-of-string — an unclosed fence must not leak its @ tokens.
_FENCED_CODE_SPAN_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")


def _is_explicit_id_body_char(ch: str) -> bool:
    """Whether a character may continue an explicit agent id body."""
    return bool(ch) and ch.isascii() and (ch.isalnum() or ch in "._-")


def extract_channel_mentions(body_md: str) -> list[str]:
    """Extract @-mention name tokens from a markdown body, code/email-safe.

    Pure syntax extraction only — returns first-occurrence-ordered, case-
    insensitive-deduplicated names, preserving the first-seen spelling. Does
    NOT resolve whether the names are registered agents; a downstream resolver
    silently skips unknown names.

    Accepted name shape mirrors valid explicit agent ids (alphanumeric start,
    [A-Za-z0-9._-] body, max 128): "BlueLake", "opencode-main", "alpha.one",
    "9-worker". A trailing '.' is trimmed ("@Name." -> "Name") but a mid-name
    '.' is kept ("@foo.bar" -> "foo.bar", never truncated to "foo").

    Ignored: fenced ``` (closed or unclosed-to-EOF) and inline `code` spans,
    email like a@b.com, "@@name", and explicit cross-project prose addressing
    ("@Name@project", "@project:slug#Name") which the initial 3c does not
    support. Ordinary punctuation boundaries ("@Name," / "@Name!" / "@Name:"
    before whitespace) yield the bare name.
    """
    if not body_md:
        return []
    # Strip code spans first so @tokens inside code are never extracted.
    stripped = _FENCED_CODE_SPAN_RE.sub(" ", body_md)
    stripped = _INLINE_CODE_SPAN_RE.sub(" ", stripped)
    seen: set[str] = set()
    names: list[str] = []
    for match in _CHANNEL_MENTION_NAME_RE.finditer(stripped):
        name = match.group(1)
        # The trailing-dot negative-lookahead above already rejected any token
        # whose next char is still an id-body char (so 129-char tokens, or a
        # 128-char id followed by another id char, are NOT truncated to 128 and
        # accepted). Here a trailing '.' that DID land inside the match is treated
        # as sentence punctuation in prose (the id grammar permits a trailing dot,
        # but a mention reads as a natural-language token), so trim it.
        candidate = name.rstrip(".")
        if not candidate or not _THREAD_ID_RE.fullmatch(candidate):
            continue
        # Reject explicit cross-project prose addressing so it cannot degrade
        # into a bare mention and misdeliver: "@Name@project" (next char '@')
        # and "@project:slug#Name" (':' followed by an id-body char).
        end = match.end()
        tail = stripped[end:end + 1]
        if tail == "@":
            continue
        if tail == ":" and _is_explicit_id_body_char(stripped[end + 1:end + 2]):
            continue
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            names.append(candidate)
    return names


def validate_thread_id_format(thread_id: str) -> bool:
    """Validate that a thread_id is safe for filenames and indexing.

    Thread IDs are used as human-facing keys and may also be used in filesystem
    paths for thread digests. For safety and portability, enforce:
    - ASCII alphanumerics plus '.', '_', '-'
    - Must start with an alphanumeric character
    - Max length 128
    """
    candidate = (thread_id or "").strip()
    if not candidate:
        return False
    return _THREAD_ID_RE.fullmatch(candidate) is not None

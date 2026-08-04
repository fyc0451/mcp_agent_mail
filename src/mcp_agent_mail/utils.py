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


_CHANNEL_MENTION_RE = re.compile(
    r"(?<![\w@])@([A-Za-z0-9][A-Za-z0-9._-]{0,127})(?![A-Za-z0-9._-])"
)
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BACKTICK_RUN_RE = re.compile(r"`+")


def _mask_markdown_code(text: str) -> str:
    """Blank Markdown code while preserving offsets and line boundaries."""
    masked_lines: list[str] = []
    fence_char = ""
    fence_length = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        leading_spaces = len(content) - len(content.lstrip(" "))
        stripped = content[leading_spaces:]
        if fence_char:
            masked_lines.append("".join(ch if ch in "\r\n" else " " for ch in line))
            if leading_spaces <= 3 and stripped.startswith(fence_char * fence_length):
                run_length = len(stripped) - len(stripped.lstrip(fence_char))
                if run_length >= fence_length and not stripped[run_length:].strip():
                    fence_char = ""
                    fence_length = 0
            continue

        opener = _FENCE_OPEN_RE.match(content)
        if opener and not (opener.group(1).startswith("`") and "`" in content[opener.end():]):
            fence_char = opener.group(1)[0]
            fence_length = len(opener.group(1))
            masked_lines.append("".join(ch if ch in "\r\n" else " " for ch in line))
        elif content.startswith("\t") or content.startswith("    "):
            masked_lines.append("".join(ch if ch in "\r\n" else " " for ch in line))
        else:
            masked_lines.append(line)

    masked = "".join(masked_lines)
    chars = list(masked)
    position = 0
    while opener := _BACKTICK_RUN_RE.search(masked, position):
        run_length = opener.end() - opener.start()
        closer = _BACKTICK_RUN_RE.search(masked, opener.end())
        while closer is not None and closer.end() - closer.start() != run_length:
            closer = _BACKTICK_RUN_RE.search(masked, closer.end())
        end = closer.end() if closer is not None else len(masked)
        for index in range(opener.start(), end):
            if chars[index] not in "\r\n":
                chars[index] = " "
        position = end
    return "".join(chars)


def _mention_is_in_url_or_path(text: str, start: int, end: int) -> bool:
    token_start = start
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    prefix = text[token_start:start]
    if "/" in prefix or "](" in prefix or any(marker in prefix for marker in ("?", "#", "&", "=")):
        return True
    if re.search(r"(?:https?|ftp|file|mailto):$", prefix, re.IGNORECASE):
        return True
    return text[end:end + 1] == "/"


def _is_explicit_id_body_char(ch: str) -> bool:
    return bool(ch) and ch.isascii() and (ch.isalnum() or ch in "._-")


def extract_channel_mentions(body_md: str) -> list[str]:
    """Extract bare @agent ids from Markdown, excluding code and URL contexts."""
    if not body_md:
        return []
    text = _mask_markdown_code(body_md)
    names: list[str] = []
    seen: set[str] = set()
    for match in _CHANNEL_MENTION_RE.finditer(text):
        candidate = match.group(1).rstrip(".")
        if not candidate or not _THREAD_ID_RE.fullmatch(candidate):
            continue
        end = match.end()
        tail = text[end:end + 1]
        if tail == "@" or (tail == ":" and _is_explicit_id_body_char(text[end + 1:end + 2])):
            continue
        if _mention_is_in_url_or_path(text, match.start(), end):
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

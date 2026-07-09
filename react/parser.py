"""
Parser for Vanilla ReAct model outputs.

Expected model output:

Thought: ...
Action: Search[entity]

or

Thought: ...
Action: Lookup[keyword]

or

Thought: ...
Action: Finish[answer]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Compile once for speed.
THOUGHT_RE = re.compile(
    r"Thought\s*:\s*(.*?)(?=\n\s*Action\s*:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

ACTION_RE = re.compile(
    r"Action\s*:\s*([A-Za-z]+)\s*\[(.*?)\]",
    flags=re.IGNORECASE | re.DOTALL,
)


VALID_ACTIONS = {
    "search",
    "lookup",
    "finish",
}


@dataclass(slots=True)
class ParsedStep:
    """
    Parsed ReAct step.
    """

    thought: str
    action: str
    argument: str

    @property
    def is_finish(self) -> bool:
        return self.action == "finish"


class ParseError(Exception):
    """Raised when model output cannot be parsed."""


def parse_step(text: str) -> ParsedStep:
    """
    Parse one ReAct generation.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    ParsedStep

    Raises
    ------
    ParseError
        If the output is malformed.
    """

    text = text.strip()

    thought_match = THOUGHT_RE.search(text)
    action_match = ACTION_RE.search(text)

    if action_match is None:
        raise ParseError(
            f"No valid Action found.\n\n{text}"
        )

    thought = ""

    if thought_match:
        thought = normalize_text(
            thought_match.group(1)
        )

    action = action_match.group(1).strip().lower()

    if action not in VALID_ACTIONS:
        raise ParseError(
            f"Unknown action '{action}'."
        )

    argument = normalize_text(
        action_match.group(2)
    )

    if not argument:
        raise ParseError(
            "Action argument is empty."
        )

    return ParsedStep(
        thought=thought,
        action=action,
        argument=argument,
    )


def normalize_text(text: str) -> str:
    """
    Clean whitespace while preserving content.
    """

    return " ".join(text.strip().split())


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final answer if Finish[...] exists.

    Returns
    -------
    str | None
    """

    try:
        step = parse_step(text)

        if step.is_finish:
            return step.argument

    except ParseError:
        return None

    return None


def is_valid_action(text: str) -> bool:
    """
    Quickly check whether the output contains
    a valid ReAct action.
    """

    try:
        parse_step(text)
        return True
    except ParseError:
        return False
"""
Parser for Vanilla ReAct outputs.

Expected model output:

Thought: ...

Action: Search[...]

or

Thought: ...

Action: Lookup[...]

or

Thought: ...

Action: Finish[...]
"""

from __future__ import annotations

import re
from dataclasses import dataclass


VALID_ACTIONS = {
    "search",
    "lookup",
    "finish",
}


@dataclass(slots=True)
class ParsedStep:
    thought: str
    action: str
    argument: str


class ParseError(Exception):
    pass


class ReActParser:

    ACTION_PATTERN = re.compile(
        r"Action\s*:\s*(Search|Lookup|Finish)\s*\[(.*?)\]",
        re.IGNORECASE | re.DOTALL,
    )

    THOUGHT_PATTERN = re.compile(
        r"Thought\s*:\s*(.*?)(?=\n\s*Action\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.strip().split())

    @classmethod
    def parse(cls, text: str) -> ParsedStep:

        text = text.strip()

        action_match = cls.ACTION_PATTERN.search(text)

        if action_match is None:
            raise ParseError(
                f"No valid action found.\n\n{text}"
            )

        thought_match = cls.THOUGHT_PATTERN.search(text)

        thought = ""

        if thought_match:
            thought = cls.normalize(
                thought_match.group(1)
            )

        action = action_match.group(1).lower()

        argument = cls.normalize(
            action_match.group(2)
        )

        if action not in VALID_ACTIONS:
            raise ParseError(
                f"Unknown action {action}"
            )

        if not argument:
            raise ParseError(
                "Empty action argument."
            )

        return ParsedStep(
            thought=thought,
            action=action,
            argument=argument,
        )

    @classmethod
    def is_finish(cls, text: str):

        try:

            step = cls.parse(text)

            return step.action == "finish"

        except ParseError:

            return False

    @classmethod
    def extract_answer(cls, text: str):

        try:

            step = cls.parse(text)

            if step.action == "finish":
                return step.argument

        except ParseError:

            pass

        return None
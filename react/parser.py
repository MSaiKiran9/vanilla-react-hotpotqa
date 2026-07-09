"""Parser for Vanilla ReAct model outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass


VALID_ACTIONS = {"search", "lookup", "finish"}


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
    BARE_ACTION_PATTERN = re.compile(
        r"^\s*(Search|Lookup|Finish)\s*\[(.*?)\]",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    THOUGHT_PATTERN = re.compile(
        r"Thought\s*:\s*(.*?)(?=\n\s*Action\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    FINAL_ANSWER_PATTERN = re.compile(
        r"(?:Final Answer|Answer)\s*:\s*(.+)",
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
            action_match = cls.BARE_ACTION_PATTERN.search(text)

        if action_match is None:
            fallback = cls.extract_answer(text)
            if fallback:
                return ParsedStep(thought="", action="finish", argument=fallback)
            raise ParseError(f"No valid action found.\n\n{text}")

        thought_match = cls.THOUGHT_PATTERN.search(text)
        thought = cls.normalize(thought_match.group(1)) if thought_match else ""
        action = action_match.group(1).lower()
        argument = cls.normalize(action_match.group(2))

        if action not in VALID_ACTIONS:
            raise ParseError(f"Unknown action {action}")
        if not argument:
            raise ParseError("Empty action argument.")

        return ParsedStep(thought=thought, action=action, argument=argument)

    @classmethod
    def extract_answer(cls, text: str) -> str | None:
        action_match = cls.ACTION_PATTERN.search(text)
        if action_match and action_match.group(1).lower() == "finish":
            answer = cls.normalize(action_match.group(2))
            return answer or None

        fallback = cls.FINAL_ANSWER_PATTERN.search(text.strip())
        if fallback:
            answer = fallback.group(1).splitlines()[0]
            answer = cls.normalize(answer)
            return answer or None

        return None

"""Vanilla ReAct agent implementation."""

from __future__ import annotations

from dataclasses import dataclass

from config import MAX_REACT_STEPS
from react.parser import ParseError, ReActParser
from react.prompt import PromptBuilder


@dataclass(slots=True)
class AgentResult:
    answer: str
    iterations: int
    retrieval_failures: int
    reasoning_failures: int


class ReActAgent:
    """Run Thought -> Action -> Observation steps until Finish."""

    def __init__(self, loader, wiki):
        self.loader = loader
        self.wiki = wiki
        self.parser = ReActParser()

    def answer(self, question: str) -> AgentResult:
        builder = PromptBuilder(question)
        self.wiki.reset()

        retrieval_failures = 0
        reasoning_failures = 0

        for iteration in range(1, MAX_REACT_STEPS + 1):
            output = self.loader.generate(builder.build())

            try:
                step = self.parser.parse(output)
            except ParseError:
                reasoning_failures += 1
                return AgentResult(
                    answer="",
                    iterations=iteration,
                    retrieval_failures=retrieval_failures,
                    reasoning_failures=reasoning_failures,
                )

            if step.action == "finish":
                return AgentResult(
                    answer=step.argument,
                    iterations=iteration,
                    retrieval_failures=retrieval_failures,
                    reasoning_failures=reasoning_failures,
                )

            action_text = f"{step.action.title()}[{step.argument}]"

            try:
                if step.action == "search":
                    observation = self.wiki.search_action(step.argument)
                elif step.action == "lookup":
                    observation = self.wiki.lookup_action(step.argument)
                else:
                    reasoning_failures += 1
                    observation = "Invalid action."
            except Exception as exc:
                observation = f"Retrieval error: {exc}"

            if self._is_retrieval_failure(observation):
                retrieval_failures += 1

            builder.add_step(step.thought, action_text, observation)

        reasoning_failures += 1
        return AgentResult(
            answer="",
            iterations=MAX_REACT_STEPS,
            retrieval_failures=retrieval_failures,
            reasoning_failures=reasoning_failures,
        )

    @staticmethod
    def _is_retrieval_failure(observation: str) -> bool:
        failure_markers = (
            "No Wikipedia page found.",
            "No page opened.",
            "Keyword not found.",
            "Retrieval error:",
        )
        return observation.startswith(failure_markers)

"""Prompt builder for the Vanilla ReAct baseline.

This module builds the prompts sent to the language model. The interaction is
the original ReAct loop:

Question -> Thought -> Action -> Observation -> Thought ...

The model is expected to generate one Thought and one Action at each iteration.
"""


SYSTEM_PROMPT = """You are a helpful research assistant.

Answer the question using only the ReAct format.

Search[entity]
Search Wikipedia for an entity.

Lookup[keyword]
Search the currently opened Wikipedia page for a keyword.

Finish[answer]
Return the final answer.

Rules:

1. Output exactly one Thought line and exactly one Action line.
2. Do not output <think> tags.
3. Do not output an Observation.
4. Wait for the Observation before continuing.
5. Finish as soon as the answer is known.
6. If the Observation directly supports the answer, use Finish instead of another Search.

Canonical example:

Question: Where was Albert Einstein born?
Thought: I should search for Albert Einstein.
Action: Search[Albert Einstein]
Observation: Title: Albert Einstein

Albert Einstein was born in Ulm, Germany.
Thought: The observation states the birthplace.
Action: Finish[Ulm]
"""


class PromptBuilder:
    """Maintains the ReAct conversation."""

    def __init__(self, question: str):
        self.question = question
        self.history = []

    def add_step(self, thought: str, action: str, observation: str) -> None:
        """Add one completed ReAct step."""

        self.history.append(
            {
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        )

    def build(self) -> str:
        """Build the complete prompt."""

        lines = [SYSTEM_PROMPT, "", f"Question: {self.question}", ""]

        for step in self.history:
            lines.append(f"Thought: {step['thought']}")
            lines.append(f"Action: {step['action']}")
            lines.append(f"Observation: {step['observation']}")
            lines.append("")

        lines.append("Thought:")

        return "\n".join(lines)

    def reset(self) -> None:
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)

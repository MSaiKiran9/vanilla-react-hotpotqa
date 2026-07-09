"""Prompt builder for the Vanilla ReAct baseline.

This module builds the prompts sent to the language model. The interaction is
the original ReAct loop:

Question -> Thought -> Action -> Observation -> Thought ...

The model is expected to generate one Thought and one Action at each iteration.
"""


SYSTEM_PROMPT = """Answer using only the ReAct format.

Search[entity]
Lookup[keyword]
Finish[answer]

Rules:
1. Output exactly one Thought line.
2. Output exactly one Action line using Search, Lookup, or Finish.
3. Never output Observation.
4. Finish immediately when sufficient evidence exists.
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

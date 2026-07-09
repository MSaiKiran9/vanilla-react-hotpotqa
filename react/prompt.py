"""
Prompt builder for the Vanilla ReAct baseline.

This module builds the prompts sent to the language model.
The implementation follows the original ReAct interaction:

Question
↓

Thought
↓

Action

↓

Observation

↓

Thought ...

The model is expected to generate ONE Thought and ONE Action
at each iteration.
"""


SYSTEM_PROMPT = """You are a helpful research assistant.

Answer the question using the following actions.

Search[entity]
Search Wikipedia for an entity.

Lookup[keyword]
Search the currently opened Wikipedia page for a keyword.

Finish[answer]
Return the final answer.

Rules:

1. Think step-by-step.

2. Generate exactly ONE Thought.

3. Generate exactly ONE Action.

4. Never generate an Observation.

5. Wait for the Observation before continuing.

Format:

Thought: ...

Action: Search[...]

or

Thought: ...

Action: Lookup[...]

or

Thought: ...

Action: Finish[answer]
"""


class PromptBuilder:
    """
    Maintains the ReAct conversation.
    """

    def __init__(self, question: str):

        self.question = question

        self.history = []

    # ---------------------------------------------------------

    def add_step(
        self,
        thought: str,
        action: str,
        observation: str,
    ):
        """
        Add one completed ReAct step.
        """

        self.history.append(
            {
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        )

    # ---------------------------------------------------------

    def build(self):
        """
        Build the complete prompt.
        """

        lines = []

        lines.append(SYSTEM_PROMPT)
        lines.append("")
        lines.append(f"Question: {self.question}")
        lines.append("")

        for step in self.history:

            lines.append(
                f"Thought: {step['thought']}"
            )

            lines.append(
                f"Action: {step['action']}"
            )

            lines.append(
                f"Observation: {step['observation']}"
            )

            lines.append("")

        lines.append("Thought:")

        return "\n".join(lines)

    # ---------------------------------------------------------

    def reset(self):

        self.history.clear()

    # ---------------------------------------------------------

    def __len__(self):

        return len(self.history)
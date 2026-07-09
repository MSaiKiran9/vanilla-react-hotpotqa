"""
Prompt templates for the Vanilla ReAct agent.

The prompt closely follows the original ReAct format:

Thought -> Action -> Observation -> ...

Supported actions:

Search[entity]
Lookup[keyword]
Finish[answer]
"""

SYSTEM_PROMPT = """You are a helpful research assistant that answers multi-hop questions.

You must solve the question by reasoning step-by-step.

You may use the following actions:

Search[entity]
- Search Wikipedia for an entity.

Lookup[keyword]
- Look up a keyword in the currently opened Wikipedia page.

Finish[answer]
- Return the final answer.

Use the following format exactly:

Question: <question>

Thought: <reasoning>

Action: Search[...]

Observation: ...

Thought: ...

Action: Lookup[...]

Observation: ...

Thought: ...

Action: Finish[answer]

Only produce one Thought and one Action at a time.
Never produce an Observation yourself.
Wait for the Observation before continuing.
"""


def build_initial_prompt(question: str) -> str:
    """
    Build the initial prompt presented to the model.

    Parameters
    ----------
    question : str

    Returns
    -------
    str
    """

    return (
        SYSTEM_PROMPT
        + "\n\n"
        + f"Question: {question}\n"
    )


def append_step(
    conversation: str,
    thought: str,
    action: str,
    observation: str,
) -> str:
    """
    Append one completed ReAct step.
    """

    return (
        conversation
        + f"Thought: {thought}\n"
        + f"Action: {action}\n"
        + f"Observation: {observation}\n"
    )


def append_observation(
    conversation: str,
    observation: str,
) -> str:
    """
    Append an observation followed by a new Thought prompt.
    """

    return (
        conversation
        + f"Observation: {observation}\n"
        + "Thought: "
    )


def continue_prompt(
    conversation: str,
) -> str:
    """
    Return the prompt for the next generation.
    """

    if conversation.endswith("Thought: "):
        return conversation

    return conversation + "\nThought: "
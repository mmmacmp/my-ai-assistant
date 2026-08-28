import opik
from smolagents import tool


@opik.track(name="what_can_i_do")
@tool
def what_can_i_do(question: str) -> str:
    """Returns a list of example questions and capabilities available in this assistant.

    Use this tool when:
    - The user explicitly asks what the system can do
    - The user asks about available features or capabilities
    - The user seems unsure what questions they can ask

    Do NOT use this tool when:
    - The user asks a specific question with a clear topic
    - The question is already answerable via the retriever or summarizer

    Args:
        question: The user's query about system capabilities. The function returns a
            standard capability list regardless of the specific wording.

    Returns:
        A formatted string listing example questions the user can ask.
    """
    return """
You can ask questions about the documents in this knowledge base, such as:

- What is [topic] and how does it work?
- Summarize the key points about [topic]
- Compare [topic A] and [topic B]
- List the main approaches to [topic]

If you're not sure where to start, try asking a broad question about
a topic you're interested in and I'll retrieve relevant context to answer it.
"""
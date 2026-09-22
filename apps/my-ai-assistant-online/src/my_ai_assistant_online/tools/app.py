import argparse

from smolagents import GradioUI

from my_ai_assistant_online.application.agents.tools.agents import (
    AgentWrapper,
    get_agent,
)


DEFAULT_QUERY = "How does agent memory work?"


def build_agent() -> AgentWrapper:
    """Build the agent once when the CLI or UI starts."""
    return get_agent(
        collection_name=None,
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_model_type="huggingface",
        retriever_type="contextual",
    )


def run(*, ui: bool, query: str, share: bool) -> None:
    """Run either the interactive Gradio app or one CLI query."""
    agent = build_agent()

    if ui:
        GradioUI(agent, reset_agent_memory=False).launch(share=share)
        return

    print(agent.run(query))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Second Brain agent.")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Gradio chat UI instead of running one CLI query.",
    )
    parser.add_argument(
        "--query",
        "-q",
        default=DEFAULT_QUERY,
        help="Question to run in CLI mode.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link. By default the UI is local only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(ui=args.ui, query=args.query, share=args.share)


if __name__ == "__main__":
    main()

from collections.abc import Callable

from my_ai_assistant_online.application.evaluation_runner import run_evaluation
from my_ai_assistant_online.application.evaluation_task import AgentRunner


def run(agent_factory: Callable[[], AgentRunner]) -> None:
    agent = agent_factory()
    run_evaluation(agent)


def main() -> None:
    from my_ai_assistant_online.tools.app import build_agent

    run(build_agent)


if __name__ == "__main__":
    main()

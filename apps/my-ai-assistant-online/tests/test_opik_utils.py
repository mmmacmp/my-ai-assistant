from unittest.mock import MagicMock, call

import pytest

from my_ai_assistant_online import opik_utils
from my_ai_assistant_online.config import OpikSettings


def test_configure_opik_disables_tracing_when_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tracing_active = MagicMock()
    update_session_config = MagicMock()
    monkeypatch.setattr(opik_utils.opik, "set_tracing_active", set_tracing_active)
    monkeypatch.setattr(
        opik_utils.opik_config,
        "update_session_config",
        update_session_config,
    )

    opik_utils.configure_opik(OpikSettings(enabled=False))

    set_tracing_active.assert_called_once_with(False)
    update_session_config.assert_not_called()


def test_configure_opik_uses_explicit_cloud_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_tracing_active = MagicMock()
    update_session_config = MagicMock()
    monkeypatch.setattr(opik_utils.opik, "set_tracing_active", set_tracing_active)
    monkeypatch.setattr(
        opik_utils.opik_config,
        "update_session_config",
        update_session_config,
    )
    settings = OpikSettings(
        enabled=True,
        api_key="opik-test-secret",
        workspace="test-workspace",
        project_name="test-project",
    )

    opik_utils.configure_opik(settings)

    assert update_session_config.call_args_list == [
        call("api_key", "opik-test-secret"),
        call("workspace", "test-workspace"),
        call("project_name", "test-project"),
    ]
    set_tracing_active.assert_called_once_with(True)


def test_enabled_opik_requires_cloud_credentials() -> None:
    with pytest.raises(RuntimeError, match="OPIK_API_KEY"):
        opik_utils.configure_opik(OpikSettings(enabled=True))

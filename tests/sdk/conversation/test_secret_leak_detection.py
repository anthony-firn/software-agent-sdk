"""Tests for secret leak detection guard in SecretRegistry."""

import pytest
from pydantic import SecretStr

from openhands.sdk.conversation.secret_registry import SecretLeakError, SecretLeakInfo, SecretRegistry
from openhands.sdk.secret import SecretSource, StaticSecret


class TestSecretLeakDetection:
    """RED tests — check_for_leaks does not exist yet, these should all fail."""

    def test_check_for_leaks_no_leak_returns_empty(self):
        """Text without any secret values should return empty dict."""
        registry = SecretRegistry()
        registry.update_secrets({
            "API_KEY": "sk-abc123secret",
            "DB_PASSWORD": "s3cur3p@ss!",
        })

        result = registry.check_for_leaks("This is clean text with no secrets.")
        assert result == {}

    def test_check_for_leaks_detects_exact_value(self):
        """Exact secret value in text should be detected."""
        registry = SecretRegistry()
        registry.update_secrets({
            "GITHUB_TOKEN": "ghp_abc123superSecretToken",
        })

        leaks = registry.check_for_leaks(
            "Error: authentication failed with token ghp_abc123superSecretToken"
        )
        assert "GITHUB_TOKEN" in leaks
        assert leaks["GITHUB_TOKEN"].secret_name == "GITHUB_TOKEN"

    def test_check_for_leaks_detects_multiple_secrets(self):
        """Multiple different secret values in the same text should all be detected."""
        registry = SecretRegistry()
        registry.update_secrets({
            "API_KEY": "sk-key-one",
            "DB_PASSWORD": "db-pass-two",
        })

        leaks = registry.check_for_leaks(
            "Using API key sk-key-one and password db-pass-two for connection"
        )
        assert "API_KEY" in leaks
        assert "DB_PASSWORD" in leaks
        assert len(leaks) == 2

    def test_check_for_leaks_does_not_match_secret_names(self):
        """Secret key NAMES (not values) in text should NOT trigger leak detection."""
        registry = SecretRegistry()
        registry.update_secrets({
            "API_KEY": "sk-real-secret-value-123",
        })

        # The name $API_KEY is fine — it's how the agent references the secret
        leaks = registry.check_for_leaks(
            "Use $API_KEY to authenticate the request."
        )
        assert leaks == {}

    def test_check_for_leaks_detects_partial_match(self):
        """If a reasonable substring (>= 8 chars) of a secret appears, flag it."""
        registry = SecretRegistry()
        registry.update_secrets({
            "LONG_TOKEN": "ghp_thisIsAVeryLongGitHubTokenValue",
        })

        # A partial but substantial substring (>= 8 chars) should be caught
        leaks = registry.check_for_leaks(
            "The token prefix is ghp_thisIsAVeryLongGitHubTokenValue"
        )
        assert "LONG_TOKEN" in leaks

    def test_check_for_leaks_short_substrings_not_flagged(self):
        """Very short substrings (< 8 chars) should NOT flag to avoid false positives."""
        registry = SecretRegistry()
        registry.update_secrets({
            "API_KEY": "sk-abcdefghijklmnop",
        })

        # "sk-abc" alone (6 chars) should not trigger
        leaks = registry.check_for_leaks("The prefix is sk-abc")
        assert leaks == {}

    def test_check_for_leaks_empty_text(self):
        """Empty text should return no leaks."""
        registry = SecretRegistry()
        registry.update_secrets({"KEY": "value"})

        leaks = registry.check_for_leaks("")
        assert leaks == {}

    def test_check_for_leaks_empty_registry(self):
        """Registry with no secrets should always return empty."""
        registry = SecretRegistry()

        leaks = registry.check_for_leaks("sk-anything-here-doesnt-matter")
        assert leaks == {}

    def test_check_for_leaks_works_with_secret_source(self):
        """Leak detection should work with SecretSource (callable) values too."""
        registry = SecretRegistry()

        class DynamicToken(SecretSource):
            def get_value(self):
                return "dynamic-token-value-xyz"

        registry.update_secrets({
            "DYNAMIC_TOKEN": DynamicToken(),
        })

        leaks = registry.check_for_leaks(
            "Got error: dynamic-token-value-xyz is invalid"
        )
        assert "DYNAMIC_TOKEN" in leaks

    def test_check_for_leaks_exports_tracked_values(self):
        """Previously exported values (via get_secrets_as_env_vars) should be checked."""
        registry = SecretRegistry()
        registry.update_secrets({
            "GITHUB_TOKEN": "ghp_exportedTokenValue",
        })

        # Simulate the secret being exported (added to _exported_values)
        registry.get_secrets_as_env_vars("echo $GITHUB_TOKEN")

        leaks = registry.check_for_leaks(
            "Command output contained: ghp_exportedTokenValue"
        )
        assert "GITHUB_TOKEN" in leaks

    def test_check_for_leaks_context_snippet_redacted(self):
        """The returned leak info should contain a truncated+redacted context snippet."""
        registry = SecretRegistry()
        registry.update_secrets({
            "API_KEY": "sk-sensitive-value-here",
        })

        leaks = registry.check_for_leaks(
            "Some output containing sk-sensitive-value-here in the middle of text"
        )
        assert "API_KEY" in leaks
        info = leaks["API_KEY"]
        snippet = info.context_snippet
        # The snippet should be truncated and the secret value should be redacted
        assert "sk-sensitive-value-here" not in snippet
        assert "<redacted>" in snippet


class TestSecretLeakError:
    """Tests for the SecretLeakError exception."""

    def test_secret_leak_error_creation(self):
        """SecretLeakError should store leaked secret info."""
        from openhands.sdk.conversation.secret_registry import SecretLeakError

        error = SecretLeakError(
            leaked_secrets={
                "API_KEY": SecretLeakInfo(secret_name="API_KEY", context_snippet="...<redacted>..."),
                "DB_PASSWORD": SecretLeakInfo(secret_name="DB_PASSWORD", context_snippet="...<redacted>..."),
            }
        )
        assert "API_KEY" in str(error)
        assert "DB_PASSWORD" in str(error)
        assert error.leaked_secrets["API_KEY"].secret_name == "API_KEY"


class TestAgentIntegration:
    """Integration tests for the leak guard in the agent execution loop."""

    def test_observation_with_leaked_secret_raises(self):
        """When an observation contains a secret value, SecretLeakError is raised
        during _ActionBatch.emit()."""
        from openhands.sdk.agent.agent import _ActionBatch
        from openhands.sdk.event import ActionEvent, ObservationEvent
        from openhands.sdk.llm import MessageToolCall, TextContent
        from openhands.sdk.tool import Action
        from openhands.sdk.tool.builtins.finish import FinishObservation

        class FakeAction(Action):
            pass

        registry = SecretRegistry()
        registry.update_secrets({"TOKEN": "super-secret-leaked-value"})

        obs = FinishObservation.from_text(
            "Error: token super-secret-leaked-value is invalid"
        )
        obs_event = ObservationEvent(
            source="environment",
            tool_name="test_tool",
            tool_call_id="call_1",
            observation=obs,
            action_id="action_1",
        )

        ae = ActionEvent(
            id="action_1",
            source="agent",
            thought=[TextContent(text="test")],
            tool_name="test_tool",
            tool_call_id="call_1",
            tool_call=MessageToolCall(
                id="call_1", name="test_tool", arguments="{}", origin="completion"
            ),
            llm_response_id="resp_1",
            action=FakeAction(),
        )

        batch = _ActionBatch(
            action_events=[ae],
            has_finish=False,
            results_by_id={"action_1": [obs_event]},
            secret_registry=registry,
        )

        with pytest.raises(SecretLeakError) as exc_info:
            batch.emit(lambda event: None)

        assert "TOKEN" in exc_info.value.leaked_secrets

    def test_observation_without_leak_emits_normally(self):
        """Clean observation text should emit without error."""
        from openhands.sdk.agent.agent import _ActionBatch
        from openhands.sdk.event import ActionEvent, ObservationEvent
        from openhands.sdk.llm import MessageToolCall, TextContent
        from openhands.sdk.tool import Action
        from openhands.sdk.tool.builtins.finish import FinishObservation

        class FakeAction(Action):
            pass

        registry = SecretRegistry()
        registry.update_secrets({"TOKEN": "super-secret-leaked-value"})

        obs = FinishObservation.from_text("Everything is fine, no secrets here.")
        obs_event = ObservationEvent(
            source="environment",
            tool_name="test_tool",
            tool_call_id="call_1",
            observation=obs,
            action_id="action_1",
        )

        ae = ActionEvent(
            id="action_1",
            source="agent",
            thought=[TextContent(text="test")],
            tool_name="test_tool",
            tool_call_id="call_1",
            tool_call=MessageToolCall(
                id="call_1", name="test_tool", arguments="{}", origin="completion"
            ),
            llm_response_id="resp_1",
            action=FakeAction(),
        )

        emitted = []

        batch = _ActionBatch(
            action_events=[ae],
            has_finish=False,
            results_by_id={"action_1": [obs_event]},
            secret_registry=registry,
        )
        batch.emit(lambda event: emitted.append(event))

        assert len(emitted) == 1
        assert isinstance(emitted[0], ObservationEvent)

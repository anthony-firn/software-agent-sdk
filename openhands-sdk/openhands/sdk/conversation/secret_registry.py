"""Secrets manager for handling sensitive data in conversations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from openhands.sdk.logger import get_logger
from openhands.sdk.secret import SecretSource, SecretValue, StaticSecret
from openhands.sdk.utils.models import OpenHandsModel

if TYPE_CHECKING:
    from openhands.sdk.tool.schema import Observation


# Minimum length for a substring of a secret value to be considered a leak.
# Shorter substrings are ignored to avoid false positives from common words.
_MIN_LEAK_SUBSTRING_LENGTH = 8

# Maximum length of context snippet to include in leak reports.
_MAX_CONTEXT_SNIPPET_LENGTH = 80


@dataclass
class SecretLeakInfo:
    """Information about a detected secret leak."""

    secret_name: str
    context_snippet: str


class SecretLeakError(Exception):
    """Raised when a registered secret value is detected in content about to be
    sent to the LLM.

    This exception should trigger an immediate halt of the agent loop and
    notify the user to rotate the leaked secret.
    """

    def __init__(self, leaked_secrets: dict[str, SecretLeakInfo]):
        self.leaked_secrets = leaked_secrets
        names = ", ".join(leaked_secrets.keys())
        super().__init__(
            f"SECRET LEAK DETECTED: The following secrets leaked into agent "
            f"context: {names}. The agent has been halted to prevent these "
            f"values from being sent to the LLM. Rotate these secrets "
            f"immediately."
        )


logger = get_logger(__name__)


class SecretRegistry(OpenHandsModel):
    """Manages secrets and injects them into bash commands when needed.

    The secret registry stores a mapping of secret keys to SecretSources
    that retrieve the actual secret values. When a bash command is about to be
    executed, it scans the command for any secret keys and injects the corresponding
    environment variables.

    Secret sources will redact / encrypt their sensitive values as appropriate when
    serializing, depending on the content of the context. If a context is present
    and contains a 'cipher' object, this is used for encryption. If it contains a
    boolean 'expose_secrets' flag set to True, secrets are dunped in plain text.
    Otherwise secrets are redacted.

    Additionally, it tracks the latest exported values to enable consistent masking
    even when callable secrets fail on subsequent calls.
    """

    secret_sources: dict[str, SecretSource] = Field(default_factory=dict)
    _exported_values: dict[str, str] = PrivateAttr(default_factory=dict)

    def update_secrets(
        self,
        secrets: Mapping[str, SecretValue],
    ) -> None:
        """Add or update secrets in the manager.

        Args:
            secrets: Dictionary mapping secret keys to either string values
                    or callable functions that return string values
        """
        secret_sources = {name: _wrap_secret(value) for name, value in secrets.items()}
        self.secret_sources.update(secret_sources)

    def find_secrets_in_text(self, text: str) -> set[str]:
        """Find all secret keys mentioned in the given text.

        Args:
            text: The text to search for secret keys

        Returns:
            Set of secret keys found in the text
        """
        found_keys = set()
        for key in self.secret_sources.keys():
            if key.lower() in text.lower():
                found_keys.add(key)
        return found_keys

    def get_secrets_as_env_vars(self, command: str) -> dict[str, str]:
        """Get secrets that should be exported as environment variables for a command.

        Args:
            command: The bash command to check for secret references

        Returns:
            Dictionary of environment variables to export (key -> value)
        """
        found_secrets = self.find_secrets_in_text(command)

        if not found_secrets:
            return {}

        logger.debug(f"Found secrets in command: {found_secrets}")

        env_vars = {}
        for key in found_secrets:
            try:
                source = self.secret_sources[key]
                value = source.get_value()
                if value:
                    env_vars[key] = value
                    # Track successfully exported values for masking
                    self._exported_values[key] = value
            except Exception as e:
                logger.error(f"Failed to retrieve secret for key '{key}': {e}")
                continue

        logger.debug(f"Prepared {len(env_vars)} secrets as environment variables")
        return env_vars

    def mask_secrets_in_output(self, text: str) -> str:
        """Mask secret values in the given text.

        This method uses both the current exported values and attempts to get
        fresh values from callables to ensure comprehensive masking.

        Args:
            text: The text to mask secrets in

        Returns:
            Text with secret values replaced by <secret-hidden>
        """
        if not text:
            return text

        masked_text = text

        # First, mask using currently exported values (always available)
        for value in self._exported_values.values():
            masked_text = masked_text.replace(value, "<secret-hidden>")

        return masked_text

    def check_for_leaks(self, text: str) -> dict[str, SecretLeakInfo]:
        """Check if any registered secret VALUES appear in the given text.

        This is a safety guard that scans content about to be sent to the LLM
        for actual secret values. Unlike ``find_secrets_in_text`` which looks
        for secret KEY names (e.g. ``$API_KEY``), this checks for secret VALUES
        (e.g. the actual token ``ghp_abc123...``).

        The check covers:
        - All registered secret source values (retrieved on demand)
        - Previously exported values (tracked in ``_exported_values``)
        - Partial substring matches >= ``_MIN_LEAK_SUBSTRING_LENGTH`` chars

        Args:
            text: The text to scan for secret value leaks.

        Returns:
            Dict mapping leaked secret names to ``SecretLeakInfo``.
            Empty dict if no leaks are detected.
        """
        if not text or not self.secret_sources:
            return {}

        leaks: dict[str, SecretLeakInfo] = {}

        for name, source in self.secret_sources.items():
            try:
                value = source.get_value()
            except Exception:
                # If we can't retrieve the value, check exported values
                value = self._exported_values.get(name)

            if not value:
                continue

            # Check for the full value first (fast path)
            idx = text.find(value)
            if idx == -1:
                # Try partial substrings >= minimum length
                idx = self._find_substring_match(text, value)

            if idx >= 0:
                snippet = self._build_context_snippet(text, value, idx)
                leaks[name] = SecretLeakInfo(
                    secret_name=name,
                    context_snippet=snippet,
                )

        return leaks

    def _find_substring_match(self, text: str, value: str) -> int:
        """Find the position of any substantial substring of ``value`` in ``text``.

        Only substrings of length >= ``_MIN_LEAK_SUBSTRING_LENGTH`` are checked
        to avoid false positives from short, common character sequences.
        """
        if len(value) < _MIN_LEAK_SUBSTRING_LENGTH:
            return -1
        # Slide a window of MIN_LENGTH across the secret value
        for start in range(len(value) - _MIN_LEAK_SUBSTRING_LENGTH + 1):
            chunk = value[start : start + _MIN_LEAK_SUBSTRING_LENGTH]
            idx = text.find(chunk)
            if idx >= 0:
                return idx
        return -1

    def get_leaked_values(self, names: list[str]) -> list[str]:
        """Retrieve the raw values for the given secret names for masking.

        The values are also tracked in ``_exported_values`` so subsequent
        calls to ``mask_secrets_in_output`` will catch them too.

        Args:
            names: List of secret names whose values to retrieve.

        Returns:
            List of non-empty secret values.
        """
        values: list[str] = []
        for name in names:
            source = self.secret_sources.get(name)
            if source is None:
                continue
            try:
                value = source.get_value()
            except Exception:
                value = self._exported_values.get(name)
            if value:
                self._exported_values[name] = value
                values.append(value)
        return values

    @staticmethod
    def _build_context_snippet(text: str, secret_value: str, match_idx: int) -> str:
        """Build a truncated context snippet around the match, with the secret
        value redacted.
        """
        half = _MAX_CONTEXT_SNIPPET_LENGTH // 2
        start = max(0, match_idx - half)
        end = min(len(text), match_idx + len(secret_value) + half)
        snippet = text[start:end]
        # Redact the secret value from the snippet
        snippet = snippet.replace(secret_value, "<redacted>")
        # Add ellipsis if truncated
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"

    def get_secret_infos(self) -> list[dict[str, str | None]]:
        """Get secret information (name and description) for prompt inclusion.

        Returns:
            List of dictionaries with 'name' and 'description' keys.
            Returns an empty list if no secrets are registered.
            Description will be None if not available.
        """
        if not self.secret_sources:
            return []
        secret_infos = []
        for name, source in self.secret_sources.items():
            description = source.description
            secret_infos.append({"name": name, "description": description})
        return secret_infos

    def get_secret_value(self, name: str) -> str | None:
        """Look up a single secret value by name.

        This method retrieves the value of a specific secret. It's designed
        to be passed as a callback to functions that need secret lookup
        (e.g., expand_mcp_variables) without exposing all secrets at once.

        Retrieved values are tracked in _exported_values for consistent masking
        in command outputs.

        Args:
            name: The name of the secret to retrieve.

        Returns:
            The secret value if found and successfully retrieved, None otherwise.

        Note:
            Returns None for both missing secrets and retrieval failures.
            Retrieval errors (network, auth, etc.) are logged as warnings.
        """
        source = self.secret_sources.get(name)
        if source is None:
            return None
        try:
            value = source.get_value()
            if value:
                # Track retrieved value for output masking
                self._exported_values[name] = value
            return value
        except (OSError, TimeoutError) as e:
            # Network/IO errors - likely transient, log and return None
            logger.warning(
                f"Transient error retrieving secret '{name}' "
                f"(may retry later): {type(e).__name__}: {e}"
            )
            return None
        except (ValueError, KeyError, TypeError) as e:
            # Configuration/data errors - likely permanent
            logger.warning(
                f"Configuration error for secret '{name}': {type(e).__name__}: {e}"
            )
            return None
        except Exception as e:
            # Unexpected errors - log with full details for debugging
            logger.warning(
                f"Unexpected error retrieving secret '{name}': {type(e).__name__}: {e}"
            )
            return None

    def check_action_for_egress(self, action: BaseModel) -> dict[str, SecretLeakInfo]:
        """Check whether a tool action's serialized parameters contain secret values.

        Serializes the action to JSON and runs ``check_for_leaks`` on the result.
        Use this before executing a tool to prevent secrets from being sent out
        via network requests, command arguments, environment variables, etc.

        Args:
            action: The Action object about to be executed.

        Returns:
            Dict of leaked secret names to ``SecretLeakInfo`` (empty if clean).
        """
        # model_dump_json exposes all field values, including any secrets the
        # LLM may have injected into tool parameters (prompt injection).
        serialized = action.model_dump_json()
        return self.check_for_leaks(serialized)

    @staticmethod
    def create_egress_blocked_observation(
        leaked_names: list[str],
    ) -> "Observation":
        """Create an error observation for a blocked egress attempt.

        The observation text names the blocked secrets but does NOT contain
        the actual secret values.

        Args:
            leaked_names: Names of the secrets that were detected.

        Returns:
            An error ``Observation`` indicating egress was blocked.
        """
        from openhands.sdk.tool.builtins.finish import FinishObservation

        names_str = ", ".join(sorted(leaked_names))
        return FinishObservation.from_text(
            (
                f"🚫 EGRESS BLOCKED: Tool execution was prevented because "
                f"the action parameters contained the following registered "
                f"secret names: {names_str}. The agent should use secret "
                f"references exactly as configured (e.g. $API_KEY) — never "
                f"pass raw secret values to any tool."
            ),
            is_error=True,
        )


def _wrap_secret(value: SecretValue) -> SecretSource:
    """Convert the value given to a secret source"""
    if isinstance(value, SecretSource):
        return value
    if isinstance(value, str):
        return StaticSecret(value=SecretStr(value))
    raise ValueError("Invalid SecretValue")

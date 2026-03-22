"""Autopilot session manager using polling through the gateway proxy.

Manages the lifecycle of an autopilot session:
  - Create session by posting initial message
  - Poll for status changes
  - Auto-approve tool calls
  - Delegate user questions to an interlocutor
  - Fetch config-writes when session is idle
  - Interrupt session if turn limit reached
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Nil UUID used to create a new session
NIL_UUID = "00000000-0000-0000-0000-000000000000"


@dataclass
class AutopilotSessionResult:
    """Result of an autopilot session."""

    session_id: str
    final_status: str
    turns: int = 0
    config_writes: list[dict[str, Any]] = field(default_factory=list)
    raw_config_writes: list[Any] = field(default_factory=list)
    error: Optional[str] = None


class AutopilotSessionManager:
    """Manages an autopilot session via the gateway's proxy endpoints."""

    _BACKGROUND_STATUSES = frozenset(
        {
            "server_side_processing",
            "waiting_for_tool_execution",
        }
    )

    def __init__(
        self,
        gateway_url: str,
        interlocutor: Any = None,
        max_turns: int = 30,
        timeout: float = 600.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.base = f"{self.gateway_url}/internal/autopilot/v1"
        self.interlocutor = interlocutor
        self.max_turns = max_turns
        self.timeout = timeout

    # Message sent when autopilot goes idle without writing config changes.
    _IDLE_NUDGE = (
        "You haven't written any config changes yet. "
        "Please write new variants with improved prompts and set up an "
        'experimentation config using the "track_and_stop" type '
        "so we can A/B test the new variants against the baseline. "
        "Use the write_config tool to make the changes. "
        "IMPORTANT: (1) Include the baseline variant (e.g. 'initial') in "
        "candidate_variants alongside new variants so it participates in the "
        "experiment — a variant cannot appear in both candidate_variants and "
        "fallback_variants. (2) Do NOT set 'weight' on individual variants "
        "when using an experimentation section."
    )

    async def run_session(
        self,
        initial_message: str,
    ) -> AutopilotSessionResult:
        """Run a full autopilot session until config writes are produced.

        The session keeps going when autopilot goes idle without having
        written config changes — we nudge it to keep working via the
        interlocutor or a default follow-up message.  The session stops
        when:
          - Config writes have been produced and autopilot is idle
          - The turn limit is reached
          - A terminal failure/error occurs

        Args:
            initial_message: The initial message to send to autopilot.

        Returns:
            AutopilotSessionResult with session ID, status, and config writes.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Create session by posting to the nil-UUID endpoint
            session_id = await self._create_session(client, initial_message)
            logger.info("Created autopilot session: %s", session_id)

            turns = 0
            final_status = "unknown"

            try:
                while turns < self.max_turns:
                    # Poll until actionable status
                    status, session_data = await self._poll_until_actionable(
                        client, session_id
                    )
                    turns += 1

                    logger.info(
                        "Session %s turn %d: status=%s",
                        session_id,
                        turns,
                        status,
                    )

                    if status == "idle":
                        # Check if autopilot has produced config writes yet
                        config_writes, raw_config_writes = (
                            await self._fetch_config_writes_response(client, session_id)
                        )
                        has_variants, has_routing = self._check_config_writes(
                            config_writes
                        )

                        if has_variants and has_routing:
                            logger.info(
                                "Session %s is idle with %d config writes "
                                "(variants=%s, routing=%s) — done",
                                session_id,
                                len(config_writes),
                                has_variants,
                                has_routing,
                            )
                            final_status = "idle"
                            return AutopilotSessionResult(
                                session_id=session_id,
                                final_status=final_status,
                                turns=turns,
                                config_writes=config_writes,
                                raw_config_writes=raw_config_writes,
                            )

                        # Missing pieces — nudge autopilot to keep going
                        missing = []
                        if not has_variants:
                            missing.append("new variants")
                        if not has_routing:
                            missing.append("an experimentation/routing config")
                        logger.info(
                            "Session %s is idle but still missing: %s — nudging",
                            session_id,
                            ", ".join(missing),
                        )
                        await self._send_nudge(
                            client,
                            session_id,
                            has_variants=has_variants,
                            has_routing=has_routing,
                        )

                    elif status == "waiting_for_tool_call_authorization":
                        await self._approve_all(client, session_id, session_data)
                    elif status == "user_questions":
                        await self._handle_user_questions(
                            client, session_id, session_data
                        )
                    elif status in {"failed", "error"}:
                        final_status = status
                        break
                    else:
                        logger.warning("Unexpected status: %s", status)
                        final_status = status
                        break
                else:
                    # Max turns reached
                    logger.warning(
                        "Session %s hit max turns (%d), interrupting",
                        session_id,
                        self.max_turns,
                    )
                    await self._interrupt(client, session_id)
                    final_status = "interrupted"

                # Fetch config writes (for non-idle exits like interrupted/error)
                config_writes, raw_config_writes = (
                    await self._fetch_config_writes_response(client, session_id)
                )

                return AutopilotSessionResult(
                    session_id=session_id,
                    final_status=final_status,
                    turns=turns,
                    config_writes=config_writes,
                    raw_config_writes=raw_config_writes,
                )

            except Exception as e:
                logger.error("Session %s failed: %s", session_id, e, exc_info=True)
                # Try to salvage config writes even on error
                try:
                    config_writes, raw_config_writes = (
                        await self._fetch_config_writes_response(client, session_id)
                    )
                except Exception:
                    config_writes = []
                    raw_config_writes = []
                return AutopilotSessionResult(
                    session_id=session_id,
                    final_status="error",
                    turns=turns,
                    config_writes=config_writes,
                    raw_config_writes=raw_config_writes,
                    error=str(e),
                )

    async def _create_session(self, client: httpx.AsyncClient, message: str) -> str:
        """Create a new autopilot session by posting the initial message."""
        url = f"{self.base}/sessions/{NIL_UUID}/events"
        payload = {
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": message}],
            }
        }
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            logger.error(
                "Session creation failed (status %d): %s",
                resp.status_code,
                resp.text,
            )
        resp.raise_for_status()
        data = resp.json()
        return data["session_id"]

    async def _poll_until_actionable(
        self, client: httpx.AsyncClient, session_id: str
    ) -> tuple[str, dict]:
        """Poll the session status until it reaches a state that requires action.

        Returns a tuple of (status_string, full_session_data).
        """
        poll_url = f"{self.base}/sessions/{session_id}/events"
        poll_interval = 2.0  # seconds between polls
        max_polls = int(self.timeout / poll_interval)
        last_status = ""

        for i in range(max_polls):
            try:
                resp = await client.get(poll_url, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", {}).get("status", "")
                last_status = status
                # Treat any non-background status as actionable/terminal. This is
                # more robust than maintaining a positive allowlist because newly
                # added terminal statuses (for example `failed`) should stop the
                # session immediately instead of polling forever.
                if status not in self._BACKGROUND_STATUSES:
                    self._log_session_events(session_id, data)
                    return status, data

                if i % 15 == 0:  # Log every ~30 seconds
                    logger.debug(
                        "Session %s polling: status=%s (poll %d)",
                        session_id,
                        status,
                        i,
                    )

            except Exception as e:
                logger.warning("Poll failed for session %s: %s", session_id, e)

            await asyncio.sleep(poll_interval)

        logger.warning(
            "Session %s: polling timed out after %ds while status=%s",
            session_id,
            self.timeout,
            last_status or "unknown",
        )
        raise TimeoutError(
            "session polling timed out while still in background status "
            f"{last_status or 'unknown'}"
        )

    def _log_session_events(self, session_id: str, session_data: dict) -> None:
        """Log a summary of session events for debugging."""
        events = session_data.get("events", [])
        for event in events:
            payload = event.get("payload", {})
            evt_type = payload.get("type", "unknown")
            evt_id = event.get("id", "?")

            if evt_type == "message":
                role = payload.get("role", "?")
                content = payload.get("content", [])
                text = ""
                if isinstance(content, list):
                    text = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and "text" in block
                    )
                elif isinstance(content, str):
                    text = content
                # Log full message (truncate only if very long)
                preview = text[:2000] + ("..." if len(text) > 2000 else "")
                logger.info(
                    "[session %s] event %s: message role=%s: %s",
                    session_id[-12:],
                    evt_id[-12:],
                    role,
                    preview,
                )

            elif evt_type == "tool_call":
                name = payload.get("name", "?")
                args = payload.get("arguments", {})
                args_preview = json.dumps(args)[:500]
                logger.info(
                    "[session %s] event %s: tool_call name=%s args=%s",
                    session_id[-12:],
                    evt_id[-12:],
                    name,
                    args_preview,
                )

            elif evt_type == "tool_result":
                name = payload.get("name", "?")
                result = payload.get("result", "")
                result_str = (
                    json.dumps(result) if not isinstance(result, str) else result
                )
                preview = result_str[:2000] + ("..." if len(result_str) > 2000 else "")
                logger.info(
                    "[session %s] event %s: tool_result name=%s result=%s",
                    session_id[-12:],
                    evt_id[-12:],
                    name,
                    preview,
                )

            else:
                logger.info(
                    "[session %s] event %s: type=%s",
                    session_id[-12:],
                    evt_id[-12:],
                    evt_type,
                )

    async def _approve_all(
        self, client: httpx.AsyncClient, session_id: str, session_data: dict
    ) -> None:
        """Auto-approve all pending tool calls."""
        # Extract the last pending tool call event ID
        pending = session_data.get("pending_tool_calls", [])
        if not pending:
            logger.warning("No pending tool calls found to approve")
            return

        # Log what we're approving
        for tc in pending:
            tc_payload = tc.get("payload", {})
            tc_name = tc_payload.get("name", "?")
            tc_args = tc_payload.get("arguments", {})
            args_preview = json.dumps(tc_args)[:300]
            logger.info(
                "Approving tool call: %s (args=%s)",
                tc_name,
                args_preview,
            )

        last_tool_call_id = pending[-1].get("id", "")

        url = f"{self.base}/sessions/{session_id}/actions/approve_all"
        resp = await client.post(
            url,
            json={"last_tool_call_event_id": last_tool_call_id},
        )
        if resp.status_code != 200:
            logger.error(
                "approve_all failed (status %d): %s",
                resp.status_code,
                resp.text,
            )
        resp.raise_for_status()
        logger.debug(
            "Approved all tool calls for session %s (last_id=%s)",
            session_id,
            last_tool_call_id,
        )

    async def _handle_user_questions(
        self, client: httpx.AsyncClient, session_id: str, session_data: dict
    ) -> None:
        """Handle user questions by delegating to the interlocutor."""
        # Extract questions from events in session_data
        events = session_data.get("events", [])
        questions_text = ""
        for event in reversed(events):
            payload = event.get("payload", {})
            if payload.get("type") == "user_questions":
                # Extract text from questions content
                content = payload.get("content", [])
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(block["text"])
                    questions_text = "\n".join(parts)
                elif isinstance(content, str):
                    questions_text = content
                break

        if not questions_text:
            logger.warning("No questions text found in session data")
            await self._send_message(
                client,
                session_id,
                "Please proceed with your best judgment.",
            )
            return

        if self.interlocutor is None:
            logger.warning("No interlocutor configured, sending default response")
            await self._send_message(
                client,
                session_id,
                "Please proceed with your best judgment.",
            )
            return

        # Get answer from interlocutor
        answer = await self.interlocutor.answer(questions_text)
        await self._send_message(client, session_id, answer)

    @staticmethod
    def _check_config_writes(
        config_writes: list[dict[str, Any]],
    ) -> tuple[bool, bool]:
        """Check whether config writes include new variants and routing.

        Returns (has_variants, has_routing).
        """
        has_variants = False
        has_routing = False
        for edit in config_writes:
            op = edit.get("operation", "")
            # Config writes may arrive either in canonical EditPayload form
            # (`operation=...`) or in simplified tool output form
            # (`{"variant": ...}`, `{"experimentation": ...}`).
            # TODO: this could be tightened but it works
            if "variant" in op or "variant" in edit or "variants" in edit:
                has_variants = True
            if (
                "rout" in op
                or "experiment" in op
                or "experimentation" in edit
                or "routing" in edit
                or "weights" in edit
            ):
                has_routing = True
        return has_variants, has_routing

    async def _send_nudge(
        self,
        client: httpx.AsyncClient,
        session_id: str,
        has_variants: bool = False,
        has_routing: bool = False,
    ) -> None:
        """Send a follow-up message nudging autopilot to write config changes."""
        # Build a specific nudge based on what's missing
        parts = []
        if not has_variants:
            parts.append(
                "write new variant(s) with improved prompts using the write_config tool"
            )
        if not has_routing:
            parts.append(
                "set up an experimentation config using the "
                '"track_and_stop" type so the new variants are '
                "A/B tested against the baseline"
            )
        action = " and ".join(parts)

        nudge_prompt = (
            f"The AI optimization system has analyzed the data but hasn't "
            f"completed its changes yet. It still needs to: {action}. "
            f"Tell it to do this now. Be direct and specific. "
            f"IMPORTANT RULES: "
            f"(1) Include the baseline variant (e.g. 'initial') in "
            f"candidate_variants alongside new variants so it participates "
            f"in the experiment. A variant cannot be in both candidate_variants "
            f"and fallback_variants. "
            f"(2) Do NOT set 'weight' on individual variants when using "
            f"an experimentation section — use one or the other, not both."
        )

        if self.interlocutor is not None:
            answer = await self.interlocutor.answer(nudge_prompt)
            logger.info("Interlocutor nudge: %s", answer[:500])
            await self._send_message(client, session_id, answer)
        else:
            fallback = (
                f"You still need to: {action}. "
                f"Please use the write_config tool to make these changes now."
            )
            logger.info("Sending default nudge: %s", fallback)
            await self._send_message(client, session_id, fallback)

    async def _send_message(
        self, client: httpx.AsyncClient, session_id: str, content: str
    ) -> None:
        """Send a user message to the session."""
        url = f"{self.base}/sessions/{session_id}/events"
        payload = {
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": content}],
            }
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

    async def _interrupt(self, client: httpx.AsyncClient, session_id: str) -> None:
        """Interrupt the session."""
        url = f"{self.base}/sessions/{session_id}/actions/interrupt"
        resp = await client.post(url)
        resp.raise_for_status()
        logger.info("Interrupted session %s", session_id)

    async def _fetch_config_writes(
        self, client: httpx.AsyncClient, session_id: str
    ) -> list[dict[str, Any]]:
        """Fetch and flatten config writes from the completed session."""
        edits, _ = await self._fetch_config_writes_response(client, session_id)
        return edits

    async def _fetch_config_writes_response(
        self, client: httpx.AsyncClient, session_id: str
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Fetch config writes from the completed session.

        The API returns event objects like:
          {"id": "...", "payload": {"type": "tool_call", "name": "write_config",
           "arguments": {"edit": [...]}}}

        We extract the edit arrays and flatten them into a single list
        of EditPayload dicts suitable for config-applier-cli.
        """
        url = f"{self.base}/sessions/{session_id}/config-writes"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        # API may return {"config_writes": [...]} or just [...]
        if isinstance(data, dict):
            raw_writes = data.get("config_writes", [])
        else:
            raw_writes = data

        # Extract edit payloads from event wrappers
        edits: list[dict[str, Any]] = []
        for write in raw_writes:
            if isinstance(write, dict) and "payload" in write:
                # Event wrapper format — extract edits from arguments
                payload = write.get("payload", {})
                arguments = payload.get("arguments", {})
                edit_list = arguments.get("edit", [])
                if isinstance(edit_list, list):
                    edits.extend(edit_list)
                else:
                    logger.warning(
                        "Unexpected edit format in config write: %s",
                        type(edit_list),
                    )
            elif isinstance(write, dict) and "operation" in write:
                # Already in EditPayload format
                edits.append(write)
            else:
                logger.warning(
                    "Unknown config write format: %s",
                    json.dumps(write)[:200],
                )

        logger.info(
            "Session %s produced %d config write events with %d edits",
            session_id,
            len(raw_writes),
            len(edits),
        )
        logger.debug("Edits: %s", json.dumps(edits)[:1000])
        return edits, raw_writes

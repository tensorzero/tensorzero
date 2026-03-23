"""LLM interlocutor for answering autopilot's questions.

Uses an embedded TensorZero gateway with a dedicated config to generate
answers to questions the autopilot asks during a session.
"""

import logging
import os
from pathlib import Path

from tensorzero import AsyncTensorZeroGateway

logger = logging.getLogger(__name__)


class Interlocutor:
    """Answers autopilot questions using an embedded T0 gateway.

    The gateway runs in dryrun mode (no Postgres needed).
    """

    def __init__(self, gateway: AsyncTensorZeroGateway):
        self._gateway = gateway

    @classmethod
    async def create(cls, config_file: str) -> "Interlocutor":
        """Create an interlocutor with an embedded gateway.

        Args:
            config_file: Path to the T0 config for the interlocutor function.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Interlocutor config not found: {config_file}")

        # Temporarily unset autopilot env vars so the embedded gateway
        # doesn't try to initialize an autopilot client (which requires Postgres).
        saved = {}
        for key in list(os.environ):
            if key.startswith("TENSORZERO_AUTOPILOT"):
                saved[key] = os.environ.pop(key)
        try:
            gateway = await AsyncTensorZeroGateway.build_embedded(
                config_file=config_file,
            )
        finally:
            os.environ.update(saved)
        logger.info("Interlocutor gateway initialized from %s", config_file)
        return cls(gateway)

    async def answer(self, question: str) -> str:
        """Generate an answer to a question from autopilot.

        Args:
            question: The question text from autopilot.

        Returns:
            The answer text.
        """
        response = await self._gateway.inference(
            function_name="interlocutor",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                ]
            },
            dryrun=True,
        )

        # Extract text from response
        content = response.content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "\n".join(parts)
        return str(content)

    async def close(self) -> None:
        """Close the embedded gateway."""
        await self._gateway.close()

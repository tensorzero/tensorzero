from typing import Optional

from pydantic import BaseModel


class ObservabilityConfig(BaseModel):
    """
    Configuration for observability.
    """

    async_writes: bool = True
    enabled: Optional[bool] = None


class GatewayConfig(BaseModel):
    """
    Configuration for the gateway.
    """

    observability: ObservabilityConfig
    bind_address: Optional[str] = None

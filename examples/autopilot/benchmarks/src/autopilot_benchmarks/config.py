"""Pydantic configuration models for the eval harness."""

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class AutopilotTarget(BaseModel):
    """Configuration for the autopilot service to connect to."""

    kind: Literal["staging", "prod", "local"] = "prod"
    base_url: Optional[str] = None
    api_key_env: str = "TENSORZERO_AUTOPILOT_API_KEY"

    @property
    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise ValueError(f"Environment variable {self.api_key_env} is not set")
        return value


class InterlocutorConfig(BaseModel):
    """Configuration for the interlocutor LLM (answers autopilot's questions)."""

    config_file: str = "interlocutor_config/tensorzero.toml"


class InfraConfig(BaseModel):
    """Infrastructure configuration."""

    gateway_binary_path: str = "/usr/local/bin/gateway"
    gateway_port: int = 3000
    test_gateway_port: int = 3001
    gateway_startup_timeout: float = 30.0
    gateway_shutdown_timeout: float = 10.0


class EnvironmentConfig(BaseModel):
    """Configuration for a single evaluation environment.

    ``name`` is the canonical identifier for this environment — used for output
    directories, database records, snapshot paths, and S3 keys.  For most envs
    it matches the llmgym environment name (e.g. ``ner_conllpp_v0``).

    Harbor environments are parameterised by *dataset* (e.g.
    ``terminal-bench@2.0``, ``lawbench@1.0``) but share the same llmgym env
    ``harbor_v0``.  Set ``llmgym_env`` to ``"harbor_v0"`` for these so that
    ``name`` can be the dataset-specific identifier while llmgym still receives
    the correct environment name.
    """

    name: str
    llmgym_env: Optional[str] = None
    function_name: str
    metric_name: str
    initial_model: str = "openai::gpt-4o-mini"
    num_iterations: int = 1
    episodes_per_iteration: int = 100
    test_episodes_per_iteration: Optional[int] = None
    episode_concurrency: int = 10
    episode_timeout: Optional[float] = None
    test_episode_concurrency: Optional[int] = None
    test_episode_timeout: Optional[float] = None
    autopilot_initial_message: str = "Improve the performance of this function."
    autopilot_max_turns: int = 30
    autopilot_session_timeout: float = 3600.0
    task_split: str = "train"
    env_config: dict[str, Any] = Field(default_factory=dict)
    function_description: Optional[str] = None
    available_models: list[str] = Field(default_factory=list)
    seed: Optional[int] = None

    @property
    def effective_llmgym_env(self) -> str:
        """The llmgym environment name to pass to ``llmgym.make()``."""
        return self.llmgym_env or self.name


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""

    autopilot_target: AutopilotTarget = Field(default_factory=AutopilotTarget)
    interlocutor: InterlocutorConfig = Field(default_factory=InterlocutorConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    environments: list[EnvironmentConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

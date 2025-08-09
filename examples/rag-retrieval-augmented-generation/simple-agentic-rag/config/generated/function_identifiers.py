from typing import Literal

FunctionIdentifier = Literal[
    "tensorzero::function_name::multi_hop_rag_agent",
    "tensorzero::function_name::multi_hop_rag_agent::variant_name::baseline",
    "tensorzero::function_name::multi_hop_rag_agent_openai_v1",
    "tensorzero::function_name::multi_hop_rag_agent_openai_v1::variant_name::baseline",
    "tensorzero::function_name::multi_hop_rag_agent_openai_v1::variant_name::o3",
]

from enum import Enum

class FunctionIdentifierEnum(str, Enum):
    MULTI_HOP_RAG_AGENT = "tensorzero::function_name::multi_hop_rag_agent"
    MULTI_HOP_RAG_AGENT_BASELINE = "tensorzero::function_name::multi_hop_rag_agent::variant_name::baseline"
    MULTI_HOP_RAG_AGENT_OPENAI_V1 = "tensorzero::function_name::multi_hop_rag_agent_openai_v1"
    MULTI_HOP_RAG_AGENT_OPENAI_V1_BASELINE = "tensorzero::function_name::multi_hop_rag_agent_openai_v1::variant_name::baseline"
    MULTI_HOP_RAG_AGENT_OPENAI_V1_O3 = "tensorzero::function_name::multi_hop_rag_agent_openai_v1::variant_name::o3"

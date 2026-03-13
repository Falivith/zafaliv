from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    trust_remote_code: bool = False
    requires_auth: bool = False
    default_max_length: int = 1024
    notes: str = ""


MODEL_SPECS = {
    "qwen2.5-1.5b": ModelSpec(
        alias="qwen2.5-1.5b",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        default_max_length=1024,
        notes="Modelo pequeno, forte e com bom suporte multilíngue para validar ganhos.",
    ),
    "qwen2.5-3b": ModelSpec(
        alias="qwen2.5-3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        default_max_length=1024,
        notes="Melhor equilibrio entre tamanho e desempenho para 16 GB.",
    ),
    "gemma-3-1b": ModelSpec(
        alias="gemma-3-1b",
        model_id="google/gemma-3-1b-it",
        requires_auth=True,
        default_max_length=1024,
        notes="Modelo bem compacto da familia Gemma 3, bom para testar ganho relativo.",
    ),
    "gemma-3-4b": ModelSpec(
        alias="gemma-3-4b",
        model_id="google/gemma-3-4b-it",
        requires_auth=True,
        default_max_length=1024,
        notes="Opcao forte, mas exige aceite de licenca no Hugging Face.",
    ),
    "phi-3.5-mini": ModelSpec(
        alias="phi-3.5-mini",
        model_id="microsoft/Phi-3.5-mini-instruct",
        trust_remote_code=True,
        default_max_length=1024,
        notes="Modelo compacto e competitivo para raciocinio.",
    ),
    "llama-3.2-3b": ModelSpec(
        alias="llama-3.2-3b",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        requires_auth=True,
        default_max_length=1024,
        notes="Baseline forte, tambem exige aceite de licenca.",
    ),
    "amadeus-verbo-1.5b": ModelSpec(
        alias="amadeus-verbo-1.5b",
        model_id="amadeusai/Amadeus-Verbo-BI-Qwen-2.5-1.5B-PT-BR-Instruct-Experimental",
        default_max_length=1024,
        notes="Modelo ajustado em PT-BR, muito util como baseline linguistico.",
    ),
}

DEFAULT_MODEL_ALIASES = tuple(MODEL_SPECS.keys())


def resolve_model_spec(model_name: str) -> ModelSpec:
    if model_name in MODEL_SPECS:
        return MODEL_SPECS[model_name]

    return ModelSpec(
        alias=model_name.replace("/", "__"),
        model_id=model_name,
    )

"""Provider catalog and resolution.

The catalog is the single source of truth mapping model ids to providers and
listing each provider's "smart" and "less-smart" model. Values come from
settings so renaming a model is a config-only change.
"""

from enum import Enum
from typing import Dict

from ...config import settings


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


def _build_catalog() -> Dict[str, Dict[str, object]]:
    """Build the model catalog from settings. Keys are model ids (lowercased)."""
    return {
        # OpenAI
        settings.OPENAI_SMART_MODEL.lower(): {
            "provider": Provider.OPENAI,
            "tier": "smart",
        },
        settings.OPENAI_LESS_SMART_MODEL.lower(): {
            "provider": Provider.OPENAI,
            "tier": "less_smart",
        },
        # Anthropic
        settings.ANTHROPIC_SMART_MODEL.lower(): {
            "provider": Provider.ANTHROPIC,
            "tier": "smart",
        },
        settings.ANTHROPIC_LESS_SMART_MODEL.lower(): {
            "provider": Provider.ANTHROPIC,
            "tier": "less_smart",
        },
        # Ollama: register every catalogued model. User-supplied ids are also
        # resolved to ollama at runtime via the provider hint in the request.
        **{
            model_id.lower(): {"provider": Provider.OLLAMA, "tier": "less_smart"}
            for model_id in settings.OLLAMA_MODELS
        },
    }


CATALOG = _build_catalog()


def resolve_provider(model_id: str, hint: str | None = None) -> Provider:
    """Resolve which provider a model id belongs to.

    Resolution order:
      1. Explicit provider hint (e.g. request body) — used for Ollama, where
         the model id is user-supplied and not in the catalog.
      2. Catalog lookup by model id.
      3. Prefix heuristics (claude-* → anthropic, sk-ant handled by key).
      4. Default to OpenAI.
    """
    if hint:
        hint_lower = hint.lower()
        for p in Provider:
            if hint_lower == p.value:
                return Provider(hint_lower)

    if model_id:
        key = model_id.lower()
        if key in CATALOG:
            return CATALOG[key]["provider"]  # type: ignore[index]
        # Heuristic: Claude model ids
        if key.startswith("claude"):
            return Provider.ANTHROPIC

    return Provider.OPENAI


def get_base_url(provider: Provider) -> str | None:
    """Return the API base url for a provider, if one is needed.

    OpenAI and Anthropic use their SDK defaults (None). Ollama Cloud uses an
    OpenAI-compatible endpoint via base_url.
    """
    if provider == Provider.OLLAMA:
        return settings.OLLAMA_CLOUD_BASE_URL
    return None


def pick_smart_model(provider: Provider) -> str:
    """Return the smart model id for a provider."""
    if provider == Provider.ANTHROPIC:
        return settings.ANTHROPIC_SMART_MODEL
    if provider == Provider.OLLAMA:
        return settings.OLLAMA_MODELS[0]
    return settings.OPENAI_SMART_MODEL


def pick_helper_model(provider: Provider) -> str:
    """Return the less-smart (cheap) model id for background helper calls
    (query classification, decomposition, title generation, quality scoring).

    For Ollama there is only one user-picked model, so reuse it.
    """
    if provider == Provider.ANTHROPIC:
        return settings.ANTHROPIC_LESS_SMART_MODEL
    if provider == Provider.OLLAMA:
        return settings.OLLAMA_MODELS[0]
    return settings.OPENAI_LESS_SMART_MODEL

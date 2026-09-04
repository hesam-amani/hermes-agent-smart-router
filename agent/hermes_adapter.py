"""Hermes integration adapter for the smart router.

The adapter is deliberately thin: Hermes remains the owner of provider
resolution, credentials, clients, retry, and fallback execution. This module
only translates Hermes' existing model metadata into router candidates and
returns a ranked decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .smart_router import (
    CostPolicy,
    ModelCandidate,
    RoutingDecision,
    RoutingRequest,
    SmartRouter,
)


def candidate_from_hermes_metadata(
    provider: str,
    model: str,
    metadata: Any,
    *,
    free_models: set[tuple[str, str]] | None = None,
) -> ModelCandidate:
    """Convert a Hermes ``ModelInfo``-like object into a router candidate.

    ``free_models`` is an explicit entitlement set. Pricing metadata is useful
    for ranking/diagnostics, but is intentionally NOT sufficient to declare a
    model free: FREE_ONLY must never spend because metadata was incomplete.
    """
    free_keys = {
        (str(p).strip().lower(), str(m).strip().lower())
        for p, m in (free_models or set())
    }
    key = (provider.strip().lower(), model.strip().lower())

    if metadata is None:
        return ModelCandidate(
            provider=provider,
            model=model,
            free=key in free_keys,
        )

    def value(name: str, default: Any = None) -> Any:
        if isinstance(metadata, Mapping):
            return metadata.get(name, default)
        return getattr(metadata, name, default)

    context = value("context_window", 0) or 0
    try:
        context = int(context)
    except (TypeError, ValueError):
        context = 0

    status = str(value("status", "") or "").lower()
    return ModelCandidate(
        provider=provider,
        model=model,
        free=key in free_keys,
        context_length=context or None,
        supports_tools=bool(value("tool_call", False)),
        supports_vision=bool(
            value("attachment", False)
            or "image" in tuple(value("input_modalities", ()) or ())
        ),
        supports_reasoning=bool(value("reasoning", False)),
        # models.dev does not expose a universal quality score. Keep the
        # neutral baseline until the router's learned/per-provider quality
        # layer is added rather than pretending price == quality.
        quality=float(value("quality", 0.5) or 0.5),
        coding=float(value("coding", 0.5) or 0.5),
        research=float(value("research", 0.5) or 0.5),
        extra={
            "family": value("family", ""),
            "release_date": value("release_date", ""),
            "status": status,
            "cost_input": value("cost_input", 0.0),
            "cost_output": value("cost_output", 0.0),
        },
    )


def build_candidates(
    provider_models: Mapping[str, Iterable[str]],
    *,
    metadata_lookup: Any,
    free_models: set[tuple[str, str]] | None = None,
) -> list[ModelCandidate]:
    """Build candidates from Hermes' provider/model discovery results.

    ``metadata_lookup(provider, model)`` should normally delegate to
    ``agent.models_dev.get_model_info``. Keeping it injectable makes the
    integration cheap to test and avoids coupling this package to Hermes'
    import graph.
    """
    candidates: list[ModelCandidate] = []
    for provider, models in provider_models.items():
        for model in models:
            metadata = metadata_lookup(provider, model)
            candidates.append(
                candidate_from_hermes_metadata(
                    provider,
                    model,
                    metadata,
                    free_models=free_models,
                )
            )
    return candidates


def route_hermes_turn(
    router: SmartRouter,
    *,
    prompt: str,
    provider_models: Mapping[str, Iterable[str]],
    metadata_lookup: Any,
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY,
    requires_tools: bool = False,
    requires_vision: bool = False,
    requires_reasoning: bool = False,
    min_context_length: int | None = None,
    free_models: set[tuple[str, str]] | None = None,
) -> RoutingDecision:
    """Return the best route for one Hermes turn.

    This function does not mutate an ``AIAgent``. The actual model/provider
    activation belongs to Hermes' existing resolution/fallback machinery.
    That separation lets the integration hook fail open and keeps credential
    handling in one place.
    """
    candidates = build_candidates(
        provider_models,
        metadata_lookup=metadata_lookup,
        free_models=free_models,
    )
    request = RoutingRequest(
        prompt=prompt,
        cost_policy=cost_policy,
        requires_tools=requires_tools,
        requires_vision=requires_vision,
        requires_reasoning=requires_reasoning,
        min_context_length=min_context_length,
    )
    return router.route(request, candidates)


__all__ = [
    "build_candidates",
    "candidate_from_hermes_metadata",
    "route_hermes_turn",
]

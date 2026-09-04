"""Hermes runtime integration helpers.

This module is intentionally small: SmartRouter decides *where* a turn should
run; Hermes remains responsible for actually constructing clients, credentials,
transports, retries, and fallback behavior.

The integration uses Hermes' existing ``AIAgent.switch_model`` as the single
runtime activation mechanism. Because that method normally means a persistent
/model switch, ``routed_turn`` snapshots the primary runtime and restores it at
the end of the user turn. The selected model therefore behaves like a
turn-scoped route, not a hidden permanent /model change.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterable, Iterator

from .hermes_adapter import build_candidates
from .smart_router import CostPolicy, ModelCandidate, RoutingDecision, RoutingRequest, SmartRouter


def _runtime_snapshot(agent: Any) -> dict[str, Any]:
    """Capture the runtime state needed to undo a transient smart route."""
    names = (
        "model",
        "provider",
        "base_url",
        "api_mode",
        "api_key",
        "client",
        "_anthropic_client",
        "_anthropic_api_key",
        "_anthropic_base_url",
        "_is_anthropic_oauth",
        "_config_context_length",
        "_bedrock_region",
        "_use_prompt_caching",
        "_use_native_cache_layout",
        "_cached_system_prompt",
        "_fallback_chain",
        "_fallback_model",
        "_fallback_index",
        "_fallback_activated",
        "_primary_runtime",
        "_client_kwargs",
    )
    snapshot: dict[str, Any] = {}
    missing = object()
    for name in names:
        value = getattr(agent, name, missing)
        if value is not missing:
            snapshot[name] = deepcopy(value)
    return snapshot


def _restore_snapshot(agent: Any, snapshot: dict[str, Any]) -> None:
    """Restore a previously captured runtime snapshot."""
    for name, value in snapshot.items():
        try:
            setattr(agent, name, value)
        except Exception:
            # Restoration is best effort; the original runtime helper remains
            # the authoritative mechanism for ordinary fallback recovery.
            continue


def _candidate_metadata_from_hermes(provider: str, model: str) -> Any:
    """Resolve rich model metadata through Hermes' canonical models.dev layer."""
    from agent.models_dev import get_model_info

    return get_model_info(provider, model)


def discover_candidates(
    router: SmartRouter,
    providers: Iterable[str],
    *,
    free_models: dict[str, Iterable[str]] | None = None,
    force_refresh: bool = False,
) -> list[ModelCandidate]:
    """Discover configured provider catalogs and translate them to router candidates.

    ``free_models`` is an explicit entitlement declaration. A missing price is
    never interpreted as free.
    """
    free_models = free_models or {}
    provider_models: dict[str, Iterable[str]] = {}
    for provider in providers:
        provider_models[provider] = router.discover_models(
            provider, force_refresh=force_refresh
        )

    return build_candidates(
        provider_models,
        _candidate_metadata_from_hermes,
        free_models=free_models,
    )


def _as_switch_kwargs(candidate: ModelCandidate) -> dict[str, Any]:
    """Build the minimal target accepted by ``AIAgent.switch_model``."""
    extra = candidate.extra or {}
    return {
        "new_model": candidate.model,
        "new_provider": candidate.provider,
        "api_key": extra.get("api_key", ""),
        "base_url": extra.get("base_url", ""),
        "api_mode": extra.get("api_mode", ""),
    }


@contextmanager
def routed_turn(agent: Any, decision: RoutingDecision) -> Iterator[RoutingDecision]:
    """Activate a routing decision for exactly one user turn.

    Explicit model selection should be checked by the caller before entering
    this context. The context itself is deliberately provider-agnostic and
    delegates activation to Hermes' existing runtime switch implementation.
    """
    snapshot = _runtime_snapshot(agent)
    try:
        if (
            decision.primary.provider == snapshot.get("provider")
            and decision.primary.model == snapshot.get("model")
        ):
            yield decision
            return

        agent.switch_model(**_as_switch_kwargs(decision.primary))

        # Make the original primary an in-turn fallback. Hermes owns the actual
        # fallback activation/retry path; this merely preserves the invariant
        # that a smart-routed turn can recover to the user's normal model.
        original_provider = str(snapshot.get("provider") or "").strip()
        original_model = str(snapshot.get("model") or "").strip()
        if original_provider and original_model:
            original_fallback = {
                "provider": original_provider,
                "model": original_model,
                "base_url": snapshot.get("base_url") or "",
                "api_key": snapshot.get("api_key") or "",
                "api_mode": snapshot.get("api_mode") or "",
            }
            current_chain = list(getattr(agent, "_fallback_chain", []) or [])
            if not any(
                str(item.get("provider") or "").strip().lower() == original_provider.lower()
                and str(item.get("model") or "").strip() == original_model
                for item in current_chain
                if isinstance(item, dict)
            ):
                agent._fallback_chain = [original_fallback, *current_chain]
                agent._fallback_model = original_fallback
                agent._fallback_index = 0

        yield decision
    finally:
        # Never leave automatic routing looking like a user-issued persistent
        # /model switch. Restoring the snapshot also restores the original
        # fallback chain and primary runtime bookkeeping.
        _restore_snapshot(agent, snapshot)


def route_turn(
    router: SmartRouter,
    *,
    prompt: str,
    candidates: Iterable[ModelCandidate],
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY,
    requires_tools: bool = False,
    requires_vision: bool = False,
    requires_reasoning: bool = False,
    min_context_length: int | None = None,
    task_class: Any = None,
) -> RoutingDecision:
    """Convenience wrapper for the per-turn decision boundary."""
    request = RoutingRequest(
        prompt=prompt,
        cost_policy=cost_policy,
        requires_tools=requires_tools,
        requires_vision=requires_vision,
        requires_reasoning=requires_reasoning,
        min_context_length=min_context_length,
        task_class=task_class,
    )
    return router.route(request, candidates)


__all__ = [
    "discover_candidates",
    "route_turn",
    "routed_turn",
]

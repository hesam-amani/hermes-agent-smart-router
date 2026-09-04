"""Hermes runtime integration helpers.

SmartRouter decides *where* a turn should run; Hermes remains responsible for
clients, credentials, transports, retries, cooldowns, and fallback execution.

The integration uses Hermes' existing ``AIAgent.switch_model`` as the runtime
activation mechanism, then restores the original runtime when the user turn
ends. This makes automatic routing turn-scoped rather than a hidden persistent
``/model`` switch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from .hermes_adapter import build_candidates
from .smart_router import (
    CostPolicy,
    ModelCandidate,
    RoutingDecision,
    RoutingRequest,
    SmartRouter,
)


_SNAPSHOT_NAMES = (
    "model", "provider", "base_url", "api_mode", "api_key", "client",
    "_anthropic_client", "_anthropic_api_key", "_anthropic_base_url",
    "_is_anthropic_oauth", "_config_context_length", "_bedrock_region",
    "_use_prompt_caching", "_use_native_cache_layout", "_cached_system_prompt",
    "_fallback_chain", "_fallback_model", "_fallback_index", "_fallback_activated",
    "_primary_runtime", "_client_kwargs",
)


def _runtime_snapshot(agent: Any) -> dict[str, Any]:
    """Capture runtime state without copying live client objects."""
    snapshot: dict[str, Any] = {}
    missing = object()
    for name in _SNAPSHOT_NAMES:
        value = getattr(agent, name, missing)
        if value is missing:
            continue
        if name in {"_fallback_chain", "_client_kwargs", "_primary_runtime"}:
            if isinstance(value, dict):
                snapshot[name] = dict(value)
            elif isinstance(value, list):
                snapshot[name] = [dict(x) if isinstance(x, dict) else x for x in value]
            else:
                snapshot[name] = value
        else:
            snapshot[name] = value
    return snapshot


def _restore_snapshot(agent: Any, snapshot: dict[str, Any]) -> None:
    """Restore a previously captured runtime snapshot."""
    for name, value in snapshot.items():
        try:
            setattr(agent, name, value)
        except Exception:
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
    """Discover provider catalogs using Hermes' existing catalog machinery."""
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
    """Build a fully resolved target for Hermes' ``AIAgent.switch_model``."""
    extra = dict(candidate.extra or {})
    runtime: dict[str, Any] = {}
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolved = resolve_runtime_provider(
            requested=candidate.provider,
            target_model=candidate.model,
            explicit_base_url=extra.get("base_url") or None,
            explicit_api_key=extra.get("api_key") or None,
        )
        if isinstance(resolved, dict):
            runtime = resolved
    except Exception:
        # Activation will fail closed through Hermes' existing switch path; the
        # router itself must not invent credentials or endpoints.
        runtime = {}

    return {
        "new_model": candidate.model,
        "new_provider": str(runtime.get("provider") or candidate.provider),
        "api_key": extra.get("api_key") or runtime.get("api_key") or "",
        "base_url": extra.get("base_url") or runtime.get("base_url") or "",
        "api_mode": extra.get("api_mode") or runtime.get("api_mode") or "",
    }


@contextmanager
def routed_turn(agent: Any, decision: RoutingDecision) -> Iterator[RoutingDecision]:
    """Activate one routing decision for exactly one user turn.

    The caller must perform the explicit-model override check before entering
    this context. Selected routes use Hermes' native runtime switch and its
    normal fallback machinery.
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
            duplicate = any(
                isinstance(item, dict)
                and str(item.get("provider") or "").strip().lower() == original_provider.lower()
                and str(item.get("model") or "").strip() == original_model
                for item in current_chain
            )
            if not duplicate:
                agent._fallback_chain = [original_fallback, *current_chain]
                agent._fallback_model = original_fallback
                agent._fallback_index = 0

        yield decision
    finally:
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


__all__ = ["discover_candidates", "route_turn", "routed_turn"]

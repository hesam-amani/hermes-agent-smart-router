# Hermes Agent Smart Router

Task-aware model selection for Hermes Agent.

## Architecture

The router is intentionally a decision layer above Hermes' existing provider runtime:

```text
request
  -> task classification
  -> capability filtering
  -> cost/entitlement policy
  -> runtime health
  -> model ranking
  -> primary + fallback candidates
  -> Hermes model/provider activation
  -> existing retry/fallback execution
```

The router does **not** own provider clients, credentials, retry logic, cooldowns, or a second model catalog.

## Current status

- `agent/smart_router.py` — core task/capability/cost/health ranking engine.
- `agent/hermes_adapter.py` — adapter for Hermes-style model metadata and discovery.
- `tests/test_smart_router.py` — core routing coverage.

Hermes already exposes `smart_model_routing` as a configuration surface, while its current provider/model machinery remains the canonical source for model discovery and activation. The next upstream-facing step is a small core integration that invokes this decision layer at the automatic model-selection boundary and hands the resulting provider/model back to Hermes' existing resolver.

### Safety invariant

`FREE_ONLY` requires an explicit free entitlement. Missing pricing metadata is **not** treated as free.

Explicit user model selection remains authoritative; automatic routing must never silently override an explicit `/model` choice.

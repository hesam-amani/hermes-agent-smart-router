from agent.smart_router import (
    CostPolicy,
    ModelCandidate,
    RoutingRequest,
    SmartRouter,
    TaskClass,
)


def candidate(name, **kwargs):
    return ModelCandidate(provider="test", model=name, **kwargs)


def test_free_only_excludes_paid_models():
    router = SmartRouter()
    models = [candidate("paid", free=False), candidate("free", free=True)]
    decision = router.route(RoutingRequest("hello", cost_policy=CostPolicy.FREE_ONLY), models)
    assert decision.primary.model == "free"
    assert all(model.free for model in decision.chain)


def test_tools_are_hard_capability_requirement():
    router = SmartRouter()
    models = [
        candidate("no-tools", free=True, supports_tools=False),
        candidate("tools", free=True, supports_tools=True),
    ]
    decision = router.route(
        RoutingRequest("use a tool", requires_tools=True),
        models,
    )
    assert decision.primary.model == "tools"


def test_coding_task_prefers_coding_model():
    router = SmartRouter()
    models = [
        candidate("general", free=True, quality=0.9, coding=0.4),
        candidate("coder", free=True, quality=0.8, coding=1.0),
    ]
    decision = router.route(RoutingRequest("fix this Python function bug"), models)
    assert decision.task_class is TaskClass.CODING
    assert decision.primary.model == "coder"


def test_reasoning_task_prefers_reasoning_model():
    router = SmartRouter()
    models = [
        candidate("general", free=True, quality=0.9),
        candidate("reasoner", free=True, quality=0.8, supports_reasoning=True),
    ]
    decision = router.route(RoutingRequest("prove why this solution is correct"), models)
    assert decision.task_class is TaskClass.REASONING
    assert decision.primary.model == "reasoner"


def test_health_observations_change_ranking():
    router = SmartRouter()
    fast = candidate("fast", free=True, quality=0.8)
    slow = candidate("slow", free=True, quality=0.9)
    router.observe(slow, success=False)
    router.observe(slow, success=False)
    router.observe(fast, success=True, latency_seconds=0.1)
    decision = router.route(RoutingRequest("hello"), [slow, fast])
    assert decision.primary.model == "fast"


def test_fallback_chain_is_ranked():
    router = SmartRouter()
    models = [
        candidate("a", free=True, quality=1.0),
        candidate("b", free=True, quality=0.8),
        candidate("c", free=True, quality=0.6),
    ]
    decision = router.route(RoutingRequest("hello"), models)
    assert [m.model for m in decision.chain] == ["a", "b", "c"]


def test_no_candidate_is_an_explicit_failure():
    router = SmartRouter()
    try:
        router.route(
            RoutingRequest("hello", cost_policy=CostPolicy.FREE_ONLY),
            [candidate("paid", free=False)],
        )
    except LookupError as exc:
        assert "No model satisfies" in str(exc)
    else:
        raise AssertionError("expected LookupError")

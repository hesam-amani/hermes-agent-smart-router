from types import SimpleNamespace
from unittest.mock import patch

from agent.hermes_runtime_integration import routed_turn, route_turn
from agent.smart_router import CostPolicy, ModelCandidate, SmartRouter, TaskClass


def _candidate(provider="nvidia", model="free-model"):
    return ModelCandidate(
        provider=provider,
        model=model,
        free=True,
        supports_tools=True,
        supports_reasoning=True,
        quality=0.9,
        coding=0.9,
    )


def _agent():
    return SimpleNamespace(
        model="main-model",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        api_key="main-key",
        client=object(),
        _anthropic_client=None,
        _anthropic_api_key="",
        _anthropic_base_url="",
        _is_anthropic_oauth=False,
        _config_context_length=None,
        _bedrock_region=None,
        _use_prompt_caching=False,
        _use_native_cache_layout=False,
        _cached_system_prompt="cached",
        _fallback_chain=[{"provider": "zai", "model": "glm-test"}],
        _fallback_model={"provider": "zai", "model": "glm-test"},
        _fallback_index=0,
        _fallback_activated=False,
        _primary_runtime={"model": "main-model", "provider": "openrouter"},
        _client_kwargs={"api_key": "main-key", "base_url": "https://openrouter.ai/api/v1"},
    )


def test_route_turn_selects_candidate():
    router = SmartRouter()
    decision = route_turn(
        router,
        prompt="write python code to fix this bug",
        candidates=[_candidate()],
        cost_policy=CostPolicy.FREE_ONLY,
    )
    assert decision.task_class is TaskClass.CODING
    assert decision.primary.provider == "nvidia"


def test_routed_turn_activates_then_restores():
    agent = _agent()
    calls = []

    def switch_model(**kwargs):
        calls.append(kwargs)
        agent.model = kwargs["new_model"]
        agent.provider = kwargs["new_provider"]
        agent.base_url = kwargs["base_url"]
        agent.api_mode = kwargs["api_mode"]
        agent.api_key = kwargs["api_key"]

    agent.switch_model = switch_model
    decision = route_turn(
        SmartRouter(),
        prompt="explain this code",
        candidates=[_candidate()],
    )

    with patch(
        "agent.hermes_runtime_integration._as_switch_kwargs",
        return_value={
            "new_model": "free-model",
            "new_provider": "nvidia",
            "api_key": "free-key",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_mode": "chat_completions",
        },
    ):
        with routed_turn(agent, decision):
            assert agent.model == "free-model"
            assert agent.provider == "nvidia"
            assert agent._fallback_chain[0]["model"] == "main-model"

    assert calls
    assert agent.model == "main-model"
    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"
    assert agent.api_key == "main-key"
    assert agent._fallback_chain == [{"provider": "zai", "model": "glm-test"}]
    assert agent._primary_runtime == {"model": "main-model", "provider": "openrouter"}


def test_same_route_does_not_switch():
    agent = _agent()
    agent.model = "free-model"
    agent.provider = "nvidia"
    agent.switch_model = lambda **_: (_ for _ in ()).throw(AssertionError("unexpected switch"))
    decision = route_turn(
        SmartRouter(),
        prompt="hello",
        candidates=[_candidate(provider="nvidia", model="free-model")],
    )
    with routed_turn(agent, decision):
        assert agent.model == "free-model"
        assert agent.provider == "nvidia"

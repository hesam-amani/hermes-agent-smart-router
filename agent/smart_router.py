"""Smart model routing layer for Hermes Agent.

This module intentionally sits above Hermes' existing provider/model
resolution machinery. It does not replace provider discovery, credential
rotation, retry, cooldown, or fallback infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Iterable


class CostPolicy(str, Enum):
    FREE_ONLY = "free_only"
    FREE_PREFERRED = "free_preferred"
    ANY = "any"


class TaskClass(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    RESEARCH = "research"
    LONG_CONTEXT = "long_context"
    VISION = "vision"
    CASUAL = "casual"
    GENERAL = "general"


@dataclass(frozen=True)
class ModelCandidate:
    """Normalized model metadata consumed by the router.

    The router deliberately accepts plain metadata rather than owning a
    second provider catalog. Callers can populate these records from
    Hermes' existing model metadata/discovery systems.
    """

    provider: str
    model: str
    free: bool = False
    context_length: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    quality: float = 0.5
    coding: float = 0.5
    research: float = 0.5
    latency_score: float = 0.5
    reliability_score: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingRequest:
    prompt: str
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY
    requires_tools: bool = False
    requires_vision: bool = False
    requires_reasoning: bool = False
    min_context_length: int | None = None
    task_class: TaskClass | None = None


@dataclass(frozen=True)
class RoutingDecision:
    task_class: TaskClass
    primary: ModelCandidate
    fallbacks: tuple[ModelCandidate, ...]
    scores: dict[str, float]

    @property
    def chain(self) -> tuple[ModelCandidate, ...]:
        return (self.primary, *self.fallbacks)


@dataclass
class _Health:
    successes: int = 0
    failures: int = 0
    latency_total: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.5

    @property
    def latency_score(self) -> float:
        if not self.successes:
            return 0.5
        avg = self.latency_total / self.successes
        return 1.0 / (1.0 + avg)


class SmartRouter:
    """Task-aware model selector built on top of Hermes model discovery."""

    def __init__(self) -> None:
        self._health: dict[tuple[str, str], _Health] = {}

    def observe(
        self,
        candidate: ModelCandidate,
        *,
        success: bool,
        latency_seconds: float | None = None,
    ) -> None:
        health = self._health.setdefault((candidate.provider, candidate.model), _Health())
        if success:
            health.successes += 1
            if latency_seconds is not None:
                health.latency_total += max(0.0, latency_seconds)
        else:
            health.failures += 1

    def classify(self, prompt: str) -> TaskClass:
        text = prompt.lower()
        coding = (
            "```" in text
            or any(x in text for x in ("code", "python", "typescript", "bug", "compile", "function", "api"))
        )
        research = any(x in text for x in ("research", "sources", "cite", "compare", "investigate", "latest"))
        reasoning = any(x in text for x in ("prove", "derive", "analyze", "reason", "why", "solve"))
        vision = any(x in text for x in ("image", "photo", "screenshot", "picture", "vision"))
        long_context = any(x in text for x in ("long document", "entire repository", "whole codebase", "large context"))

        if vision:
            return TaskClass.VISION
        if coding:
            return TaskClass.CODING
        if research:
            return TaskClass.RESEARCH
        if long_context:
            return TaskClass.LONG_CONTEXT
        if reasoning:
            return TaskClass.REASONING
        if len(text.split()) < 20:
            return TaskClass.CASUAL
        return TaskClass.GENERAL

    def filter_candidates(
        self,
        candidates: Iterable[ModelCandidate],
        request: RoutingRequest,
    ) -> list[ModelCandidate]:
        result: list[ModelCandidate] = []
        for candidate in candidates:
            if request.cost_policy is CostPolicy.FREE_ONLY and not candidate.free:
                continue
            if request.requires_tools and not candidate.supports_tools:
                continue
            if request.requires_vision and not candidate.supports_vision:
                continue
            if request.requires_reasoning and not candidate.supports_reasoning:
                continue
            if (
                request.min_context_length is not None
                and (candidate.context_length or 0) < request.min_context_length
            ):
                continue
            result.append(candidate)
        return result

    def score(self, candidate: ModelCandidate, task: TaskClass) -> float:
        health = self._health.get((candidate.provider, candidate.model))
        reliability = health.success_rate if health else candidate.reliability_score
        latency = health.latency_score if health else candidate.latency_score

        task_fit = {
            TaskClass.CODING: candidate.coding,
            TaskClass.RESEARCH: candidate.research,
            TaskClass.REASONING: candidate.supports_reasoning * 0.5 + candidate.quality * 0.5,
            TaskClass.VISION: float(candidate.supports_vision),
            TaskClass.LONG_CONTEXT: min((candidate.context_length or 0) / 200_000, 1.0),
            TaskClass.CASUAL: candidate.quality,
            TaskClass.GENERAL: candidate.quality,
        }[task]

        # Quality/task fit dominate; reliability and latency adapt over time.
        return (
            0.40 * task_fit
            + 0.25 * candidate.quality
            + 0.20 * reliability
            + 0.15 * latency
        )

    def route(
        self,
        request: RoutingRequest,
        candidates: Iterable[ModelCandidate],
    ) -> RoutingDecision:
        task = request.task_class or self.classify(request.prompt)
        filtered = self.filter_candidates(candidates, request)
        if not filtered:
            raise LookupError("No model satisfies the smart-routing policy and capabilities")

        ranked = sorted(
            filtered,
            key=lambda c: self.score(c, task),
            reverse=True,
        )
        scores = {
            f"{c.provider}/{c.model}": self.score(c, task)
            for c in ranked
        }
        return RoutingDecision(task, ranked[0], tuple(ranked[1:]), scores)

    def discover_models(self, provider: str, *, force_refresh: bool = False) -> list[str]:
        """Delegate discovery to Hermes' canonical model catalog machinery."""
        from hermes_cli.models import provider_model_ids

        return provider_model_ids(provider, force_refresh=force_refresh)


__all__ = [
    "CostPolicy",
    "ModelCandidate",
    "RoutingDecision",
    "RoutingRequest",
    "SmartRouter",
    "TaskClass",
]

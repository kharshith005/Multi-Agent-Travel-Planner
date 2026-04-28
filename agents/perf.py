"""Performance instrumentation — Timeline for coordinator phase breakdown.

A Timeline is created per plan_trip call, passed through CoordinatorState,
and consumed by the Streamlit UI (Performance expander) and eval CSV.

Usage:
    tl = Timeline()
    with tl.phase("parse"):
        ...  # timed block
    summary = tl.summary()  # dict with total_ms and per-phase list

merge_timelines is a LangGraph reducer for Annotated[Timeline, merge_timelines]:
parallel nodes each return a fresh local Timeline; the reducer concatenates them.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class PhaseEntry:
    name: str
    start: float
    end: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end <= 0:
            return 0.0
        return (self.end - self.start) * 1000.0


@dataclass
class Timeline:
    phases: list[PhaseEntry] = field(default_factory=list)

    @contextmanager
    def phase(self, name: str) -> Generator[PhaseEntry, None, None]:
        entry = PhaseEntry(name=name, start=time.perf_counter())
        self.phases.append(entry)
        try:
            yield entry
        finally:
            entry.end = time.perf_counter()

    def total_ms(self) -> float:
        return sum(e.duration_ms for e in self.phases)

    def total_cache_hits(self) -> int:
        return sum(e.cache_hits for e in self.phases)

    def total_cache_misses(self) -> int:
        return sum(e.cache_misses for e in self.phases)

    def summary(self) -> dict:
        sorted_phases = sorted(self.phases, key=lambda e: e.start)
        return {
            "total_ms": self.total_ms(),
            "cache_hits": self.total_cache_hits(),
            "cache_misses": self.total_cache_misses(),
            "phases": [
                {
                    "name": e.name,
                    "duration_ms": round(e.duration_ms, 1),
                    "cache_hits": e.cache_hits,
                    "cache_misses": e.cache_misses,
                }
                for e in sorted_phases
            ],
        }


def merge_timelines(left: "Timeline | None", right: "Timeline | None") -> "Timeline":
    """LangGraph reducer: merge two Timelines written by parallel superstep nodes."""
    if left is None:
        return right or Timeline()
    if right is None:
        return left
    if left is right:
        return left
    return Timeline(phases=left.phases + right.phases)

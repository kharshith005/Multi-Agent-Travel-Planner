"""Tests for agents/perf.py — Timeline and merge_timelines reducer."""
from __future__ import annotations

import time

import pytest

from agents.perf import PhaseEntry, Timeline, merge_timelines


# ── PhaseEntry ────────────────────────────────────────────────────────────────

def test_phase_entry_duration():
    now = time.perf_counter()
    e = PhaseEntry(name="p", start=now, end=now + 0.1)
    assert abs(e.duration_ms - 100.0) < 5

def test_phase_entry_not_closed():
    e = PhaseEntry(name="p", start=time.perf_counter(), end=0.0)
    assert e.duration_ms == 0.0


# ── Timeline.phase context manager ───────────────────────────────────────────

def test_timeline_records_phase():
    tl = Timeline()
    with tl.phase("parse") as ph:
        time.sleep(0.01)
    assert len(tl.phases) == 1
    assert tl.phases[0].name == "parse"
    assert tl.phases[0].duration_ms >= 10

def test_timeline_multiple_phases():
    tl = Timeline()
    with tl.phase("a"): pass
    with tl.phase("b"): pass
    assert [p.name for p in tl.phases] == ["a", "b"]

def test_timeline_total_ms():
    tl = Timeline()
    with tl.phase("x"):
        time.sleep(0.02)
    assert tl.total_ms() >= 20

def test_timeline_cache_counters():
    tl = Timeline()
    with tl.phase("llm") as ph:
        ph.cache_hits = 2
        ph.cache_misses = 1
    assert tl.total_cache_hits() == 2
    assert tl.total_cache_misses() == 1


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_keys():
    tl = Timeline()
    with tl.phase("p"): pass
    s = tl.summary()
    assert "total_ms" in s
    assert "cache_hits" in s
    assert "cache_misses" in s
    assert "phases" in s
    assert s["phases"][0]["name"] == "p"

def test_summary_sorted_by_start():
    """Phases added out of chronological order still sort by start time."""
    tl = Timeline()
    # Manufacture two entries with swapped start times
    early = PhaseEntry(name="early", start=1000.0, end=1001.0)
    late = PhaseEntry(name="late", start=2000.0, end=2001.0)
    tl.phases = [late, early]  # intentionally reversed
    s = tl.summary()
    assert s["phases"][0]["name"] == "early"
    assert s["phases"][1]["name"] == "late"


# ── merge_timelines ───────────────────────────────────────────────────────────

def test_merge_none_none():
    result = merge_timelines(None, None)
    assert isinstance(result, Timeline)
    assert result.phases == []

def test_merge_left_none():
    t = Timeline()
    with t.phase("p"): pass
    result = merge_timelines(None, t)
    assert result is t

def test_merge_right_none():
    t = Timeline()
    with t.phase("p"): pass
    result = merge_timelines(t, None)
    assert result is t

def test_merge_same_object():
    t = Timeline()
    result = merge_timelines(t, t)
    assert result is t

def test_merge_two_timelines():
    t1 = Timeline()
    t2 = Timeline()
    with t1.phase("parse"): pass
    with t2.phase("lodging"): pass
    with t2.phase("dining"): pass
    merged = merge_timelines(t1, t2)
    assert len(merged.phases) == 3
    names = {p.name for p in merged.phases}
    assert names == {"parse", "lodging", "dining"}

def test_merge_does_not_mutate_inputs():
    t1 = Timeline()
    t2 = Timeline()
    with t1.phase("a"): pass
    with t2.phase("b"): pass
    merge_timelines(t1, t2)
    assert len(t1.phases) == 1
    assert len(t2.phases) == 1

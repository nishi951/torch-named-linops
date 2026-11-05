import torch

from benchmarks._compiled_phases import pipeline_break_kind


def test_pipeline_break_kind_attributes_wrapper_logging():
    assert (
        pipeline_break_kind(["at SyncContext logger.debug"])
        == "wrapper (logging/namedlinop)"
    )


def test_pipeline_break_kind_attributes_triton():
    assert (
        pipeline_break_kind(["triton kernel unsupported"]) == "inner (triton/inductor)"
    )


def test_pipeline_break_kind_none():
    assert pipeline_break_kind([]) == "none"


def test_simple_chain_builds_on_cpu():
    from benchmarks._compiled_phases import _simple_chain

    chain, x = _simple_chain("cpu")
    assert chain(x).shape == (2, 8)

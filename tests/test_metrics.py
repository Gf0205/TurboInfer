from turboinfer.metrics import summarize_token_timings


def test_summarize_token_timings_empty_output() -> None:
    ttft, tpot, tokens_per_second = summarize_token_timings([], 10.0, 12.0)

    assert ttft is None
    assert tpot is None
    assert tokens_per_second == 0.0


def test_summarize_token_timings_multiple_tokens() -> None:
    ttft, tpot, tokens_per_second = summarize_token_timings(
        [10.5, 11.0, 12.0],
        10.0,
        13.0,
    )

    assert ttft == 0.5
    assert tpot == 0.75
    assert tokens_per_second == 1.0


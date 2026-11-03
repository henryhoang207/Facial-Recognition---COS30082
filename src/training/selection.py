"""Ordered checkpoint comparison without mixing metrics on different scales."""

from collections.abc import Mapping, Sequence

from .config import MetricMonitor


def improves(
    current: Mapping[str, float],
    previous: Mapping[str, float] | None,
    monitors: Sequence[MetricMonitor],
) -> bool:
    """Compare by priority; differences within min_delta advance to the next metric.

    A tie on every configured metric keeps the earlier checkpoint.
    """
    if previous is None:
        return True
    for item in monitors:
        gain = current[item.metric] - previous[item.metric]
        if item.mode == "min":
            gain = -gain
        if gain > item.min_delta:
            return True
        if gain < -item.min_delta:
            return False
    return False

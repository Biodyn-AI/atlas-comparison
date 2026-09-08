"""Depth-matched layer selection — Igor's spec: first (0), 25%, 50%, 75%, last.
Makes layer choice comparable across models of different depths (a 6-layer model and a
26-layer model are read at the same relative depths), removing the layer confound.

depth_matched(n_layers) returns block indices 0..n_layers-1 at the five relative depths,
deduplicated (small models may collapse some), sorted.
"""
from __future__ import annotations


def depth_matched(n_layers: int) -> tuple[int, ...]:
    last = n_layers - 1
    picks = {round(f * last) for f in (0.0, 0.25, 0.5, 0.75, 1.0)}
    return tuple(sorted(picks))


if __name__ == "__main__":
    for n in (4, 6, 12, 18, 26, 33):
        print(f"{n:>3} layers -> {depth_matched(n)}")

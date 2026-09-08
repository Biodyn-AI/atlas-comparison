"""Phase-0 adapter contract. Each model plugs in by implementing `iter_activations`,
which yields per-layer residual-stream activation rows plus the HGNC gene symbol for
every row (position). Everything downstream (SAE training, catalog, annotation) is
model-agnostic and consumes only these.

Mirrors bio-sae's Phase 0 (src/01_extract_activations.py, scgpt_src/01_...): forward
hooks on each transformer block's POST-residual output, unfolded to per-(cell,gene) rows.
"""
from __future__ import annotations
from typing import Iterator, Sequence
import numpy as np


class Adapter:
    name: str = "base"
    layers: Sequence[int] = ()      # residual layers to extract (0 = input embedding)
    d_model: int = 0

    def load(self, device: str = "cuda") -> None:
        """Load model weights + vocab in eval mode onto `device`."""
        raise NotImplementedError

    def iter_activations(self, adata, batch_size: int = 8
                         ) -> Iterator[tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]]:
        """Yield, per batch of cells:
            acts: {layer: float32 array [n_positions, d_model]}
            syms: str array [n_positions]  — HGNC symbol (UPPER) for each position
            cell_ids: int array [n_positions] — global processing-order cell index (0-based)
        n_positions = sum over the batch's cells of that cell's gene-token count.
        Only *gene* token positions (drop CLS/pad/depth/special tokens).

        For the perturbation test, an adapter that preprocesses/filters cells MUST expose
        `self.processed_obs` (a pandas DataFrame of the cells actually processed, in yield
        order) so callers can recover per-cell labels aligned to cell_ids.
        """
        raise NotImplementedError

    processed_obs = None    # set by iter_activations after preprocessing (perturbation test)
    meta: dict = {}

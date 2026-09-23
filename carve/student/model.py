from __future__ import annotations

import random

from .dataset import GraphArrays


class NumpyGraphStudent:
    """Small dependency-light graph reward head for smoke tests.

    The full paper path can replace this with Qwen embeddings + R-GAT while preserving
    the same graph input and prediction interface.
    """

    def __init__(self, num_event_types: int, hidden_dim: int = 32, seed: int = 0):
        rng = random.Random(seed)
        self.type_emb = [[rng.gauss(0.0, 0.05) for _ in range(hidden_dim)] for _ in range(num_event_types)]
        self.feature_proj = [[rng.gauss(0.0, 0.05) for _ in range(hidden_dim)] for _ in range(5)]
        self.out = [rng.gauss(0.0, 0.05) for _ in range(hidden_dim)]

    def forward(self, arrays: GraphArrays) -> list[float]:
        h = []
        for typ, feats in zip(arrays.node_type, arrays.node_features, strict=True):
            row = list(self.type_emb[typ])
            for feat_idx, feat in enumerate(feats):
                for j, weight in enumerate(self.feature_proj[feat_idx]):
                    row[j] += feat * weight
            h.append(row)
        if arrays.edge_index:
            agg = [[0.0 for _ in row] for row in h]
            counts = [1 for _ in h]
            for src, dst in arrays.edge_index:
                for j, value in enumerate(h[src]):
                    agg[dst][j] += value
                counts[dst] += 1
            for i, row in enumerate(h):
                for j in range(len(row)):
                    row[j] += agg[i][j] / counts[i]
        return [sum(v * w for v, w in zip(row, self.out, strict=True)) for row in h]


class RelationalGraphStudent:
    """Dependency-light relational graph reward head.

    This mirrors the paper-facing CARVE-S interface: typed event nodes, role
    embeddings, text features, and relation-specific graph propagation. A later
    Qwen/R-GAT backend can replace the projections while preserving this API.
    """

    def __init__(
        self,
        num_event_types: int,
        num_roles: int,
        num_edge_types: int,
        hidden_dim: int = 32,
        text_dim: int = 8,
        seed: int = 0,
    ):
        rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.type_emb = [[rng.gauss(0.0, 0.05) for _ in range(hidden_dim)] for _ in range(num_event_types)]
        self.role_emb = [[rng.gauss(0.0, 0.05) for _ in range(hidden_dim)] for _ in range(num_roles)]
        self.feature_proj = [[rng.gauss(0.0, 0.05) for _ in range(hidden_dim)] for _ in range(5)]
        self.text_proj = [[rng.gauss(0.0, 0.05) for _ in range(hidden_dim)] for _ in range(text_dim)]
        self.rel_proj = [
            [[rng.gauss(0.0, 0.03) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
            for _ in range(num_edge_types)
        ]
        self.out = [rng.gauss(0.0, 0.05) for _ in range(hidden_dim)]
        self.uncertainty_out = [abs(rng.gauss(0.0, 0.03)) for _ in range(hidden_dim)]

    def _initial_states(self, arrays: GraphArrays) -> list[list[float]]:
        states = []
        for typ, role, feats, text_feats in zip(
            arrays.node_type,
            arrays.role_id,
            arrays.node_features,
            arrays.text_features,
            strict=True,
        ):
            row = [a + b for a, b in zip(self.type_emb[typ], self.role_emb[role], strict=True)]
            for feat_idx, feat in enumerate(feats):
                for j, weight in enumerate(self.feature_proj[feat_idx]):
                    row[j] += feat * weight
            for feat_idx, feat in enumerate(text_feats):
                if feat_idx >= len(self.text_proj):
                    break
                for j, weight in enumerate(self.text_proj[feat_idx]):
                    row[j] += feat * weight
            states.append(row)
        return states

    def _propagate(self, arrays: GraphArrays) -> list[list[float]]:
        states = self._initial_states(arrays)
        if not arrays.edge_index:
            return states
        agg = [[0.0 for _ in range(self.hidden_dim)] for _ in states]
        counts = [1 for _ in states]
        for (src, dst), rel in zip(arrays.edge_index, arrays.edge_type, strict=True):
            rel_matrix = self.rel_proj[rel]
            for j in range(self.hidden_dim):
                agg[dst][j] += sum(states[src][k] * rel_matrix[k][j] for k in range(self.hidden_dim))
            counts[dst] += 1
        for i, row in enumerate(states):
            for j in range(self.hidden_dim):
                row[j] += agg[i][j] / counts[i]
        return states

    def forward(self, arrays: GraphArrays) -> list[float]:
        return [sum(v * w for v, w in zip(row, self.out, strict=True)) for row in self._propagate(arrays)]

    def uncertainty(self, arrays: GraphArrays) -> list[float]:
        values = []
        for row in self._propagate(arrays):
            raw = sum(abs(v) * w for v, w in zip(row, self.uncertainty_out, strict=True))
            values.append(max(0.0, raw))
        return values

    def to_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "text_dim": self.text_dim,
            "type_emb": self.type_emb,
            "role_emb": self.role_emb,
            "feature_proj": self.feature_proj,
            "text_proj": self.text_proj,
            "rel_proj": self.rel_proj,
            "out": self.out,
            "uncertainty_out": self.uncertainty_out,
        }

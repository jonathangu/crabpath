"""Low-rank learned routing model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .index import VectorIndex


@dataclass
class RouteModel:
    """Low-rank bilinear route scorer with edge-feature linear terms."""

    r: int
    A: np.ndarray
    B: np.ndarray
    w_feat: np.ndarray
    b: float
    T: float

    @property
    def dq(self) -> int:
        return int(self.A.shape[0])

    @property
    def dt(self) -> int:
        return int(self.B.shape[0])

    @property
    def df(self) -> int:
        return int(self.w_feat.shape[0])

    def project_query(self, q_vec: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(q_vec, dtype=float)
        if arr.ndim != 1:
            raise ValueError("query vector must be 1D")
        if arr.shape[0] == self.dq:
            return arr @ self.A
        if arr.shape[0] == self.r and self.dq != self.r:
            return arr
        raise ValueError(f"query vector length mismatch: expected {self.dq} or {self.r}, got {arr.shape[0]}")

    def project_target(self, t_vec: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(t_vec, dtype=float)
        if arr.ndim != 1:
            raise ValueError("target vector must be 1D")
        if arr.shape[0] == self.dt:
            return arr @ self.B
        if arr.shape[0] == self.r and self.dt != self.r:
            return arr
        raise ValueError(f"target vector length mismatch: expected {self.dt} or {self.r}, got {arr.shape[0]}")

    def score_projected(
        self,
        q_proj: list[float] | np.ndarray,
        t_proj: list[float] | np.ndarray,
        feat_vec: list[float] | np.ndarray,
    ) -> float:
        q_arr = np.asarray(q_proj, dtype=float)
        t_arr = np.asarray(t_proj, dtype=float)
        if q_arr.ndim != 1 or q_arr.shape[0] != self.r:
            raise ValueError(f"projected query length mismatch: expected {self.r}, got {q_arr.shape[0] if q_arr.ndim == 1 else 'nd'}")
        if t_arr.ndim != 1 or t_arr.shape[0] != self.r:
            raise ValueError(f"projected target length mismatch: expected {self.r}, got {t_arr.shape[0] if t_arr.ndim == 1 else 'nd'}")
        feat = np.asarray(feat_vec, dtype=float)
        if feat.ndim != 1 or feat.shape[0] != self.df:
            raise ValueError(f"feature vector length mismatch: expected {self.df}, got {feat.shape[0] if feat.ndim == 1 else 'nd'}")
        denom = self.T if self.T > 0 else 1.0
        value = float(np.dot(q_arr, t_arr) + np.dot(self.w_feat, feat) + float(self.b)) / denom
        return float(value)

    def score(
        self,
        q_vec: list[float] | np.ndarray,
        t_vec: list[float] | np.ndarray,
        feat_vec: list[float] | np.ndarray,
    ) -> float:
        q_proj = self.project_query(q_vec)
        t_proj = self.project_target(t_vec)
        return self.score_projected(q_proj, t_proj, feat_vec)

    def save_npz(self, path: str | Path) -> None:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            r=np.asarray(self.r, dtype=np.int64),
            A=self.A,
            B=self.B,
            w_feat=self.w_feat,
            b=np.asarray(self.b, dtype=float),
            T=np.asarray(self.T, dtype=float),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "RouteModel":
        payload = np.load(Path(path).expanduser(), allow_pickle=False)
        return cls(
            r=int(payload["r"]),
            A=np.asarray(payload["A"], dtype=float),
            B=np.asarray(payload["B"], dtype=float),
            w_feat=np.asarray(payload["w_feat"], dtype=float),
            b=float(payload["b"]),
            T=max(1e-6, float(payload["T"])),
        )

    def precompute_target_projections(self, index: VectorIndex) -> dict[str, np.ndarray]:
        projections: dict[str, np.ndarray] = {}
        for node_id, vector in index._vectors.items():
            try:
                projections[str(node_id)] = self.project_target(vector)
            except ValueError:
                continue
        return projections

    @classmethod
    def init_random(
        cls,
        dq: int,
        dt: int,
        df: int,
        rank: int,
    ) -> "RouteModel":
        if dq <= 0 or dt <= 0 or df <= 0 or rank <= 0:
            raise ValueError("dq, dt, df, and rank must be positive")
        rng = np.random.default_rng(0)
        scale = 0.01
        A = rng.normal(0.0, scale, size=(dq, rank)).astype(float)
        B = rng.normal(0.0, scale, size=(dt, rank)).astype(float)
        w_feat = np.zeros(df, dtype=float)
        return cls(r=int(rank), A=A, B=B, w_feat=w_feat, b=0.0, T=1.0)

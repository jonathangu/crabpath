from __future__ import annotations

from pathlib import Path

import numpy as np

from openclawbrain import VectorIndex
from openclawbrain.route_model import RouteModel


def test_route_model_save_load_roundtrip(tmp_path: Path) -> None:
    model = RouteModel.init_random(dq=4, dt=4, df=3, rank=2)
    model.w_feat = np.asarray([0.1, -0.2, 0.3], dtype=float)
    model.b = 0.7
    model.T = 0.9

    path = tmp_path / "route_model.npz"
    model.save_npz(path)
    loaded = RouteModel.load_npz(path)

    assert loaded.r == model.r
    assert np.allclose(loaded.A, model.A)
    assert np.allclose(loaded.B, model.B)
    assert np.allclose(loaded.w_feat, model.w_feat)
    assert loaded.b == model.b
    assert loaded.T == model.T


def test_route_model_score_is_deterministic() -> None:
    model = RouteModel.init_random(dq=3, dt=3, df=3, rank=2)
    q = [0.1, 0.2, 0.3]
    t = [0.4, 0.5, 0.6]
    f = [0.7, 0.8, 1.0]

    first = model.score(q, t, f)
    second = model.score(q, t, f)
    assert first == second


def test_route_model_init_random_shapes_and_projection() -> None:
    model = RouteModel.init_random(dq=5, dt=7, df=3, rank=4)
    assert model.A.shape == (5, 4)
    assert model.B.shape == (7, 4)
    assert model.w_feat.shape == (3,)

    index = VectorIndex()
    index.upsert("n1", [1.0] * 7)
    projections = model.precompute_target_projections(index)
    assert "n1" in projections
    assert len(projections["n1"]) == 4

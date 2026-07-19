"""Feasibility projection for sampled layouts.

Two projections onto the feasible set defined by
:func:`wind_rl.design.geometry.is_feasible` (in-bounds + pairwise min distance):
a soft Adam-penalty projection and a hard scipy-SLSQP projection. Both act on
``(B, N, 2)`` map-metre layouts and preserve batch/order. A small margin above
``min_distance`` guards against solver tolerance leaving a pair marginally inside
the (closed) constraint. Ported from diffusion-co-design's constraint module.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize

from wind_rl.scenario import ScenarioConfig

# Solvers stop at first-order tolerance; clearing min_distance by this factor
# keeps the result on the feasible side of the closed `>= min_distance` check.
_MARGIN = 1.001


# Both projections' pairwise-distance constraint must track
# design.geometry.pairwise_min_distance's Euclidean, closed `>=` definition --
# feasibility verdicts elsewhere route through `is_feasible`, which calls it.
def _min_pairwise(pos: torch.Tensor, min_d: float) -> torch.Tensor:
    n = pos.shape[1]
    dist = torch.cdist(pos, pos, p=2)
    mask = torch.triu(torch.ones(n, n, device=pos.device), diagonal=1).bool()
    violation = (min_d - dist).clamp(min=0.0)
    return violation[:, mask].sum(dim=-1)


def project_soft(
    layouts: NDArray[np.float64],
    scenario: ScenarioConfig,
    *,
    iters: int = 200,
    lr: float = 0.05,
    anchor: float = 0.05,
) -> NDArray[np.float64]:
    """Adam descent on min-distance violations, projected back into bounds each step.

    Scale-free: positions are divided by ``min_distance`` so hyperparameters do
    not depend on map size. Not guaranteed feasible (soft) -- use
    :func:`project_slsqp` when strict feasibility is required.
    """
    min_d = scenario.min_distance_between_turbines
    hi_x = scenario.map_x_length / min_d
    hi_y = scenario.map_y_length / min_d
    pos0 = torch.as_tensor(layouts, dtype=torch.float32) / min_d
    pos = pos0.clone().requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        penalty = _min_pairwise(pos, _MARGIN).sum()
        penalty = penalty + anchor * (pos - pos0).norm(dim=-1).sum()
        penalty.backward()
        opt.step()
        with torch.no_grad():
            pos[..., 0].clamp_(min=0.0, max=hi_x)
            pos[..., 1].clamp_(min=0.0, max=hi_y)
    return (pos.detach() * min_d).numpy().astype(np.float64)


def project_slsqp(
    layouts: NDArray[np.float64],
    scenario: ScenarioConfig,
    *,
    max_iter: int = 100,
    ftol: float = 1e-4,
) -> NDArray[np.float64]:
    """Per-layout Euclidean projection onto the feasible set via SLSQP.

    Minimises displacement from the input subject to pairwise ``>= min_distance``
    and box bounds. Falls back to the input layout for any sample the solver
    fails to converge.
    """
    min_d = scenario.min_distance_between_turbines * _MARGIN
    batch, n, _ = layouts.shape
    out = layouts.copy()

    def distance(i: int, j: int) -> Callable[[NDArray[np.float64]], float]:
        def constr(x: NDArray[np.float64]) -> float:
            d = x[2 * i : 2 * i + 2] - x[2 * j : 2 * j + 2]
            return float(np.linalg.norm(d) - min_d)

        return constr

    constraints = [
        {"type": "ineq", "fun": distance(i, j)}
        for i in range(n)
        for j in range(i + 1, n)
    ]
    bounds = [(0.0, scenario.map_x_length), (0.0, scenario.map_y_length)] * n

    for b in range(batch):
        x0 = layouts[b].flatten()
        res = minimize(
            lambda x, x0=x0: float(np.sum((x - x0) ** 2)),
            x0=x0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"disp": False, "ftol": ftol, "maxiter": max_iter},
        )
        if res.success:
            out[b] = np.asarray(res.x, dtype=np.float64).reshape(n, 2)
    return out

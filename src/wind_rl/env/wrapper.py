"""TorchRL wrapper turning the designable wfcrl env into a co-design env.

:class:`WfcrlCoDesignWrapper` adapts :class:`DesignableWindFarmEnv` (wrapped as a
PettingZoo *parallel* env) to torchrl 0.11's :class:`PettingZooWrapper`, and adds
the co-design plumbing:

* a **layout override** — either a stored ``(N, 2)`` tensor set via
  :meth:`set_layout_override`, or an optional ``reset_policy`` module that
  samples one — is forwarded to the underlying env as ``reset`` options so the
  MDP is rebuilt around it;
* the global ``state()`` (including the turbine ``"layout"``) is injected into the
  output tensordict on both reset and step.

Layout-override delivery (design note)
--------------------------------------
torchrl's :class:`~torchrl.envs.TransformedEnv` strips unknown keys from a reset
tensordict before it reaches the base env, so a per-reset ``"layout_override"``
tensordict field does **not** survive through the transform stack. We therefore
use a **stored attribute** (:meth:`set_layout_override`) as the reliable channel;
a ``"layout_override"`` field in the reset tensordict is still honoured when the
wrapper is driven directly (e.g. in tests), but the stored attribute takes
priority. Precedence on reset: stored override -> tensordict field ->
``reset_policy`` -> no rebuild (keep current layout).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import PettingZooWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend

#: Reset-policy output key holding the sampled ``(N, 2)`` layout.
LAYOUT_WEIGHTS_KEY = ("environment_design", "layout_weights")


class WfcrlCoDesignWrapper(PettingZooWrapper):  # type: ignore[misc]
    """PettingZoo -> torchrl wrapper with layout override and state injection."""

    def __init__(
        self,
        env: Any = None,
        *,
        reset_policy: TensorDictModule | None = None,
        categorical_actions: bool = False,
        **kwargs: Any,
    ) -> None:
        self._reset_policy = reset_policy
        self._layout_override: torch.Tensor | None = None
        self._wind_override: tuple[float, float] | None = None
        # ``return_state=False``: torchrl's base ``_reset``/``_step`` would call
        # ``torch.as_tensor(self.state())`` which fails on our dict-valued state.
        # We build the (Composite) state spec and inject the state ourselves.
        super().__init__(
            env,
            return_state=False,
            use_mask=False,
            categorical_actions=categorical_actions,
            **kwargs,
        )

    def set_layout_override(self, layout: torch.Tensor | NDArray[Any] | None) -> None:
        """Set (or clear, with ``None``) the ``(N, 2)`` layout used on next reset."""
        if layout is None:
            self._layout_override = None
        else:
            self._layout_override = torch.as_tensor(layout, dtype=torch.float32)

    def set_wind_override(self, wind: tuple[float, float] | None) -> None:
        """Fix (or clear, with ``None``) the reset wind as ``(direction_deg, speed_ms)``."""
        self._wind_override = wind

    def _make_specs(self, env: Any) -> None:
        super()._make_specs(env)
        # Build a proper Composite spec for the (dict-valued) global state.
        # ``set_gym_backend`` selects the gymnasium spec-conversion path.
        with set_gym_backend("gymnasium"):
            state_spec = _gym_to_torchrl_spec_transform(
                self._env.state_space,
                remap_state_to_observation=False,
                device=self.device,
            )
        self.observation_spec["state"] = state_spec

    def _resolve_layout(self, tensordict: TensorDictBase | None) -> NDArray[Any] | None:
        if self._layout_override is not None:
            return self._layout_override.numpy(force=True)
        if tensordict is not None and "layout_override" in tensordict.keys():  # noqa: SIM118
            layout = tensordict.get("layout_override")
            return np.asarray(layout.numpy(force=True))
        if self._reset_policy is not None:
            source = tensordict if tensordict is not None else TensorDict({}, [])
            out = self._reset_policy(source.to(self.device))
            return np.asarray(out.get(LAYOUT_WEIGHTS_KEY).numpy(force=True))
        return None

    def _state_tensordict(self) -> TensorDict:
        raw_state = self._env.state()
        state_spec = self.observation_spec["state"]
        state = TensorDict({}, batch_size=[], device=self.device)
        for key, value in raw_state.items():
            tensor = torch.as_tensor(np.asarray(value), device=self.device)
            state.set(key, tensor.to(dtype=state_spec[key].dtype))
        return state

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs: Any
    ) -> TensorDictBase:
        theta = self._resolve_layout(tensordict)
        options: dict[str, Any] = {}
        if theta is not None:
            options["xcoords"] = theta[:, 0]
            options["ycoords"] = theta[:, 1]
        if self._wind_override is not None:
            options["wind_direction"], options["wind_speed"] = self._wind_override
        out = super()._reset(tensordict, options=options or None, **kwargs)
        out.set("state", self._state_tensordict())
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = super()._step(tensordict)
        out.set("state", self._state_tensordict())
        return out

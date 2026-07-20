"""TorchRL wrapper turning the designable wfcrl env into a co-design env.

:class:`WfcrlCoDesignWrapper` adapts :class:`DesignableWindFarmEnv` (wrapped as a
PettingZoo *parallel* env) to torchrl 0.11's :class:`PettingZooWrapper`, and adds
the co-design plumbing:

* a **layout override** — a stored ``(N, 2)`` tensor set via
  :meth:`set_layout_override`, a ``layout_consumer`` popped from the shared
  cross-process layout buffer (the parallel-collection designer path), or an
  optional ``reset_policy`` module that samples one — is forwarded to the
  underlying env as ``reset`` options so the MDP is rebuilt around it;
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
``layout_consumer`` (buffer pop) -> ``reset_policy`` -> no rebuild (keep current
layout).
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import Unbounded
from torchrl.envs import PettingZooWrapper
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, set_gym_backend

from wind_rl.design.base import LAYOUT_WEIGHTS_KEY
from wind_rl.design.buffer import LayoutConsumer
from wind_rl.env.windfarm import ResetOptions


class WfcrlCoDesignWrapper(PettingZooWrapper):  # type: ignore[misc]
    """PettingZoo -> torchrl wrapper with layout override and state injection."""

    def __init__(
        self,
        env: object | None = None,
        *,
        reset_policy: TensorDictModule | None = None,
        layout_consumer: LayoutConsumer | None = None,
        categorical_actions: bool = False,
        **kwargs: object,
    ) -> None:
        # Stored as a plain attribute (never a tracked submodule): as an
        # ``nn.Module`` this env would reject a ``TensorDictModule`` assignment
        # before ``Module.__init__`` runs, and torchrl's ``EnvBase.__getattr__``
        # delegates to the wrapped env instead of resolving ``_modules`` -- a
        # registered submodule would then be unreachable at reset time.
        object.__setattr__(self, "_reset_policy", reset_policy)
        self._layout_consumer = layout_consumer
        self._layout_override: torch.Tensor | None = None
        self._wind_override: tuple[float, float] | None = None
        # Populated in ``_make_specs``; cached so the per-step state build (see
        # ``_state_tensordict``) doesn't re-index the Composite state spec (which
        # rebuilds a sub-spec per key) on every reset/step.
        self._state_dtypes: dict[str, torch.dtype] = {}
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

    @property
    def designable_env(self) -> object:
        """The wrapped :class:`DesignableWindFarmEnv` (through the PettingZoo hop).

        Exposed so the visualiser can reach the live FLORIS interface for
        wake-resolved rendering. ``self._env`` is an ``aec_to_parallel`` wrapper;
        its ``aec_env`` is the designable env.
        """
        return self._env.aec_env

    def set_layout_override(
        self, layout: torch.Tensor | NDArray[np.float64] | None
    ) -> None:
        """Set (or clear, with ``None``) the ``(N, 2)`` layout used on next reset."""
        if layout is None:
            self._layout_override = None
        else:
            self._layout_override = torch.as_tensor(layout, dtype=torch.float32)

    def set_wind_override(self, wind: tuple[float, float] | None) -> None:
        """Fix (or clear, with ``None``) the reset wind as ``(direction_deg, speed_ms)``."""
        self._wind_override = wind

    def farm_power(self) -> float:
        """Total farm power (MW) from the last step's per-turbine infos; 0 pre-step.

        wfcrl stores per-turbine power (already divided to MW) in each agent's
        ``info["power"]`` after every joint step; the farm total is their sum.
        Read raw so the unnormalised episode power is observable alongside the
        ``power/u_inf^3 - load`` reward the policy optimises.
        """
        infos = getattr(self.designable_env, "infos", {})
        return float(sum(float(info.get("power", 0.0)) for info in infos.values()))

    def _make_specs(self, env: object) -> None:
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
        self._state_dtypes = {key: spec.dtype for key, spec in state_spec.items()}
        # Farm-level raw power (MW), injected each step so the unnormalised
        # episode power travels through rollouts as a plain observation.
        self.observation_spec["power"] = Unbounded(
            shape=torch.Size([1]), device=self.device
        )

    def _resolve_layout(
        self, tensordict: TensorDictBase | None
    ) -> NDArray[np.float32] | None:
        if self._layout_override is not None:
            return self._layout_override.numpy(force=True)
        if tensordict is not None and "layout_override" in tensordict.keys():  # noqa: SIM118
            layout = tensordict.get("layout_override")
            return np.asarray(layout.numpy(force=True))
        if self._layout_consumer is not None:
            popped = self._layout_consumer.pop()
            if popped is not None:
                return popped.astype(np.float32)
        if self._reset_policy is not None:
            source = tensordict if tensordict is not None else TensorDict({}, [])
            out = self._reset_policy(source.to(self.device))
            return np.asarray(out.get(LAYOUT_WEIGHTS_KEY).numpy(force=True))
        return None

    def _state_tensordict(self) -> TensorDict:
        raw_state = self._env.state()
        state = TensorDict({}, batch_size=[], device=self.device)
        for key, value in raw_state.items():
            tensor = torch.as_tensor(np.asarray(value), device=self.device)
            state.set(key, tensor.to(dtype=self._state_dtypes[key]))
        return state

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs: object
    ) -> TensorDictBase:
        theta = self._resolve_layout(tensordict)
        options: ResetOptions = {}
        if theta is not None:
            options["xcoords"] = theta[:, 0]
            options["ycoords"] = theta[:, 1]
        if self._wind_override is not None:
            options["wind_direction"], options["wind_speed"] = self._wind_override
        out = super()._reset(tensordict, options=options or None, **kwargs)
        out.set("state", self._state_tensordict())
        out.set("power", torch.zeros(1, device=self.device))
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = super()._step(tensordict)
        out.set("state", self._state_tensordict())
        out.set("power", torch.tensor([self.farm_power()], device=self.device))
        return out

"""MLP actor/critic for the shared-parameter ``turbine`` agent group.

Ports DiCoDe's ``model_type="mlp"`` path to torchrl 0.11:

* a feature module engineering the per-agent observation (wind direction/speed,
  yaw, layout) into a 5-vector — cartesian wind, yaw in radians, map-normalised
  layout — so the raw metre-scale coordinates do not dominate the input;
* a :class:`~torchrl.modules.MultiAgentMLP` policy with a learned per-action
  ``log_std`` -> :class:`~tensordict.nn.distributions.NormalParamExtractor` ->
  :class:`~torchrl.modules.ProbabilisticActor` over a :class:`TanhNormal` yaw
  action (``share_params=True``);
* a centralised :class:`MultiAgentMLP` critic emitting one ``state_value`` per
  agent from the pooled group observation.
"""

from __future__ import annotations

from typing import Literal, override

import torch
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import Tensor, nn
from torchrl.envs import EnvBase
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from wind_rl.config import Config
from wind_rl.scenario import ScenarioConfig
from wind_rl.static import GROUP_NAME

_OBS_FEATURES = ("wind_direction", "wind_speed", "yaw", "layout")
_OBS_KEYS = [(GROUP_NAME, "observation", name) for name in _OBS_FEATURES]
_OBS_VEC_KEY = (GROUP_NAME, "obs_vec")
_LOC_KEY = (GROUP_NAME, "loc")
_SCALE_KEY = (GROUP_NAME, "scale")
_VALUE_KEY = (GROUP_NAME, "state_value")
_LOG_PROB_KEY = (GROUP_NAME, "sample_log_prob")

#: Width of the engineered per-agent observation vector (see :class:`_ObservationFeatures`).
_FEATURE_DIM = 5
#: Leading (cartesian wind_x, wind_y) entries of the feature vector built by
#: :class:`_ObservationFeatures`; consumers that reason about the wind direction
#: (e.g. :mod:`wind_rl.models.transformer`'s wind-frame canonicalisation) slice
#: this out.
_WIND_SLICE = slice(0, 2)
#: Trailing (layout_x, layout_y) entries of the feature vector built by
#: :class:`_ObservationFeatures`; consumers that need turbine positions (e.g.
#: :mod:`wind_rl.models.gnn`'s graph construction) slice this out.
_POS_SLICE = slice(_FEATURE_DIM - 2, _FEATURE_DIM)


class MlpModelConfig(Config):
    kind: Literal["mlp"] = "mlp"
    num_cells: int = 64
    depth: int = 2
    initial_std: float = 1.0
    share_params: bool = True


class _ObservationFeatures(nn.Module):
    def __init__(
        self,
        map_x_length: float,
        map_y_length: float,
        wind_speed_low: float,
        wind_speed_high: float,
    ) -> None:
        super().__init__()
        self.map_x_length = map_x_length
        self.map_y_length = map_y_length
        self.wind_speed_low = wind_speed_low
        self.wind_speed_high = wind_speed_high

    @override
    def forward(
        self,
        wind_direction: Tensor,
        wind_speed: Tensor,
        yaw: Tensor,
        layout: Tensor,
    ) -> Tensor:
        wind_speed = (wind_speed - self.wind_speed_low) / (
            self.wind_speed_high - self.wind_speed_low
        )
        wind_direction = torch.deg2rad(wind_direction)
        wind = torch.cat(
            [
                wind_speed * torch.cos(wind_direction),
                wind_speed * torch.sin(wind_direction),
            ],
            dim=-1,
        )
        yaw = torch.deg2rad(yaw)
        layout_x = layout[..., 0:1] / self.map_x_length * 2 - 1
        layout_y = layout[..., 1:2] / self.map_y_length * 2 - 1
        return torch.cat([wind, yaw, layout_x, layout_y], dim=-1)


class _GaussianParams(nn.Module):
    def __init__(self, net: MultiAgentMLP, action_dim: int) -> None:
        super().__init__()
        self.net = net
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    @override
    def forward(self, features: Tensor) -> Tensor:
        loc = self.net(features)
        return torch.cat([loc, torch.ones_like(loc) * self.log_std], dim=-1)


def _feature_module(scenario: ScenarioConfig, env: EnvBase) -> TensorDictModule:
    wind_speed_spec = env.observation_spec[GROUP_NAME, "observation", "wind_speed"]
    features = _ObservationFeatures(
        map_x_length=scenario.map_x_length,
        map_y_length=scenario.map_y_length,
        wind_speed_low=float(wind_speed_spec.low.min()),
        wind_speed_high=float(wind_speed_spec.high.max()),
    )
    return TensorDictModule(features, in_keys=_OBS_KEYS, out_keys=[_OBS_VEC_KEY])


def build_mlp_actor_critic(
    env: EnvBase,
    scenario: ScenarioConfig,
    cfg: MlpModelConfig,
    device: str | torch.device,
) -> tuple[ProbabilisticActor, TensorDictSequential]:
    """Build the (policy, critic) pair for ``env``, initialised on a reset tensordict."""
    feature_module = _feature_module(scenario, env)
    action_dim = env.full_action_spec[env.action_key].shape[-1]

    policy_head = nn.Sequential(
        _GaussianParams(
            MultiAgentMLP(
                n_agent_inputs=_FEATURE_DIM,
                n_agent_outputs=action_dim,
                n_agents=env.num_agents,
                centralized=False,
                share_params=cfg.share_params,
                depth=cfg.depth,
                num_cells=cfg.num_cells,
                activation_class=nn.Tanh,
            ),
            action_dim,
        ),
        NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.initial_std}", scale_lb=0.01
        ),
    )
    policy_body = TensorDictSequential(
        feature_module,
        TensorDictModule(
            policy_head, in_keys=[_OBS_VEC_KEY], out_keys=[_LOC_KEY, _SCALE_KEY]
        ),
        selected_out_keys=[_LOC_KEY, _SCALE_KEY],
    )
    action_space = env.full_action_spec_unbatched[env.action_key].space
    policy = ProbabilisticActor(
        module=policy_body,
        spec=env.action_spec_unbatched,
        in_keys=[_LOC_KEY, _SCALE_KEY],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        default_interaction_type=InteractionType.RANDOM,
        distribution_kwargs={"low": action_space.low, "high": action_space.high},
        return_log_prob=True,
        log_prob_key=_LOG_PROB_KEY,
    ).to(device)

    critic = TensorDictSequential(
        feature_module,
        # Pools per-agent obs_vec (centralized=True); deliberately does not read
        # the injected ("state", ...) key, so it is not state-conditioned.
        TensorDictModule(
            MultiAgentMLP(
                n_agent_inputs=_FEATURE_DIM,
                n_agent_outputs=1,
                n_agents=env.num_agents,
                centralized=True,
                share_params=True,
                depth=cfg.depth,
                num_cells=cfg.num_cells,
                activation_class=nn.Tanh,
            ),
            in_keys=[_OBS_VEC_KEY],
            out_keys=[_VALUE_KEY],
        ),
        selected_out_keys=[_VALUE_KEY],
    ).to(device)

    reset_td = env.reset().to(device)
    with torch.no_grad():
        policy(reset_td)
        critic(reset_td)

    return policy, critic

from windrl_engine.env import WindFarmEnvConfig
from windrl_train.config import Config


class IPPOConfig(Config):
    """Independent-PPO hyperparameters and the env they train against."""

    env: WindFarmEnvConfig
    total_timesteps: int = 1_000_000  # env steps summed over envs
    rollout_length: int = 128
    ppo_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.001
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 0.5
    width: int = 64
    depth: int = 2
    eval_every_updates: int = 10
    eval_steps: int = 512  # deterministic steps per eval (env auto-resets)

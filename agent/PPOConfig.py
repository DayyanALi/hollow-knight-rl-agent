from dataclasses import dataclass

@dataclass
class PPOConfig:
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    lr: float = 3e-4
    epochs_per_update: int = 4
    rollout_steps: int = 2048
    mini_batch_size: int = 64
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    
    # Environment
    num_actions: int = 18
    frame_stack: int = 4
    frame_size: int = 84
    action_duration: float = 0.1
    window_title: str = "Hollow Knight"
    
    # Training
    total_timesteps: int = 10_000_000
    save_interval: int = 1  # save every N updates
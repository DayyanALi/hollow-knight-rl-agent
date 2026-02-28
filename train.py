import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
from agent.action import run_recovery_macro
import os
import csv
import logging

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from agent.model import ActorCritic
from agent.PPOConfig import PPOConfig
from environment.get_env import GameEnvironment
from environment.rollout_buffer import RolloutBuffer


class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[PPO] Using device: {self.device}")
        
        # Actor-Critic network
        self.policy = ActorCritic(
            input_channels=config.frame_stack,
            num_actions=config.num_actions
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        
        # Environment
        self.env = GameEnvironment(config)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Logging
        self.total_steps = 0
        self.episode_count = 0
        self.episode_rewards = []
        
        # CSV Logging Setup
        file_exists = os.path.isfile("training_metrics.csv")
        self.csv_file = open("training_metrics.csv", mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(["update", "total_steps", "policy_loss", "value_loss", "entropy", "avg_reward"])
            
        # Auto-resume from latest checkpoint
        self._load_latest_checkpoint()

    def _load_latest_checkpoint(self, path="checkpoints"):
        """Find and load the latest checkpoint in the directory."""
        if not os.path.exists(path):
            return
            
        checkpoints = [f for f in os.listdir(path) if f.startswith("ppo_step_") and f.endswith(".pth")]
        if not checkpoints:
            return
            
        # Parse step numbers from filenames to find the latest
        latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
        filepath = os.path.join(path, latest_cp)
        
        logger.info(f"Auto-resuming from previous run. Found {len(checkpoints)} checkpoints.")
        self.load(filepath)
    
    def collect_rollout(self):
        """Run the policy for rollout_steps and fill the buffer.
        
        Handles episode boundaries (death/win) by resetting the environment.
        """
        state = self.env.reset()
        episode_reward = 0.0
        
        for step in range(self.config.rollout_steps):
            # Move state to device for inference
            state_tensor = state.unsqueeze(0).to(self.device)
            
            # Get action from the policy
            with torch.no_grad():
                action, log_prob, value = self.policy.act(state_tensor)
            
            # Execute action in the game
            next_state, reward, done = self.env.step(action)
            
            episode_reward += reward
            
            # Store transition in buffer
            self.buffer.store(state, action, reward, done, log_prob, value)
            
            self.total_steps += 1
            
            if done:
                # Episode ended (player died or boss died)
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)
                logger.info(f"  Episode {self.episode_count} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(10): {avg_reward:.2f} | "
                      f"Steps: {self.total_steps}")
                
                # Reset for next episode
                episode_reward = 0.0
                
                # Wait for user to reset the boss fight
                logger.info("  Waiting 5 seconds for boss fight reset...")
                time.sleep(5)
                state = self.env.reset()
            else:
                state = next_state
        
        # Get value estimate for the last state (needed for GAE)
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.act(state_tensor)
        
        return last_value
    
    def update(self):
        """Run PPO update on the collected rollout."""
        # Normalize advantages (stabilizes training)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.numpy()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for epoch in range(self.config.epochs_per_update):
            for batch in self.buffer.get_mini_batches(self.config.mini_batch_size):
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                returns = batch["returns"].to(self.device)
                advantages_batch = batch["advantages"].to(self.device)
                
                # Forward pass through current policy
                action_probs, values = self.policy(states)
                values = values.squeeze(-1)
                
                # Create distribution and get new log probs
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # ---- Policy Loss (Clipped Surrogate) ----
                # Probability ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clipped and unclipped objectives
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 
                                    1.0 - self.config.clip_epsilon, 
                                    1.0 + self.config.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ---- Value Loss ----
                value_loss = nn.MSELoss()(values, returns)
                
                # ---- Total Loss ----
                loss = (policy_loss 
                        + self.config.value_loss_coeff * value_loss 
                        - self.config.entropy_coeff * entropy)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Log averages
        if num_updates > 0:
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            
            logger.info(f"  Update | "
                  f"Policy Loss: {avg_policy_loss:.4f} | "
                  f"Value Loss: {avg_value_loss:.4f} | "
                  f"Entropy: {avg_entropy:.4f}")
            
            return avg_policy_loss, avg_value_loss, avg_entropy
        return 0.0, 0.0, 0.0
    
    def save(self, path="checkpoints"):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"ppo_step_{self.total_steps}.pth")
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
        }, filepath)
        logger.info(f"  Saved checkpoint: {filepath}")
    
    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.episode_count = checkpoint["episode_count"]
        logger.info(f"  Loaded checkpoint: {filepath}")
    
    def train(self):
        """Main PPO training loop."""
        logger.info(f"[PPO] Starting training for {self.config.total_timesteps} timesteps")
        logger.info(f"[PPO] Rollout: {self.config.rollout_steps} steps | "
              f"Updates: {self.config.epochs_per_update} epochs | "
              f"Mini-batch: {self.config.mini_batch_size}")
        logger.info("-" * 60)
        
        update_count = 0
        
        while self.total_steps < self.config.total_timesteps:
            update_count += 1
            logger.info(f"\n--- Update {update_count} | Total Steps: {self.total_steps} ---")
            
            # Phase 1: Collect experience
            last_value = self.collect_rollout()
            
            # Phase 2: Compute returns and advantages
            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda
            )
            
            # Phase 3: PPO update
            p_loss, v_loss, ent = self.update()
            
            # Phase 4: Clear buffer
            self.buffer.clear()
            
            # Save metrics to CSV
            avg_reward = sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10) if self.episode_rewards else 0.0
            self.csv_writer.writerow([update_count, self.total_steps, p_loss, v_loss, ent, avg_reward])
            self.csv_file.flush()
            
            # Save checkpoint periodically
            if update_count % self.config.save_interval == 0:
                self.save()
        
        # Final save
        self.save()
        logger.info(f"\n[PPO] Training complete! Total steps: {self.total_steps}")


if __name__ == "__main__":
    time.sleep(5)
    config = PPOConfig()
    run_recovery_macro()
    trainer = PPOTrainer(config)
    trainer.train()
import torch
import numpy as np


class RolloutBuffer:
    """Stores experience from one rollout and computes GAE advantages."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # Computed after rollout
        self.returns = None
        self.advantages = None
    
    def store(self, state, action, reward, done, log_prob, value):
        """Store a single transition.
        
        Args:
            state: torch tensor (frame_stack, 84, 84)
            action: int, action taken
            reward: float
            done: bool, episode ended
            log_prob: float, log probability of the action under the policy
            value: float, critic's value estimate V(s)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and discounted returns.
        
        Args:
            last_value: V(s_last) from the critic for the final state
            gamma: discount factor
            gae_lambda: GAE smoothing parameter
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        
        # GAE computation (working backwards)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            # If episode ended at step t, next_value should be 0
            next_non_terminal = 1.0 - float(self.dones[t])
            
            # TD error: δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE: Aₜ = δₜ + (γλ)·Aₜ₊₁
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Returns = advantages + values
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)
    
    def get_mini_batches(self, mini_batch_size):
        """Yield shuffled mini-batches of experience as tensors.
        
        Yields:
            dict with keys: states, actions, log_probs, returns, advantages
        """
        n = len(self.states)
        indices = np.random.permutation(n)
        
        for start in range(0, n, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            
            yield {
                "states": torch.stack([self.states[i] for i in batch_indices]),
                "actions": torch.tensor([self.actions[i] for i in batch_indices], dtype=torch.long),
                "log_probs": torch.tensor([self.log_probs[i] for i in batch_indices], dtype=torch.float32),
                "returns": torch.tensor(self.returns[batch_indices], dtype=torch.float32),
                "advantages": torch.tensor(self.advantages[batch_indices], dtype=torch.float32),
            }
    
    def clear(self):
        """Clear all stored data for the next rollout."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.returns = None
        self.advantages = None
    
    def __len__(self):
        return len(self.states)

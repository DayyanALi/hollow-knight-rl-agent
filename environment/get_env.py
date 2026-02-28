import numpy as np
import torch
from collections import deque

from environment.capture import ScreenCapture
from agent.action import execute_action
from agent.action import run_recovery_macro

import time
import socket
import json
import threading

# UDP socket setup to listen to the C# BepInEx mod
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.1)

# Global variables holding the most recently received game state
latest_player_hp = 5  # Default max masks
latest_boss_hp = 0

def listen_for_telemetry():
    global latest_player_hp, latest_boss_hp
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            state = json.loads(data.decode("utf-8"))
            latest_player_hp = state.get("player_hp", latest_player_hp)
            latest_boss_hp = state.get("boss_hp", latest_boss_hp)
        except socket.timeout:
            pass  # Expected if game isn't sending data right now
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[Telemetry] UDP Error: {e}")

# Start the background listener thread
listener_thread = threading.Thread(target=listen_for_telemetry, daemon=True)
listener_thread.start()

def read_player_hp():
    """Returns the most recent player HP received from the UDP socket."""
    return latest_player_hp

def read_boss_hp():
    """Returns the most recent boss HP received from the UDP socket."""
    return latest_boss_hp


class GameEnvironment:
    def __init__(self, config):
        """
        Args:
            config: PPOConfig instance
        """
        self.config = config
        self.num_actions = config.num_actions
        self.frame_stack_size = config.frame_stack
        self.action_duration = config.action_duration
        
        # Screen capture
        self.capture = ScreenCapture(
            window_title=config.window_title,
            width=config.frame_size,
            height=config.frame_size
        )
        
        # Frame stacking
        self.frame_stack = deque(maxlen=self.frame_stack_size)
        
        # HP tracking for reward computation
        self.prev_player_hp = None
        self.prev_boss_hp = None
    
    def reset(self):
        """Reset environment for a new episode.
        
        Call this when starting a new boss attempt.
        Returns the initial state as a torch tensor.
        """
        self.frame_stack.clear()
        
        # 1. Wait for player to respawn (HP > 0)
        print("Waiting for player to respawn...")
        # while read_player_hp() <= 0:
        import time
        time.sleep(3)
            
        # 2. Run the automated recovery macro
        run_recovery_macro()
        
        # 3. Wait until the boss fight actually triggers
        # print("Waiting for Boss health bar to appear...")
        # while read_boss_hp() <= 0:
        #     import time
        #     time.sleep(0.2)
            
        print(f"Boss fight started! Initial HP -> Player: {read_player_hp()}, Boss: {read_boss_hp()}")
        
        # Read initial HP values
        self.prev_player_hp = read_player_hp()
        self.prev_boss_hp = read_boss_hp()
        
        # Fill frame stack with initial frames
        state = self._get_state()
        return state
    
    def step(self, action_id):
        """Execute an action and return the result.
        
        Args:
            action_id: integer 0-8 corresponding to an action
            
        Returns:
            next_state: torch tensor of shape (frame_stack, 84, 84)
            reward: float reward signal
            done: bool, True if episode ended (player or boss died)
        """
        # 1. Execute the action in the game
        execute_action(action_id)
        
        # 2. Capture new state
        next_state = self._get_state()
        
        # 3. Read game memory for HP
        player_hp = read_player_hp()
        boss_hp = read_boss_hp()
        
        # 4. Compute reward
        reward = self._compute_reward(player_hp, boss_hp)
        
        # 5. Check if episode is over
        done = (player_hp <= 0) or (boss_hp <= 0)
        
        # 6. Update HP tracking
        self.prev_player_hp = player_hp
        self.prev_boss_hp = boss_hp
        
        return next_state, reward, done
    
    def _get_state(self):
        """Capture a frame and return stacked state as tensor.
        
        Returns:
            torch tensor of shape (frame_stack, 84, 84)
        """
        frame = self.capture.capture()  # (84, 84) numpy float32
        self.frame_stack.append(frame)
        
        # Pad with zeros if we don't have enough frames yet
        while len(self.frame_stack) < self.frame_stack_size:
            self.frame_stack.appendleft(np.zeros_like(frame))
        
        # Stack frames into (4, 84, 84) and convert to tensor
        stacked = np.stack(list(self.frame_stack), axis=0)
        return torch.from_numpy(stacked)
    
    def _compute_reward(self, player_hp, boss_hp):
        """Compute reward based on HP changes.
        
        Reward structure:
            - Deal damage to boss: +1.0 per HP point
            - Take damage: -1.0 per HP mask lost
            - Kill boss (win): +10.0 bonus
            - Player dies (lose): -5.0 penalty
            - Time penalty: -0.01 per step (encourages aggression)
        """
        reward = 0.0
        
        # Reward for dealing damage to boss
        boss_damage = self.prev_boss_hp - boss_hp
        if boss_damage > 0:
            reward += boss_damage * 1.0
        
        # Penalty for taking damage, reward for healing
        hp_change = player_hp - self.prev_player_hp
        if hp_change < 0:
            # Took damage (hp_change is negative)
            reward += hp_change * 1.0  # -1.0 per HP lost
        elif hp_change > 0:
            # Healed (hp_change is positive)
            reward += hp_change * 1.0  # +1.0 per HP gained
        
        # Win/lose bonuses
        if boss_hp <= 0:
            reward += 10.0
        if player_hp <= 0:
            reward -= 5.0
        
        # Time penalty to encourage aggressive play
        reward -= 0.01
        
        return reward
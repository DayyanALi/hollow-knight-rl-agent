# Hollow Knight / Silksong PPO Agent

This project implements a Reinforcement Learning agent (using Proximal Policy Optimization) trained to play *Hollow Knight* and *Silksong* via screen capture and simulated controller inputs.

## Setup & Architecture
The agent uses a combination of visual input (via `mss` screen capture) and real-time game telemetry (via a custom BepInEx C# mod broadcasting UDP packets). 

1. **Modding:** A BepInEx mod is required to extract real-time player and boss HP directly from the Unity Engine since memory reading is highly unstable in modern games.
2. **Environment:** `get_env.py` manages the PyTorch state framing and calculates rewards based on the telemetry (e.g., +1 for dealing damage, -1 for taking damage, +1 for healing).
3. **Actions:** `action.py` emulates an Xbox 360 controller via the `vgamepad` library to bypass the Unity game engine's input blocking. It includes a custom macro (`run_recovery_macro()`) to automate the character's walk back from a Godhome respawn point to the boss arena door.
4. **Training:** `train.py` contains the PPO loop, recording trajectories in a `RolloutBuffer` and updating the Actor-Critic network. It aggressively checkpoints the model state every 2048 steps.

## How to Run
1. Ensure the BepInEx telemetry mod (`HKSMod.dll` / `HollowKnightMod.dll`) is installed in the game's `plugins/` directory and is actively broadcasting to port `5005`.
2. Teleport your character to a Godhome boss arena spawn or configure an override save point.
3. Run `python train.py`.

The agent will automatically wait out the death screen, execute the run-back macro, wait for the boss HP bar to appear, and then take over the controller to begin training!

## Logging & Visualization
Metrics are logged to the console, appended to `training_log.txt`, and mapped inside `training_metrics.csv` for easy plotting via Excel or Pandas. Checkpoints are deposited into the `/checkpoints` directory and automatically resume on subsequent executions.

import torch
from agent.model import ActorCritic
from agent.action import execute_action
import time
from agent.action import run_recovery_macro

def test_model():
    state = torch.randn(1, 4, 84, 84)
    model = ActorCritic(4, 9)
    action, action_log, value = model.act(state)
    print(action, action_log, value)

def test_keyboard():
    action_list = [8,8,8,8,3,3,4,2,2,2,3,8,3,8,3,8]
    print("executing")
    for action in action_list:
        execute_action(action)
        time.sleep(0.1)

# test_keyboard()

run_recovery_macro()

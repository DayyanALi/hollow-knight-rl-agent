import vgamepad as vg
import time

# Create simulated Xbox 360 controller
gamepad = vg.VX360Gamepad()

# Durations: short (quick tap) and long (sustained press)
SHORT = 0.1   # 100ms
LONG = 0.4    # 400ms

# Xbox Button Mappings
# Jump: A button
# Attack: X button
# Dash: Right Trigger (RT) - not used in this basic map but could be
# Heal: B button
# Movement: D-Pad Left / Right

BTN_JUMP = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
BTN_ATTACK = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
BTN_HEAL = vg.XUSB_BUTTON.XUSB_GAMEPAD_B
BTN_LEFT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
BTN_RIGHT = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT
BTN_DOWN = vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN

# 18 discrete actions: 9 actions × 2 durations
# Format: (buttons_list, duration)
ACTION_MAP = {
    # No-op
    0:  ([], SHORT),                          # no-op short
    1:  ([], LONG),                           # no-op long
    
    # Movement (D-pad)
    2:  ([BTN_LEFT], SHORT),                  # left short
    3:  ([BTN_LEFT], LONG),                   # left long
    4:  ([BTN_RIGHT], SHORT),                 # right short
    5:  ([BTN_RIGHT], LONG),                  # right long
    
    # Jump (A)
    6:  ([BTN_JUMP], SHORT),                  # jump short (short hop)
    7:  ([BTN_JUMP], LONG),                   # jump long (full jump)
    
    # Attack (X)
    8:  ([BTN_ATTACK], SHORT),                # attack short
    9:  ([BTN_ATTACK], LONG),                 # attack long
    
    # Heal / Focus (B)
    10: ([BTN_HEAL], SHORT),                  # heal short
    11: ([BTN_HEAL], LONG),                   # heal long
    
    # Combos
    12: ([BTN_JUMP, BTN_LEFT], SHORT),        # jump + left short
    13: ([BTN_JUMP, BTN_LEFT], LONG),         # jump + left long
    14: ([BTN_JUMP, BTN_RIGHT], SHORT),       # jump + right short
    15: ([BTN_JUMP, BTN_RIGHT], LONG),        # jump + right long
    16: ([BTN_DOWN, BTN_ATTACK], SHORT),      # down + attack short (pogo)
    17: ([BTN_DOWN, BTN_ATTACK], LONG),       # down + attack long (pogo)
}

NUM_ACTIONS = len(ACTION_MAP)


def execute_action(action_id):
    """Press the buttons on the simulated Xbox controller, hold, then release."""
    buttons, duration = ACTION_MAP.get(action_id, ([], SHORT))
    
    if not buttons:
        # No-op: just wait
        time.sleep(duration)
        return
    
    # Press all buttons
    for btn in buttons:
        gamepad.press_button(button=btn)
    gamepad.update()  # Send the input state to the OS
    
    # Hold for duration
    time.sleep(duration)
    
    # Release all buttons
    for btn in buttons:
        gamepad.release_button(button=btn)
    gamepad.update()  # Send the new input state to the OS


def run_recovery_macro(movement_time=1.7):
    """
    Automated macro to walk from Godhome spawn to the Boss trigger.
    Sequence: Right 1.7s -> Jump Right -> Jump Left
    """
    print("  [Macro] Walking right...")
    
    # 1. Move Right for 1.7 seconds
    gamepad.press_button(button=BTN_RIGHT)
    gamepad.update()
    time.sleep(movement_time) 
    
    # 2. Jump Right (We are already holding Right)
    print("  [Macro] Jumping Right...")
    gamepad.press_button(button=BTN_JUMP)
    gamepad.update()
    time.sleep(0.9) # hold jump to get max distance right
    
    # Release both Right and Jump
    gamepad.release_button(button=BTN_JUMP)
    gamepad.release_button(button=BTN_RIGHT)
    gamepad.update()
    
    time.sleep(0.2) # small pause before jumping left

    # 3. Jump Left 
    print("  [Macro] Jumping Left...")
    gamepad.press_button(button=BTN_LEFT)
    gamepad.press_button(button=BTN_JUMP)
    gamepad.update()
    time.sleep(1.5) # hold jump to get max distance left
    
    # Release both Left and Jump
    gamepad.release_button(button=BTN_JUMP)
    gamepad.release_button(button=BTN_LEFT)
    gamepad.update()

    time.sleep(0.2) # small pause before jumping left

    # 4. Jump Left 
    print("  [Macro] Jumping Left...")
    gamepad.press_button(button=BTN_LEFT)
    gamepad.press_button(button=BTN_JUMP)
    gamepad.update()
    time.sleep(1) # hold jump to get max distance left
    
    # Release both Left and Jump
    gamepad.release_button(button=BTN_JUMP)
    gamepad.release_button(button=BTN_LEFT)
    gamepad.update()
    
    time.sleep(0.2) # small pause before jumping left
    
    # Move left 
    gamepad.press_button(button=BTN_LEFT)
    gamepad.update()
    time.sleep(4)
    gamepad.release_button(button=BTN_LEFT)
    gamepad.update()
    
    print("  [Macro] Finished walk. Waiting for boss HP to appear...")


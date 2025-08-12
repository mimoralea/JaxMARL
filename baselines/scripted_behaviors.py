#!/usr/bin/env python3
"""
Original Scripted Behaviors Module

This module provides the original, sophisticated scripted behavior implementations
from the existing JaxMARL codebase. These are proper sumo strategies with complex
logic for position analysis, finite state machines, and arena awareness.

Supported Behaviors:
- noop: No-operation (stationary)
- random: Random action selection
- seek: Complex FSM with chase/retreat modes and position/velocity analysis
- guardian: Defensive strategy staying near center with safety radius
- dodge: Orbital movement with safety bounds and tangential motion

Usage:
    from baselines.scripted_behaviors import get_scripted_action, list_scripted_behaviors

    # Get available behaviors
    behaviors = list_scripted_behaviors()

    # Get action from a behavior
    action = get_scripted_action(obs, "seek", rng_key)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, Callable


def get_scripted_agent(agent_name: str, seed: int = 0) -> Callable:
    """Get a scripted agent policy function - original implementations from eval_arena.py.

    Args:
        agent_name: Name of the scripted behavior
        seed: Random seed for stochastic behaviors

    Returns:
        Policy function that takes observation and returns action
    """

    if agent_name == "noop":
        return lambda obs: jnp.array(0)  # Action 0: NO-OP

    elif agent_name == "random":
        rng = jax.random.PRNGKey(seed)
        def random_policy(obs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return jax.random.randint(key, (), 0, 5)
        return random_policy

    elif agent_name == "seek":
        # Complex FSM with chase/retreat modes - exact reimplementation from eval_arena.py
        mode = "chase"  # persistent FSM state
        def seek_policy(obs):
            nonlocal mode
            sx, sy = obs[0], obs[1]
            dx = obs[4] - sx
            dy = obs[5] - sy
            vx, vy = obs[2], obs[3]
            dist = jnp.sqrt(sx ** 2 + sy ** 2)
            # If moving outward, brake and steer inward
            outward_dot = sx * vx + sy * vy
            # Alignment with target (opponent)
            dir_dot = vx * dx + vy * dy
            # Switch modes based on position / velocity
            if mode == "chase" and ((dist > 0.25) or (outward_dot > 0)):
                mode = "retreat"
            elif mode == "retreat" and (dist < 0.15) and (outward_dot <= 0):
                mode = "chase"
            # Decide action
            if mode == "retreat":
                # Move back toward centre along dominant axis
                if jnp.abs(sx) > jnp.abs(sy):
                    action = 1 if sx > 0 else 2  # step toward x=0
                else:
                    action = 3 if sy > 0 else 4  # step toward y=0
                return jnp.array(action)
            # CHASE mode
            # If heading away from opponent, immediately correct course
            if dir_dot < 0:
                if jnp.abs(dx) > jnp.abs(dy):
                    action = 2 if dx > 0 else 1
                else:
                    action = 4 if dy > 0 else 3
                return jnp.array(action)
            # Otherwise chase opponent
            if jnp.abs(dx) > jnp.abs(dy):
                action = 2 if dx > 0 else 1
            else:
                action = 4 if dy > 0 else 3
            return jnp.array(action)
        return seek_policy

    elif agent_name == "guardian":
        # Defensive strategy staying near center - exact reimplementation from eval_arena.py
        SAFE_RAD = 0.15  # stay very close (<=0.15) to centre (arena R≈0.4)
        def guardian_policy(obs):
            sx, sy = obs[0], obs[1]
            ox, oy = obs[4], obs[5]
            self_dist = jnp.sqrt(sx ** 2 + sy ** 2)
            opp_dist = jnp.sqrt(ox ** 2 + oy ** 2)
            # If we are drifting outwards move back to centre
            if self_dist > SAFE_RAD * 0.9:
                # vector toward centre is (-sx, -sy)
                if jnp.abs(sx) > jnp.abs(sy):
                    action = 1 if sx > 0 else 2  # step inward in x
                else:
                    action = 3 if sy > 0 else 4  # inward in y
                return jnp.array(action)
            # Otherwise hold ground at centre; no aggressive pushing
            return jnp.array(0)
        return guardian_policy

    elif agent_name == "dodge":
        # Orbital movement with safety bounds - exact reimplementation from eval_arena.py
        rng = jax.random.PRNGKey(seed)
        steps_left = 0
        direction = 1  # +1 = CCW, -1 = CW

        # Environment constants from eval_arena.py
        ARENA_RADIUS = 0.4
        AGENT_RADIUS = 0.05
        SAFETY_MARGIN = 0.1
        MAX_SAFE_DIST = ARENA_RADIUS - AGENT_RADIUS - SAFETY_MARGIN  # 0.25
        INNER_BOUND = 0.15
        OUTER_BOUND = 0.20

        def dodge_policy(obs):
            nonlocal rng, steps_left, direction
            sx, sy = obs[0], obs[1]  # Self position
            ox, oy = obs[4], obs[5]  # Opponent position
            self_dist = jnp.sqrt(sx**2 + sy**2)
            opp_dist = jnp.sqrt(ox**2 + oy**2)

            # SAFETY FIRST: Check if we're outside the safe band
            if self_dist > OUTER_BOUND:
                # Too close to edge, move directly toward center
                if abs(sx) > abs(sy):
                    action = 1 if sx > 0 else 2  # left if on right, right if on left
                else:
                    action = 3 if sy > 0 else 4  # down if on top, up if on bottom
                steps_left = 0
                return jnp.array(action)
            elif self_dist < INNER_BOUND:
                # Too close to center, move directly outward
                if abs(sx) < abs(sy):
                    action = 2 if sx >= 0 else 1  # right or left
                else:
                    action = 4 if sy >= 0 else 3  # up or down
                steps_left = 0
                return jnp.array(action)

            # Normal orbit behavior - simplified tangential movement
            if steps_left <= 0:
                rng, subkey = jax.random.split(rng)
                direction = jax.random.choice(subkey, jnp.array([-1, 1]))
                rng, subkey = jax.random.split(rng)
                steps_left = jax.random.randint(subkey, (), 5, 15)

            steps_left -= 1

            # Simple tangential movement
            if abs(sx) > abs(sy):
                # Move along y-axis
                action = 3 if direction > 0 else 4  # down or up
            else:
                # Move along x-axis
                action = 2 if direction > 0 else 1  # right or left

            return jnp.array(action)
        return dodge_policy

    else:
        raise ValueError(f"Unknown scripted agent: {agent_name}")


def list_scripted_behaviors() -> Dict[str, str]:
    """List all available scripted behaviors.

    Returns:
        Dictionary mapping behavior names to descriptions
    """
    return {
        "noop": "No-operation (stationary)",
        "random": "Random action selection",
        "seek": "Complex FSM with chase/retreat modes and position/velocity analysis",
        "guardian": "Defensive strategy staying near center with safety radius",
        "dodge": "Orbital movement with safety bounds and tangential motion"
    }


def get_scripted_action(obs: jnp.ndarray, behavior_name: str,
                       rng_key: jax.random.PRNGKey,
                       opponent_obs: Optional[jnp.ndarray] = None) -> int:
    """Get action from a scripted behavior.

    Args:
        obs: Agent's observation
        behavior_name: Name of the behavior
        rng_key: Random key for stochastic behaviors
        opponent_obs: Opponent's observation (optional)

    Returns:
        Action as integer
    """
    # For compatibility, we need to maintain state across calls
    # This is a simplified version - full implementation would need proper state management
    if behavior_name == "noop":
        return 0
    elif behavior_name == "random":
        return int(jax.random.randint(rng_key, (), 0, 5))
    elif behavior_name == "seek":
        # Simplified seek behavior for stateless calls
        sx, sy = obs[0], obs[1]
        dx = obs[4] - sx
        dy = obs[5] - sy
        if jnp.abs(dx) > jnp.abs(dy):
            action = 2 if dx > 0 else 1
        else:
            action = 4 if dy > 0 else 3
        return int(action)
    elif behavior_name == "guardian":
        sx, sy = obs[0], obs[1]
        self_dist = jnp.sqrt(sx ** 2 + sy ** 2)
        SAFE_RAD = 0.15
        if self_dist > SAFE_RAD * 0.9:
            if jnp.abs(sx) > jnp.abs(sy):
                action = 1 if sx > 0 else 2
            else:
                action = 3 if sy > 0 else 4
            return int(action)
        return 0
    elif behavior_name == "dodge":
        # Simplified dodge behavior for stateless calls
        sx, sy = obs[0], obs[1]
        self_dist = jnp.sqrt(sx**2 + sy**2)
        INNER_BOUND = 0.15
        OUTER_BOUND = 0.20

        if self_dist > OUTER_BOUND:
            if abs(sx) > abs(sy):
                action = 1 if sx > 0 else 2
            else:
                action = 3 if sy > 0 else 4
            return int(action)
        elif self_dist < INNER_BOUND:
            if abs(sx) < abs(sy):
                action = 2 if sx >= 0 else 1
            else:
                action = 4 if sy >= 0 else 3
            return int(action)
        else:
            # Simple tangential movement
            if abs(sx) > abs(sy):
                action = 3  # down
            else:
                action = 2  # right
            return int(action)
    else:
        raise ValueError(f"Unknown scripted behavior: {behavior_name}")


if __name__ == "__main__":
    # Demo usage
    print("🤖 Available Scripted Behaviors:")
    behaviors = list_scripted_behaviors()
    for name, description in behaviors.items():
        print(f"  - {name}: {description}")

    # Test each behavior
    print("\n🧪 Testing behaviors:")
    rng_key = jax.random.PRNGKey(42)
    test_obs = jnp.array([1.0, 2.0, 3.0, 4.0])

    for name in behaviors.keys():
        rng_key, action_key = jax.random.split(rng_key)
        action = get_scripted_action(test_obs, name, action_key)
        print(f"  - {name}: action = {action}")

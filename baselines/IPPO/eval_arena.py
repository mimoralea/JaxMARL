"""Evaluation arena for MPE Sumo.

Allows loading arbitrary checkpoints (or scripted baselines) for each side and
produces a GIF of the rollout via the existing `MPEVisualizer`.
"""
from __future__ import annotations
import pickle
import time
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pprint

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _HAS_TQDM = False

import jaxmarl
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
from jaxmarl.environments.mpe.default_params import MAX_STEPS
try:
    from .ippo_ff_mpe import ActorCritic  # type: ignore
except ImportError:
    from baselines.IPPO.ippo_ff_mpe import ActorCritic

# -----------------------------------------------------------------------------
# Helper to load parameters or create scripted behaviour
# -----------------------------------------------------------------------------


def _load_params_or_baseline(path_or_baseline: str, env, agent_name: str, seed: int, activation: str = "tanh"):
    """Return a function(state, obs) -> action for the given side."""
    if path_or_baseline == "noop":
        print(f'Loading noop policy for {agent_name}')
        return lambda *_: jnp.array(0)  # action 0 assumed to be noop
    if path_or_baseline == "seek":
        print(f'Loading heuristic SEEK policy for {agent_name}')
        # Observation layout: [self_x, self_y, self_vx, self_vy, opp_x, opp_y, opp_vx, opp_vy]
        # Discrete actions: 0 noop, 1 left(-x), 2 right(+x), 3 down(-y), 4 up(+y)
        mode = "chase"  # persistent FSM state
        def _seek(_, obs: jnp.ndarray):
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
                # print(f"SEEK RETREAT: dist={float(dist):.2f} outward={float(outward_dot):.2f} action={action}")
                return jnp.array(action)
            # CHASE mode
            # If heading away from opponent, immediately correct course
            # print(f"SEEK CHASE: dist={float(dist):.2f} outward={float(outward_dot):.2f}")
            if dir_dot < 0:
                if jnp.abs(dx) > jnp.abs(dy):
                    action = jnp.array(2) if dx > 0 else jnp.array(1)
                else:
                    action = jnp.array(4) if dy > 0 else jnp.array(3)
                return action
            # Otherwise chase opponent
            if jnp.abs(dx) > jnp.abs(dy):
                return jnp.array(2) if dx > 0 else jnp.array(1)
            else:
                return jnp.array(4) if dy > 0 else jnp.array(3)
        return _seek

    if path_or_baseline == "centaur":
        print(f'Loading defensive CENTAUR policy for {agent_name}')
        SAFE_RAD = 0.15  # stay very close (<=0.15) to centre (arena R≈0.4)
        def _centaur(_, obs: jnp.ndarray):
            sx, sy = obs[0], obs[1]
            ox, oy = obs[4], obs[5]
            self_dist = jnp.sqrt(sx ** 2 + sy ** 2)
            opp_dist = jnp.sqrt(ox ** 2 + oy ** 2)
            # If we are drifting outwards move back to centre
            if self_dist > SAFE_RAD * 0.9:
                # vector toward centre is (-sx, -sy)
                if jnp.abs(sx) > jnp.abs(sy):
                    return jnp.array(1) if sx > 0 else jnp.array(2)  # step inward in x
                else:
                    return jnp.array(3) if sy > 0 else jnp.array(4)  # inward in y
            # Otherwise hold ground at centre; no aggressive pushing
            return jnp.array(0)
        return _centaur

    if path_or_baseline == "dodge":
        print(f'Loading evasive DODGE policy for {agent_name}')
        # Orbit behavior with strategic pushing when opponent is closer to edge
        rng = jax.random.PRNGKey(seed)
        steps_left = 0
        direction = 1  # +1 = CCW, -1 = CW

        # Critical safety calculations:
        # Arena radius = 0.4, agent radius = 0.05
        # Terminal condition: dist + agent_radius >= arena_radius
        # So agent center must stay at: dist <= 0.4 - 0.05 = 0.35
        # Add safety margin of 0.1 to avoid getting pushed out
        # So maximum safe distance from center: 0.35 - 0.1 = 0.25

        # Environment constants
        ARENA_RADIUS = 0.4
        AGENT_RADIUS = 0.05
        SAFETY_MARGIN = 0.1  # Safety margin
        MAX_SAFE_DIST = ARENA_RADIUS - AGENT_RADIUS - SAFETY_MARGIN  # 0.25

        # Orbit band parameters
        INNER_BOUND = 0.15   # Min distance from center
        OUTER_BOUND = 0.20   # Max distance (well below danger zone)
        TARGET_RADIUS = 0.18  # Target radius for orbit

        # Constants for orbit behavior
        MIN_ORBIT_STEPS = 6
        MAX_ORBIT_STEPS = 10

        # Pushing strategy parameters
        PUSH_THRESHOLD = 0.00  # Push even if opponent is at same distance from edge
        AGGRESSIVE_PUSH_THRESHOLD = 0.22  # When opponent is this close to edge, push aggressively
        PUSH_PRIORITY_THRESHOLD = 0.28  # When opponent is this close to edge, prioritize pushing over safety

        def _tangential_action(sx, sy, dir_sign):
            """Return action that moves tangentially around center."""
            # Choose axis with smaller absolute coordinate for tangential movement
            if abs(sx) < abs(sy):
                # Move along x-axis
                if sy > 0:  # Above center
                    return 2 if dir_sign > 0 else 1  # CCW: right, CW: left
                else:  # Below center
                    return 1 if dir_sign > 0 else 2  # CCW: left, CW: right
            else:
                # Move along y-axis
                if sx > 0:  # Right of center
                    return 3 if dir_sign > 0 else 4  # CCW: down, CW: up
                else:  # Left of center
                    return 4 if dir_sign > 0 else 3  # CCW: up, CW: down

        def _dodge(_, obs: jnp.ndarray):
            nonlocal rng, steps_left, direction

            # Extract positions and velocities from observation
            # obs format: [self_pos_x, self_pos_y, self_vel_x, self_vel_y, opp_pos_x, opp_pos_y, opp_vel_x, opp_vel_y]
            sx, sy = obs[0], obs[1]  # Self position
            ox, oy = obs[4], obs[5]  # Opponent position

            # Calculate distances from center
            self_dist = jnp.sqrt(sx**2 + sy**2)  # Self distance from center
            opp_dist = jnp.sqrt(ox**2 + oy**2)   # Opponent distance from center

            # SAFETY FIRST: Check if we're outside the safe band
            # First priority: Never get too close to the edge
            if self_dist > OUTER_BOUND:
                # Too close to edge, EMERGENCY CORRECTION
                # print(f"[DODGE] Too close to edge ({self_dist:.3f}), EMERGENCY inward correction")

                # Always move directly toward center
                # Use the dominant coordinate for fastest correction
                if abs(sx) > abs(sy):
                    action = 1 if sx > 0 else 2  # left if on right, right if on left
                else:
                    action = 3 if sy > 0 else 4  # down if on top, up if on bottom

                # Reset orbit counter to ensure we keep correcting
                steps_left = 0
                return jnp.array(action)

            # Second priority: Don't get too close to center
            elif self_dist < INNER_BOUND:
                # Too close to center, move directly outward
                # print(f"[DODGE] Too close to center ({self_dist:.3f}), EMERGENCY outward correction")

                # Calculate unit vector from center to agent
                if self_dist > 0.001:  # Avoid division by zero
                    ux, uy = sx/self_dist, sy/self_dist
                else:
                    ux, uy = 1.0, 0.0  # Default if at center

                # Always move along strongest axis for fastest correction
                if abs(sx) < abs(sy):
                    # Move along x-axis (smaller coordinate)
                    action = 2 if sx >= 0 else 1  # right or left
                else:
                    # Move along y-axis (smaller coordinate)
                    action = 4 if sy >= 0 else 3  # up or down

                # Reset orbit counter to ensure we keep correcting
                steps_left = 0
                return jnp.array(action)

            # Check if opponent is vulnerable (close to edge) or if we can push them
            # High priority push - opponent very close to edge
            if opp_dist > PUSH_PRIORITY_THRESHOLD:
                # Opponent is extremely close to edge - prioritize pushing!
                # print(f"[DODGE] PRIORITY PUSH! Opponent at {opp_dist:.3f}, self at {self_dist:.3f}")

                # Direct push along center-opponent line
                if abs(ox) > abs(oy):
                    # Opponent is more along x-axis
                    action = 2 if ox > 0 else 1  # right if opp right, left if opp left
                else:
                    # Opponent is more along y-axis
                    action = 4 if oy > 0 else 3  # up if opp up, down if opp down

                # Reset orbit counter since we're pushing
                steps_left = 0
                return jnp.array(action)

            # Normal push condition - we're in a safe position and opponent is closer to edge
            elif opp_dist > self_dist + PUSH_THRESHOLD and self_dist < 0.8 * MAX_SAFE_DIST:
                # Opponent is closer to edge - PUSH OUTWARD!

                # Calculate vector from center to opponent
                if opp_dist > 0.001:  # Avoid division by zero
                    # Calculate vector from self to opponent
                    dx, dy = ox - sx, oy - sy
                    dist_to_opp = jnp.sqrt(dx**2 + dy**2)

                    # Unit vector from center to opponent (outward direction)
                    oux, ouy = ox/opp_dist, oy/opp_dist

                    # Determine if opponent is very close to edge (aggressive push)
                    is_aggressive = opp_dist > AGGRESSIVE_PUSH_THRESHOLD

                    # if is_aggressive:
                    #     print(f"[DODGE] AGGRESSIVE PUSH! Opponent at {opp_dist:.3f}, self at {self_dist:.3f}")
                    # else:
                    #     print(f"[DODGE] PUSH! Opponent at {opp_dist:.3f}, self at {self_dist:.3f}")

                    # Calculate optimal push direction
                    # If we're close to opponent, push directly in line with center
                    if dist_to_opp < 0.15 or is_aggressive:
                        # Direct push along center-opponent line
                        if abs(ox) > abs(oy):
                            # Opponent is more along x-axis
                            action = 2 if ox > 0 else 1  # right if opp right, left if opp left
                        else:
                            # Opponent is more along y-axis
                            action = 4 if oy > 0 else 3  # up if opp up, down if opp down
                    else:
                        # We're not close enough for direct push
                        # Move toward opponent first
                        if abs(dx) > abs(dy):
                            action = 2 if dx > 0 else 1  # right if opp right, left if opp left
                        else:
                            action = 4 if dy > 0 else 3  # up if opp up, down if opp down

                    # Reset orbit counter since we're pushing
                    steps_left = 0
                    return jnp.array(action)

            # Only handle orbit behavior if in safe band

            # Check if we're close to the boundary danger zone
            if self_dist > 0.7 * MAX_SAFE_DIST:
                # Getting close to the danger zone, apply preventive correction
                # print(f"[DODGE] Near danger zone ({self_dist:.3f}), preventive inward correction")

                # Move directly inward toward center for fastest correction
                if abs(sx) > abs(sy):
                    action = 1 if sx > 0 else 2  # left if on right, right if on left
                else:
                    action = 3 if sy > 0 else 4  # down if on top, up if on bottom

                # Reset orbit counter to prioritize safety
                steps_left = 0
                return jnp.array(action)

            # Normal orbit behavior - change direction periodically
            if steps_left <= 0:
                # Time to change direction
                rng, subkey = jax.random.split(rng)
                direction = jax.random.choice(subkey, jnp.array([-1, 1]))
                rng, subkey = jax.random.split(rng)
                steps_left = jax.random.randint(subkey, (), MIN_ORBIT_STEPS, MAX_ORBIT_STEPS + 1)
                # print(f"[DODGE] New direction: {direction}, steps: {steps_left}")

            # Decrement steps counter
            steps_left -= 1

            # Choose action to move tangentially around center
            # print(f"[DODGE] Orbiting ({self_dist:.3f}), dir: {direction}, action: {_tangential_action(sx, sy, direction)}")
            return jnp.array(_tangential_action(sx, sy, direction), dtype=jnp.int32)

        return _dodge

    if path_or_baseline == "random":
        print(f'Loading random policy for {agent_name}')
        rng = jax.random.PRNGKey(seed)
        def _random(_, obs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return jax.random.randint(key, (), 0, env.action_space(agent_name).n)
        return _random


    # Otherwise treat as checkpoint path
    print(f"Loading checkpoint '{path_or_baseline}' for {agent_name}")
    with open(path_or_baseline, "rb") as fh:
        params = pickle.load(fh)

    # Create network matching the one in ippo_ff_mpe.py
    action_space_n = env.action_space(agent_name).n
    network = ActorCritic(action_space_n, activation=activation)

    def _policy(_, obs):
        pi, _ = network.apply(params, obs.flatten())
        # Use the same action selection as in get_rollout
        return int(pi.mode())

    return _policy


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_arena(
    env_name: str,
    env_kwargs: Dict,
    green: str,
    red: str,
    save_path: str = "arena.gif",
    max_steps: int = MAX_STEPS,
    seed: int = None,
    activation: str = "tanh",
    use_tqdm: bool = True,
):
    """Run a single rollout given two policies/baselines and export a gif."""
    env = jaxmarl.make(env_name, **env_kwargs)
    print(f'Created environment {env_name} with kwargs {env_kwargs}')

    # containers for action sequences
    action_seq_green: list[int] = []
    action_seq_red: list[int] = []

    # Get the two agents (green and red for MPE visualization)
    green_agent, red_agent = env.agents  # assumes 2p env with standard MPE coloring

    # Use current time as seed if none provided for variety
    if seed is None:
        import time
        seed = int(time.time())

    # Now that we have a valid seed, use it
    green_policy = _load_params_or_baseline(green, env, green_agent, seed, activation)
    red_policy = _load_params_or_baseline(red, env, red_agent, seed, activation)

    obs, state = env.reset(jax.random.PRNGKey(seed))
    state_seq, reward_seq = [], {a: [] for a in env.agents}

    step_iter = tqdm(range(max_steps), desc="rollout", leave=False) if (use_tqdm and _HAS_TQDM) else range(max_steps)
    for step in step_iter:
        # Get actions from policies
        green_action = int(green_policy(state, obs[green_agent]))
        red_action = int(red_policy(state, obs[red_agent]))

        # Log chosen actions
        action_seq_green.append(green_action)
        action_seq_red.append(red_action)

        # Create action dictionary for environment step
        actions = {green_agent: green_action, red_agent: red_action}

        # Step environment
        obs, next_state, reward, done, _ = env.step(
            jax.random.PRNGKey(seed + step), state, actions
        )

        # Log rewards
        for a in env.agents:
            reward_seq[a].append(reward[a])

        # Save state for visualisation (freeze final frame on termination)
        if done["__all__"]:
            frozen_state = next_state.replace(
                p_pos=next_state.snap.p_pos,
                p_vel=next_state.snap.p_vel,
                step=next_state.snap.step,
            )
            state_seq.append(frozen_state)
            if hasattr(step_iter, "close"):
                step_iter.close()
            break
        else:
            state_seq.append(next_state)

        state = next_state

    # ------------------------------------------------------------------
    # Metrics: action counts + KL divergence between the two policies
    # ------------------------------------------------------------------
    action_dim = env.action_space(green_agent).n
    green_counts = np.bincount(action_seq_green, minlength=action_dim)
    red_counts = np.bincount(action_seq_red, minlength=action_dim)
    p = green_counts / green_counts.sum()
    q = red_counts / red_counts.sum()
    eps = 1e-12
    kl_gr = float((p * np.log((p + eps) / (q + eps))).sum())
    kl_rg = float((q * np.log((q + eps) / (p + eps))).sum())

    print("[arena] Action counts – green:", green_counts, " red:", red_counts)
    print(f"[arena] KL(green‖red)={kl_gr:.6f}  KL(red‖green)={kl_rg:.6f}")

    viz = MPEVisualizer(env, state_seq, reward_seq)
    viz.animate(save_fname=save_path, view=False, loop=False)
    print(f"[arena] Saved rollout -> {save_path}")

    # Convert agent names to colors for reward sequence output
    reward_dict = {}
    for i, agent in enumerate(env.agents):
        color = "green" if i == 0 else "red"
        reward_dict[color] = [r.item() for r in reward_seq[agent]]

    # Prepare match outcome for later display
    green_final_reward = reward_dict["green"][-1] if reward_dict["green"] else 0
    red_final_reward = reward_dict["red"][-1] if reward_dict["red"] else 0

    if green_final_reward > 0:
        result = "GREEN WINS"
        ascii_art = """
+---------------------+
|                     |
|     GREEN WINS!     |
|       \\(^o^)/      |
|                     |
+---------------------+
"""
    elif red_final_reward > 0:
        result = "RED WINS"
        ascii_art = """
+---------------------+
|                     |
|      RED WINS!      |
|       \\(^o^)/      |
|                     |
+---------------------+
"""
    else:
        result = "TIE"
        ascii_art = """
+---------------------+
|                     |
|        TIE          |
|       ¯\\_(ツ)_/¯    |
|                     |
+---------------------+
"""

    return state_seq, reward_dict, result, ascii_art


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate two policies/baselines in MPE Sumo arena and export GIF.")
    parser.add_argument("--green", required=True, help="Checkpoint path or baseline name for green agent")
    parser.add_argument("--red", required=True, help="Checkpoint path or baseline name for red agent")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bar to ensure debug prints are visible")
    parser.add_argument("--save-folder", default="results", help="Output directory for the generated GIF (filename will be auto-generated)")
    parser.add_argument("--env", default="MPE_simple_sumo_v3", help="Registered env name in jaxmarl.make()")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: use time)")
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu"], help="Network activation function")

    parser.add_argument("--random-spawn", action="store_true", default=False,
                        help="Use randomized initial positions for both players")

    args = parser.parse_args()

    # Handle backward compatibility with old parameter names
    if args.green is None or args.red is None:
        parser.error("--green and --red must be provided")

    # ------------------------------------------------------------------
    # Derive output GIF filename
    # ------------------------------------------------------------------
    from pathlib import Path
    def _name_from_arg(arg: str) -> str:
        """Return a short identifier for checkpoint path or baseline name."""
        p = Path(arg)
        # If arg is a file path, use stem without extension; otherwise return raw string
        return p.stem if p.suffix else arg

    gif_name = f"{_name_from_arg(args.green)}_vs_{_name_from_arg(args.red)}.gif"
    save_dir = Path(args.save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / gif_name

    state_seq, reward_dict, result, ascii_art = run_arena(
        env_name=args.env,
        env_kwargs={'random_spawn': args.random_spawn},
        green=args.green,
        red=args.red,
        save_path=str(save_path),
        seed=args.seed,
        activation=args.activation,
        use_tqdm=not args.no_tqdm,
    )

    print("Rollout complete with reward sequence:")
    for agent, rewards in reward_dict.items():
        print(f"\t{agent}: {rewards}")

    # Print the match result with ASCII art at the very end
    print(f"\n[MATCH RESULT] {result}")
    print(ascii_art)

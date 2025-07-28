"""Evaluation arena for FSPPPO checkpoints in MPE Sumo.

Allows loading FSPPPO Orbax checkpoints (or scripted baselines) for each side and
produces a GIF of the rollout via the existing `MPEVisualizer`.

Usage:
    python -m baselines.FSPPPO.eval_arena \
        --green checkpoints/fspppo/run_xyz_seed0/main_agent/step_001000/ \
        --red seek \
        --save-folder arena_results
"""
from __future__ import annotations
import pickle
import time
from typing import Dict
from pathlib import Path

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
    from .fspppo_ff_mpe import ActorCritic  # type: ignore
    from .jax_checkpoint_utils import load_checkpoint_for_opponent, create_abstract_train_state
except ImportError:
    from baselines.FSPPPO.fspppo_ff_mpe import ActorCritic
    from baselines.FSPPPO.jax_checkpoint_utils import load_checkpoint_for_opponent, create_abstract_train_state

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
                return jnp.array(action)
            # CHASE mode
            # If heading away from opponent, immediately correct course
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
                if jnp.abs(sx) > jnp.abs(sy):
                    return jnp.array(1) if sx > 0 else jnp.array(2)
                else:
                    return jnp.array(3) if sy > 0 else jnp.array(4)
            # If opponent is outside safe radius, stay put
            if opp_dist > SAFE_RAD:
                return jnp.array(0)  # noop
            # If opponent is close, move away
            dx, dy = ox - sx, oy - sy
            if jnp.abs(dx) > jnp.abs(dy):
                return jnp.array(1) if dx > 0 else jnp.array(2)
            else:
                return jnp.array(3) if dy > 0 else jnp.array(4)
        return _centaur

    if path_or_baseline == "dodge":
        print(f'Loading evasive DODGE policy for {agent_name}')
        def _dodge(_, obs: jnp.ndarray):
            sx, sy = obs[0], obs[1]
            ox, oy = obs[4], obs[5]
            ovx, ovy = obs[6], obs[7]
            
            # Predict opponent position in next few steps
            PREDICT_STEPS = 3
            pred_ox = ox + ovx * PREDICT_STEPS
            pred_oy = oy + ovy * PREDICT_STEPS
            
            # Move away from predicted opponent position
            dx = pred_ox - sx
            dy = pred_oy - sy
            
            # Stay near center while dodging
            self_dist = jnp.sqrt(sx ** 2 + sy ** 2)
            if self_dist > 0.3:  # if too far from center
                # Move toward center
                if jnp.abs(sx) > jnp.abs(sy):
                    return jnp.array(1) if sx > 0 else jnp.array(2)
                else:
                    return jnp.array(3) if sy > 0 else jnp.array(4)
            
            # Dodge opponent
            if jnp.abs(dx) > jnp.abs(dy):
                return jnp.array(1) if dx > 0 else jnp.array(2)  # move away in x
            else:
                return jnp.array(3) if dy > 0 else jnp.array(4)  # move away in y
        return _dodge

    if path_or_baseline == "random":
        print(f'Loading random policy for {agent_name}')
        key = jax.random.PRNGKey(seed)
        def _random(_, obs):
            nonlocal key
            key, subkey = jax.random.split(key)
            return jax.random.randint(subkey, (), 0, env.action_space(agent_name).n)
        return _random

    # Otherwise treat as FSPPPO checkpoint path
    checkpoint_path = Path(path_or_baseline).resolve()  # Convert to absolute path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_baseline}")
    
    print(f"Loading FSPPPO checkpoint '{checkpoint_path}' for {agent_name}")
    
    # Create abstract train state for loading
    config = {
        "LR": 2.5e-4,
        "ANNEAL_LR": True,
        "MAX_GRAD_NORM": 0.5,
    }
    
    # Create network matching the one in fspppo_ff_mpe.py
    action_space_n = env.action_space(agent_name).n
    network = ActorCritic(action_space_n, activation=activation)
    
    # Create abstract train state
    abstract_train_state = create_abstract_train_state(config, env, network)
    
    # Load checkpoint parameters
    params = load_checkpoint_for_opponent(str(checkpoint_path), abstract_train_state)
    
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
    if seed is None:
        seed = int(time.time())
    
    print(f"[arena] Running match with seed={seed}")
    print(f"[arena] Green: {green}")
    print(f"[arena] Red: {red}")
    
    # Create environment
    env = jaxmarl.make(env_name, **env_kwargs)
    print(f"Created environment {env_name} with kwargs {env_kwargs}")
    
    # Load policies
    green_agent = env.agents[0]  # typically "agent_0"
    red_agent = env.agents[1]    # typically "agent_1"
    
    green_policy = _load_params_or_baseline(green, env, green_agent, seed, activation)
    red_policy = _load_params_or_baseline(red, env, red_agent, seed + 1, activation)
    
    # Run rollout
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    
    obs, state = env.reset(reset_key)
    state_seq = [state]
    reward_seq = {agent: [] for agent in env.agents}
    
    iterator = range(max_steps)
    if use_tqdm and _HAS_TQDM:
        iterator = tqdm(iterator, desc="Arena rollout")
    
    for step in iterator:
        # Get actions from both policies
        green_action = green_policy(state, obs[green_agent])
        red_action = red_policy(state, obs[red_agent])
        
        actions = {
            green_agent: green_action,
            red_agent: red_action,
        }
        
        # Step environment
        key, step_key = jax.random.split(key)
        obs, next_state, rewards, dones, infos = env.step(step_key, state, actions)
        
        # Log rewards
        for agent in env.agents:
            reward_seq[agent].append(rewards[agent])
        
        # Save state for visualisation (freeze final frame on termination)
        if dones["__all__"]:
            # Use snap state to show the actual match outcome before auto-reset
            frozen_state = next_state.replace(
                p_pos=next_state.snap.p_pos,
                p_vel=next_state.snap.p_vel,
                step=next_state.snap.step,
            )
            state_seq.append(frozen_state)
            if use_tqdm and _HAS_TQDM and hasattr(iterator, "close"):
                iterator.close()
            break
        else:
            state_seq.append(next_state)
        
        state = next_state
    
    print(f"[arena] Rollout completed in {len(state_seq)-1} steps")
    
    # Calculate action distribution similarity (KL divergence)
    # This is a placeholder - would need to collect action probabilities for real KL
    print(f"[arena] KL divergence calculation not implemented for FSPPPO checkpoints")
    
    # Create visualization
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


def _name_from_arg(arg: str) -> str:
    """Return a short identifier for checkpoint path or baseline name."""
    p = Path(arg)
    # If arg is a directory path, use the parent directory name (step_XXXX)
    if p.is_dir():
        return p.name
    # If arg is a file path, use stem without extension; otherwise return raw string
    return p.stem if p.suffix else arg


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate two FSPPPO policies/baselines in MPE Sumo arena and export GIF.")
    parser.add_argument("--green", required=True, help="FSPPPO checkpoint path or baseline name for green agent")
    parser.add_argument("--red", required=True, help="FSPPPO checkpoint path or baseline name for red agent")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bar to ensure debug prints are visible")
    parser.add_argument("--save-folder", default="arena_results", help="Output directory for the generated GIF")
    parser.add_argument("--env", default="MPE_simple_sumo_v3", help="Registered env name in jaxmarl.make()")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: use time)")
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu"], help="Network activation function")
    parser.add_argument("--random-spawn", action="store_true", default=False,
                        help="Use randomized initial positions for both players")

    args = parser.parse_args()

    # Derive output GIF filename
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

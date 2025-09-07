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
    # Check if it's a scripted behavior
    scripted_behaviors = ["noop", "random", "seek", "guardian", "dodge"]
    if path_or_baseline in scripted_behaviors:
        print(f'Loading scripted {path_or_baseline} policy for {agent_name}')
        try:
            from baselines.scripted_behaviors import get_scripted_agent
            return get_scripted_agent(path_or_baseline, seed)
        except ImportError:
            raise ImportError("Cannot import scripted_behaviors module - ensure it's available")


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
    # Use fixed starting positions for consistent evaluation
    if env_name == "MPE_simple_sumo_v3":
        env_kwargs = env_kwargs.copy()
        env_kwargs["random_spawn"] = False
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

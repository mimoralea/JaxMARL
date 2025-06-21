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


def _load_params_or_baseline(path_or_baseline: str, env, agent_name: str, seed: int = 0, activation: str = "tanh"):
    """Return a function(state, obs) -> action for the given side."""
    if path_or_baseline == "noop":
        return lambda *_: jnp.array(0)  # action 0 assumed to be noop
    if path_or_baseline == "random_walk":
        # Use the provided seed for reproducibility but different across runs
        rng = jax.random.PRNGKey(seed)

        def _random(_, __):
            nonlocal rng
            rng, sub = jax.random.split(rng)
            return jax.random.randint(sub, (), 0, env.action_space(agent_name).n)

        return _random

    # Otherwise treat as checkpoint path
    with open(path_or_baseline, "rb") as fh:
        params = pickle.load(fh)

    # Create network matching the one in ippo_ff_mpe.py
    action_space_n = env.action_space(agent_name).n
    network = ActorCritic(action_space_n, activation=activation)

    # Initialize with dummy input (same as in get_rollout)
    key = jax.random.PRNGKey(seed)
    init_x = jnp.zeros(env.observation_space(agent_name).shape).flatten()
    network.init(key, init_x)

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
):
    """Run a single rollout given two policies/baselines and export a gif."""
    env = jaxmarl.make(env_name, **env_kwargs)

    # Get the two agents (player_0 is green, player_1 is red in MPE visualization)
    green_agent, red_agent = env.agents  # assumes 2p env with standard MPE coloring

    # Use current time as seed if none provided for variety
    if seed is None:
        import time
        seed = int(time.time())

    # Now that we have a valid seed, use it
    green_pol = _load_params_or_baseline(green, env, green_agent, seed=seed, activation=activation)
    red_pol = _load_params_or_baseline(red, env, red_agent, seed=seed+100, activation=activation)

    obs, state = env.reset(jax.random.PRNGKey(seed))
    state_seq, reward_seq = [], {a: [] for a in env.agents}

    for step in range(max_steps):
        # Choose actions
        green_act = int(green_pol(state, obs[green_agent]))
        red_act = int(red_pol(state, obs[red_agent]))
        actions = {green_agent: green_act, red_agent: red_act}

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
            break
        else:
            state_seq.append(next_state)

        state = next_state

    viz = MPEVisualizer(env, state_seq, reward_seq)
    viz.animate(save_fname=save_path, view=False, loop=False)
    print(f"[arena] Saved rollout -> {save_path}")

    return state_seq, reward_seq


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate two policies/baselines in MPE Sumo arena and export GIF.")
    parser.add_argument("--green", required=True, help="Checkpoint path or baseline name for green agent (player_0)")
    parser.add_argument("--red", required=True, help="Checkpoint path or baseline name for red agent (player_1)")
    parser.add_argument("--save", default="arena.gif", help="Output GIF filename")
    parser.add_argument("--env", default="MPE_simple_sumo_v3", help="Registered env name in jaxmarl.make()")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: use time)")
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu"], help="Network activation function")
    
    # For backward compatibility, also accept the old --left and --right parameters
    parser.add_argument("--left", help="[DEPRECATED] Use --green instead. Checkpoint path or baseline name for green agent")
    parser.add_argument("--right", help="[DEPRECATED] Use --red instead. Checkpoint path or baseline name for red agent")
    
    args = parser.parse_args()
    
    # Handle backward compatibility with old parameter names
    green = args.green if args.green is not None else args.left
    red = args.red if args.red is not None else args.right
    
    if green is None or red is None:
        parser.error("Either --green and --red OR --left and --right must be provided")
    
    if args.left is not None or args.right is not None:
        print("[WARNING] --left and --right parameters are deprecated. Use --green and --red instead.")

    run_arena(
        env_name=args.env,
        env_kwargs={},
        green=green,
        red=red,
        save_path=args.save,
        seed=args.seed,
        activation=args.activation,
    )

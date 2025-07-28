"""Unified Tournament Evaluation System for IPPO, SPPPO, and FSPPPO.

This script provides comprehensive evaluation capabilities for all baseline algorithms:
- IPPO: Independent PPO with Orbax checkpoints (two separate policies per agent)
- SPPPO: Self-Play PPO with Orbax checkpoints (single shared policy)  
- FSPPPO: Fictitious Self-Play PPO with Orbax checkpoints (single policy)

All algorithms now use unified Orbax checkpoint format for consistent loading.

Features:
- Cross-algorithm evaluation (IPPO vs SPPPO, etc.)
- Algorithm vs scripted baseline evaluation
- Statistical analysis with CSV export
- Batch tournament execution with multiple seeds
- Win rate, robustness, and generalization metrics

Usage:
    python -m baselines.tournament_eval --config tournament_config.yaml
    python -m baselines.tournament_eval --single-match --green IPPO:checkpoints/ippo/run_xyz_seed0/agent_0/step_1000/ --red seek
"""

from __future__ import annotations
import csv
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import yaml

import jax
import jax.numpy as jnp
import numpy as np

# Optional progress bar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

import jaxmarl
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
from jaxmarl.environments.mpe.default_params import MAX_STEPS

# Import all algorithm networks
from baselines.IPPO.ippo_ff_mpe import ActorCritic as IPPOActorCritic
from baselines.SPPPO.spppo_ff_mpe import ActorCritic as SPPPOActorCritic  
from baselines.FSPPPO.fspppo_ff_mpe import ActorCritic as FSPPPOActorCritic

# Import unified checkpoint utilities (all algorithms now use Orbax)
from baselines.FSPPPO.jax_checkpoint_utils import load_checkpoint_for_opponent, create_abstract_train_state
from baselines.IPPO.orbax_checkpoint_manager import IPPOCheckpointManager

# -----------------------------------------------------------------------------
# Policy Loading Functions
# -----------------------------------------------------------------------------

def load_ippo_policy(checkpoint_path: str, env, agent_name: str, activation: str = "tanh"):
    """Load IPPO policy from Orbax checkpoint."""
    print(f"Loading IPPO checkpoint '{checkpoint_path}' for {agent_name}")
    
    # Use direct Orbax loading for IPPO checkpoints
    # Path format: checkpoints/ippo/run_xyz_seed0/agent_0/4882.0/
    import orbax.checkpoint as ocp
    
    checkpoint_path = Path(checkpoint_path).resolve()
    
    # Create checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    
    # Load train_state from the checkpoint
    train_state = checkpointer.restore(str(checkpoint_path / "train_state"))
    
    # Extract parameters (train_state might be a dict or TrainState object)
    if hasattr(train_state, 'params'):
        params = train_state.params
    else:
        params = train_state['params']  # If it's a dict
    
    # Create network
    action_space_n = env.action_space(agent_name).n
    network = IPPOActorCritic(action_space_n, activation=activation)
    
    def policy(_, obs):
        pi, _ = network.apply(params, obs.flatten())
        return int(pi.mode())
    
    return policy

def load_spppo_policy(checkpoint_path: str, env, agent_name: str, activation: str = "tanh"):
    """Load SPPPO policy from Orbax checkpoint."""
    print(f"Loading SPPPO checkpoint '{checkpoint_path}' for {agent_name}")
    
    # Create abstract train state for loading
    config = {
        "LR": 2.5e-4,
        "ANNEAL_LR": True,
        "MAX_GRAD_NORM": 0.5,
    }
    
    action_space_n = env.action_space(agent_name).n
    network = SPPPOActorCritic(action_space_n, activation=activation)
    
    # Create abstract train state and load checkpoint
    abstract_train_state = create_abstract_train_state(config, env, network)
    params = load_checkpoint_for_opponent(str(Path(checkpoint_path).resolve()), abstract_train_state)
    
    def policy(_, obs):
        pi, _ = network.apply(params, obs.flatten())
        return int(pi.mode())
    
    return policy

def load_fspppo_policy(checkpoint_path: str, env, agent_name: str, activation: str = "tanh"):
    """Load FSPPPO policy from Orbax checkpoint."""
    print(f"Loading FSPPPO checkpoint '{checkpoint_path}' for {agent_name}")
    
    # Create abstract train state for loading
    config = {
        "LR": 2.5e-4,
        "ANNEAL_LR": True,
        "MAX_GRAD_NORM": 0.5,
    }
    
    action_space_n = env.action_space(agent_name).n
    network = FSPPPOActorCritic(action_space_n, activation=activation)
    
    # Create abstract train state and load checkpoint
    abstract_train_state = create_abstract_train_state(config, env, network)
    params = load_checkpoint_for_opponent(str(Path(checkpoint_path).resolve()), abstract_train_state)
    
    def policy(_, obs):
        pi, _ = network.apply(params, obs.flatten())
        return int(pi.mode())
    
    return policy

def load_scripted_baseline(baseline_name: str, env, agent_name: str, seed: int):
    """Load scripted baseline policy."""
    print(f"Loading scripted baseline '{baseline_name}' for {agent_name}")
    
    if baseline_name == "noop":
        return lambda *_: jnp.array(0)
    
    elif baseline_name == "seek":
        mode = "chase"
        def _seek(_, obs: jnp.ndarray):
            nonlocal mode
            sx, sy = obs[0], obs[1]
            dx = obs[4] - sx
            dy = obs[5] - sy
            vx, vy = obs[2], obs[3]
            dist = jnp.sqrt(sx ** 2 + sy ** 2)
            outward_dot = sx * vx + sy * vy
            dir_dot = vx * dx + vy * dy
            
            if mode == "chase" and ((dist > 0.25) or (outward_dot > 0)):
                mode = "retreat"
            elif mode == "retreat" and (dist < 0.15) and (outward_dot <= 0):
                mode = "chase"
            
            if mode == "retreat":
                if jnp.abs(sx) > jnp.abs(sy):
                    action = 1 if sx > 0 else 2
                else:
                    action = 3 if sy > 0 else 4
                return jnp.array(action)
            
            if dir_dot < 0:
                if jnp.abs(dx) > jnp.abs(dy):
                    action = jnp.array(2) if dx > 0 else jnp.array(1)
                else:
                    action = jnp.array(4) if dy > 0 else jnp.array(3)
                return action
            
            if jnp.abs(dx) > jnp.abs(dy):
                return jnp.array(2) if dx > 0 else jnp.array(1)
            else:
                return jnp.array(4) if dy > 0 else jnp.array(3)
        return _seek
    
    elif baseline_name == "centaur":
        SAFE_RAD = 0.15
        def _centaur(_, obs: jnp.ndarray):
            sx, sy = obs[0], obs[1]
            ox, oy = obs[4], obs[5]
            self_dist = jnp.sqrt(sx ** 2 + sy ** 2)
            opp_dist = jnp.sqrt(ox ** 2 + oy ** 2)
            
            if self_dist > SAFE_RAD * 0.9:
                if jnp.abs(sx) > jnp.abs(sy):
                    return jnp.array(1) if sx > 0 else jnp.array(2)
                else:
                    return jnp.array(3) if sy > 0 else jnp.array(4)
            
            if opp_dist > SAFE_RAD:
                return jnp.array(0)
            
            dx, dy = ox - sx, oy - sy
            if jnp.abs(dx) > jnp.abs(dy):
                return jnp.array(1) if dx > 0 else jnp.array(2)
            else:
                return jnp.array(3) if dy > 0 else jnp.array(4)
        return _centaur
    
    elif baseline_name == "dodge":
        def _dodge(_, obs: jnp.ndarray):
            sx, sy = obs[0], obs[1]
            ox, oy = obs[4], obs[5]
            ovx, ovy = obs[6], obs[7]
            
            PREDICT_STEPS = 3
            pred_ox = ox + ovx * PREDICT_STEPS
            pred_oy = oy + ovy * PREDICT_STEPS
            
            dx = pred_ox - sx
            dy = pred_oy - sy
            
            self_dist = jnp.sqrt(sx ** 2 + sy ** 2)
            if self_dist > 0.3:
                if jnp.abs(sx) > jnp.abs(sy):
                    return jnp.array(1) if sx > 0 else jnp.array(2)
                else:
                    return jnp.array(3) if sy > 0 else jnp.array(4)
            
            if jnp.abs(dx) > jnp.abs(dy):
                return jnp.array(1) if dx > 0 else jnp.array(2)
            else:
                return jnp.array(3) if dy > 0 else jnp.array(4)
        return _dodge
    
    elif baseline_name == "random":
        key = jax.random.PRNGKey(seed)
        def _random(_, obs):
            nonlocal key
            key, subkey = jax.random.split(key)
            return jax.random.randint(subkey, (), 0, env.action_space(agent_name).n)
        return _random
    
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

def parse_agent_spec(agent_spec: str) -> Tuple[str, str]:
    """Parse agent specification string.
    
    Format: ALGORITHM:PATH or BASELINE_NAME
    Examples:
        - IPPO:checkpoints/ippo/run_123_seed0/agent_0/step_1000/
        - SPPPO:checkpoints/spppo/run_123_seed0/step_1000/
        - FSPPPO:checkpoints/fspppo/run_123_seed0/main_agent/step_1000/
        - seek
        - centaur
    """
    if ":" in agent_spec:
        algorithm, path = agent_spec.split(":", 1)
        return algorithm.upper(), path
    else:
        # Assume it's a scripted baseline
        return "SCRIPTED", agent_spec

def load_policy(agent_spec: str, env, agent_name: str, seed: int, activation: str = "tanh"):
    """Load policy from agent specification."""
    algorithm, path_or_baseline = parse_agent_spec(agent_spec)
    
    if algorithm == "IPPO":
        return load_ippo_policy(path_or_baseline, env, agent_name, activation)
    elif algorithm == "SPPPO":
        return load_spppo_policy(path_or_baseline, env, agent_name, activation)
    elif algorithm == "FSPPPO":
        return load_fspppo_policy(path_or_baseline, env, agent_name, activation)
    elif algorithm == "SCRIPTED":
        return load_scripted_baseline(path_or_baseline, env, agent_name, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# -----------------------------------------------------------------------------
# Tournament Execution
# -----------------------------------------------------------------------------

def run_single_match(
    env_name: str,
    env_kwargs: Dict,
    green_spec: str,
    red_spec: str,
    match_seed: int,
    max_steps: int = MAX_STEPS,
    activation: str = "tanh",
    save_gif: bool = False,
    gif_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single match between two agents."""
    
    # Create environment
    env = jaxmarl.make(env_name, **env_kwargs)
    
    # Load policies
    green_agent = env.agents[0]
    red_agent = env.agents[1]
    
    green_policy = load_policy(green_spec, env, green_agent, match_seed, activation)
    red_policy = load_policy(red_spec, env, red_agent, match_seed + 1, activation)
    
    # Run match
    key = jax.random.PRNGKey(match_seed)
    key, reset_key = jax.random.split(key)
    
    obs, state = env.reset(reset_key)
    state_seq = [state]
    reward_seq = {agent: [] for agent in env.agents}
    
    for step in range(max_steps):
        # Get actions
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
        
        # Save state for visualization (freeze final frame on termination)
        if dones["__all__"]:
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
    
    # Calculate results
    green_total_reward = sum(reward_seq[green_agent])
    red_total_reward = sum(reward_seq[red_agent])
    
    if green_total_reward > red_total_reward:
        winner = "green"
    elif red_total_reward > green_total_reward:
        winner = "red"
    else:
        winner = "tie"
    
    # Save GIF if requested
    if save_gif and gif_path:
        viz = MPEVisualizer(env, state_seq, reward_seq)
        viz.animate(save_fname=gif_path, view=False, loop=False)
    
    return {
        "green_spec": green_spec,
        "red_spec": red_spec,
        "match_seed": match_seed,
        "winner": winner,
        "green_reward": float(green_total_reward),
        "red_reward": float(red_total_reward),
        "episode_length": len(state_seq) - 1,
        "green_algorithm": parse_agent_spec(green_spec)[0],
        "red_algorithm": parse_agent_spec(red_spec)[0],
    }

def run_tournament(
    tournament_config: Dict[str, Any],
    output_csv: str,
    save_gifs: bool = False,
    gif_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a full tournament based on configuration."""
    
    env_name = tournament_config.get("env_name", "MPE_simple_sumo_v3")
    env_kwargs = tournament_config.get("env_kwargs", {})
    agents = tournament_config["agents"]
    num_seeds = tournament_config.get("num_seeds", 5)
    base_seed = tournament_config.get("base_seed", 42)
    
    # Generate all matchups
    matchups = []
    for i, green_spec in enumerate(agents):
        for j, red_spec in enumerate(agents):
            if i != j:  # Don't match agent against itself
                matchups.append((green_spec, red_spec))
    
    # Add scripted baseline matchups
    scripted_baselines = tournament_config.get("scripted_baselines", ["seek", "centaur", "dodge", "random", "noop"])
    for agent_spec in agents:
        for baseline in scripted_baselines:
            matchups.append((agent_spec, baseline))
            matchups.append((baseline, agent_spec))
    
    print(f"Running tournament with {len(matchups)} matchup types × {num_seeds} seeds = {len(matchups) * num_seeds} total matches")
    
    results = []
    
    # Create progress bar
    total_matches = len(matchups) * num_seeds
    if _HAS_TQDM:
        pbar = tqdm(total=total_matches, desc="Tournament Progress")
    
    for matchup_idx, (green_spec, red_spec) in enumerate(matchups):
        for seed_idx in range(num_seeds):
            match_seed = base_seed + matchup_idx * num_seeds + seed_idx
            
            # Determine GIF path if saving
            gif_path = None
            if save_gifs and gif_dir:
                green_name = parse_agent_spec(green_spec)[0] + "_" + Path(parse_agent_spec(green_spec)[1]).stem if ":" in green_spec else green_spec
                red_name = parse_agent_spec(red_spec)[0] + "_" + Path(parse_agent_spec(red_spec)[1]).stem if ":" in red_spec else red_spec
                gif_filename = f"{green_name}_vs_{red_name}_seed{seed_idx}.gif"
                gif_path = Path(gif_dir) / gif_filename
                Path(gif_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                result = run_single_match(
                    env_name=env_name,
                    env_kwargs=env_kwargs,
                    green_spec=green_spec,
                    red_spec=red_spec,
                    match_seed=match_seed,
                    save_gif=save_gifs,
                    gif_path=str(gif_path) if gif_path else None,
                )
                result["seed_idx"] = seed_idx
                results.append(result)
                
            except Exception as e:
                print(f"Error in match {green_spec} vs {red_spec} (seed {seed_idx}): {e}")
                # Add failed match record
                results.append({
                    "green_spec": green_spec,
                    "red_spec": red_spec,
                    "match_seed": match_seed,
                    "seed_idx": seed_idx,
                    "winner": "error",
                    "green_reward": 0.0,
                    "red_reward": 0.0,
                    "episode_length": 0,
                    "green_algorithm": parse_agent_spec(green_spec)[0],
                    "red_algorithm": parse_agent_spec(red_spec)[0],
                    "error": str(e),
                })
            
            if _HAS_TQDM:
                pbar.update(1)
    
    if _HAS_TQDM:
        pbar.close()
    
    # Save results to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Tournament results saved to {output_csv}")
        
        # Print summary statistics
        total_matches = len([r for r in results if r["winner"] != "error"])
        error_matches = len([r for r in results if r["winner"] == "error"])
        
        print(f"\nTournament Summary:")
        print(f"  Total matches: {total_matches}")
        print(f"  Failed matches: {error_matches}")
        
        if total_matches > 0:
            win_rates = {}
            for result in results:
                if result["winner"] == "error":
                    continue
                
                green_alg = result["green_algorithm"]
                red_alg = result["red_algorithm"]
                
                if green_alg not in win_rates:
                    win_rates[green_alg] = {"wins": 0, "total": 0}
                if red_alg not in win_rates:
                    win_rates[red_alg] = {"wins": 0, "total": 0}
                
                win_rates[green_alg]["total"] += 1
                win_rates[red_alg]["total"] += 1
                
                if result["winner"] == "green":
                    win_rates[green_alg]["wins"] += 1
                elif result["winner"] == "red":
                    win_rates[red_alg]["wins"] += 1
            
            print(f"\nWin Rates by Algorithm:")
            for alg, stats in win_rates.items():
                if stats["total"] > 0:
                    win_rate = stats["wins"] / stats["total"]
                    print(f"  {alg}: {win_rate:.3f} ({stats['wins']}/{stats['total']})")
    
    return results

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Tournament Evaluation System")
    
    # Single match mode
    parser.add_argument("--single-match", action="store_true", help="Run a single match")
    parser.add_argument("--green", help="Green agent specification (ALGORITHM:PATH or BASELINE)")
    parser.add_argument("--red", help="Red agent specification (ALGORITHM:PATH or BASELINE)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-gif", action="store_true", help="Save match GIF")
    parser.add_argument("--gif-path", help="Path to save GIF")
    
    # Tournament mode
    parser.add_argument("--config", help="Tournament configuration YAML file")
    parser.add_argument("--output-csv", default="tournament_results.csv", help="Output CSV file")
    parser.add_argument("--save-gifs", action="store_true", help="Save GIFs for all matches")
    parser.add_argument("--gif-dir", default="tournament_gifs", help="Directory to save GIFs")
    
    # Environment options
    parser.add_argument("--env", default="MPE_simple_sumo_v3", help="Environment name")
    parser.add_argument("--random-spawn", action="store_true", help="Use random spawn positions")
    parser.add_argument("--activation", default="tanh", choices=["tanh", "relu"], help="Network activation")
    
    args = parser.parse_args()
    
    if args.single_match:
        if not args.green or not args.red:
            parser.error("--single-match requires --green and --red")
        
        env_kwargs = {"random_spawn": args.random_spawn}
        
        result = run_single_match(
            env_name=args.env,
            env_kwargs=env_kwargs,
            green_spec=args.green,
            red_spec=args.red,
            match_seed=args.seed,
            activation=args.activation,
            save_gif=args.save_gif,
            gif_path=args.gif_path,
        )
        
        print(f"\nMatch Result:")
        print(f"  Green ({args.green}): {result['green_reward']:.2f}")
        print(f"  Red ({args.red}): {result['red_reward']:.2f}")
        print(f"  Winner: {result['winner'].upper()}")
        print(f"  Episode length: {result['episode_length']} steps")
        
    elif args.config:
        with open(args.config, 'r') as f:
            tournament_config = yaml.safe_load(f)
        
        run_tournament(
            tournament_config=tournament_config,
            output_csv=args.output_csv,
            save_gifs=args.save_gifs,
            gif_dir=args.gif_dir if args.save_gifs else None,
        )
        
    else:
        parser.error("Must specify either --single-match or --config")

if __name__ == "__main__":
    main()

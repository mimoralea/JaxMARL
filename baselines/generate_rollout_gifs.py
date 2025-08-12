#!/usr/bin/env python3
"""
Generate rollout GIFs for scripted agents in SimpleSumoMPE environment.

This script creates animated GIFs showing rollouts between different scripted
agents to visualize their behaviors and strategies.
"""

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import distrax

from jaxmarl import make
from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer

# Learned agent imports (loaded dynamically in load_learned_agent function)
# No need for top-level imports since we load them when needed

try:
    from scripted_behaviors import get_scripted_action
except ImportError:
    try:
        from baselines.scripted_behaviors import get_scripted_action
    except ImportError:
        print("Warning: scripted_behaviors import failed - using local implementation")
        get_scripted_action = None

def get_scripted_agent(agent_name, seed=0):
    """Get a scripted agent policy function - exact reimplementation from eval_arena.py."""

    # Action ID to string mapping for display
    def action_id_to_string(action_id):
        """Convert discrete action ID to readable string."""
        action_map = {0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "DOWN", 4: "UP"}
        return action_map.get(int(action_id), "UNKNOWN")

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
        # Exact reimplementation from eval_arena.py
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
        # Exact reimplementation from eval_arena.py
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
        # Exact reimplementation from eval_arena.py (simplified version for now)
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
        raise ValueError(f"Unknown agent: {agent_name}")


def load_learned_agent(algorithm, checkpoint_dir="checkpoints"):
    """Load a learned agent from checkpoint using simple direct approach."""
    if algorithm.upper() == "FSPPPO":
        import glob
        import orbax.checkpoint as ocp

        # Find latest FSPPPO checkpoint directory
        fspppo_pattern = os.path.join(checkpoint_dir, "fspppo", "run_*_seed*", "main", "*")
        checkpoint_dirs = glob.glob(fspppo_pattern)

        if not checkpoint_dirs:
            raise FileNotFoundError(f"No FSPPPO checkpoints found in {checkpoint_dir}")

        # Filter to only numeric directories (actual checkpoints)
        numeric_dirs = [d for d in checkpoint_dirs if os.path.basename(d).isdigit()]

        if not numeric_dirs:
            raise FileNotFoundError(f"No valid FSPPPO checkpoint steps found in {checkpoint_dir}")

        # Get the latest checkpoint (highest step number)
        latest_checkpoint = max(numeric_dirs, key=lambda x: int(os.path.basename(x)))

        # Convert to absolute path for Orbax compatibility
        latest_checkpoint = os.path.abspath(latest_checkpoint)

        print(f"📥 Loading FSPPPO checkpoint from {latest_checkpoint}")

        try:
            # Import FSPPPO network - try multiple import paths
            from FSPPPO.train import ActorCritic
        except ImportError:
            try:
                from baselines.FSPPPO.train import ActorCritic
            except ImportError:
                try:
                    import sys
                    sys.path.append(os.path.dirname(__file__))
                    from FSPPPO.train import ActorCritic
                except ImportError:
                    raise ImportError("Cannot import FSPPPO ActorCritic - ensure FSPPPO is available")

        # Create dummy environment to get observation space
        dummy_env = SimpleSumoMPE()
        dummy_obs, _ = dummy_env.reset(jax.random.PRNGKey(0))
        obs_shape = dummy_obs['green'].shape

        # Initialize network
        network = ActorCritic(action_dim=5, activation="tanh")

        # Create dummy input for network initialization
        dummy_input = jnp.zeros((1,) + obs_shape)
        network_params = network.init(jax.random.PRNGKey(0), dummy_input)

        # Load checkpoint using Orbax
        checkpointer = ocp.StandardCheckpointer()
        loaded_params = checkpointer.restore(latest_checkpoint, network_params)

        print("✅ Successfully loaded FSPPPO checkpoint")

        # Create policy function
        def learned_policy(obs):
            # Ensure obs is properly shaped
            if obs.ndim == 1:
                obs = obs[None, ...]  # Add batch dimension

            # Get action distribution and value
            pi, _ = network.apply(loaded_params, obs)

            # Sample action from distribution
            action = pi.sample(seed=jax.random.PRNGKey(0))

            # Return scalar action (remove batch dimension)
            return action[0] if action.ndim > 0 else action

        return learned_policy

    else:
        raise ValueError(f"Unsupported learned algorithm: {algorithm}")

def action_id_to_string(action_id):
    """Convert discrete action ID to readable action string."""
    action_map = {0: "NOOP", 1: "LEFT", 2: "RIGHT", 3: "DOWN", 4: "UP"}
    return action_map.get(int(action_id), "UNKNOWN")

def run_rollout(env, agent1_name, agent2_name, key, max_steps=100, learned_agents=None):
    """Run a rollout between two agents (scripted or learned)."""
    learned_agents = learned_agents or {}

    # Get agent policies
    if agent1_name in learned_agents:
        agent1_policy = learned_agents[agent1_name]
    else:
        agent1_policy = get_scripted_agent(agent1_name, seed=0)

    if agent2_name in learned_agents:
        agent2_policy = learned_agents[agent2_name]
    else:
        agent2_policy = get_scripted_agent(agent2_name, seed=1)

    # Reset environment
    obs, state = env.reset(key)

    # Store trajectory
    trajectory = {
        'observations': [obs],
        'states': [state],
        'actions': [],
        'action_strings': [],  # Store action strings for display
        'rewards': [],
        'dones': [],
        'infos': [],
        'snaps': []  # Store snap data for true final states
    }

    for step in range(max_steps):
        # Get actions from both agents (discrete action IDs)
        action1_id = agent1_policy(obs['green'])
        action2_id = agent2_policy(obs['red'])

        # Convert to action strings for display
        action1_str = action_id_to_string(action1_id)
        action2_str = action_id_to_string(action2_id)

        actions = {'green': action1_id, 'red': action2_id}
        action_strings = {'green': action1_str, 'red': action2_str}

        # Step environment
        obs, state, rewards, dones, infos = env.step(key, state, actions)

        # Store step data
        trajectory['actions'].append(actions)
        trajectory['action_strings'].append(action_strings)
        trajectory['rewards'].append(rewards)
        trajectory['dones'].append(dones)
        trajectory['infos'].append(infos)
        trajectory['observations'].append(obs)
        trajectory['states'].append(state)

        # Store snap data if available (captures true final state before autoreset)
        if hasattr(state, 'snap') and state.snap is not None:
            trajectory['snaps'].append(state.snap)
        else:
            trajectory['snaps'].append(None)

        # Check if episode is done
        if dones['__all__']:
            break

    return trajectory

def create_rollout_frames(trajectory, env, agent1_name, agent2_name, extend_final_frames=30):
    """Create frames for animation from trajectory using custom rendering."""
    frames = []

    # Analyze outcome once for final frames
    outcome = analyze_rollout_outcome(trajectory)

    # Find the true final state using snap data (before autoreset)
    final_state_idx = len(trajectory['states']) - 1
    final_snap = None

    # Look for the last available snap (true termination state)
    for i in range(len(trajectory['snaps']) - 1, -1, -1):
        if trajectory['snaps'][i] is not None:
            final_snap = trajectory['snaps'][i]
            break

    for i in range(len(trajectory['states']) + extend_final_frames):
        # For regular frames (except the final one), use the current state
        if i < final_state_idx:
            state_idx = i
            state = trajectory['states'][state_idx]
            is_final_frame = False
        else:
            # Final frame and extended frames: use the true final state from snap if available
            state_idx = final_state_idx
            if final_snap is not None:
                # Create a state-like object from snap for visualization
                state = type('SnapState', (), {
                    'p_pos': final_snap.p_pos,
                    'p_vel': final_snap.p_vel,
                    'step': final_snap.step
                })()
            else:
                # Fallback to final state if no snap available
                state = trajectory['states'][final_state_idx]
            is_final_frame = True

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw arena boundary (circle)
        arena_radius = env.R if hasattr(env, 'R') else 0.4
        circle = plt.Circle((0, 0), arena_radius, fill=False, color='black', linewidth=3)
        ax.add_patch(circle)

        # Extract agent positions from state
        if hasattr(state, 'p_pos') and len(state.p_pos) >= 2:
            # Green agent (agent 0)
            green_pos = state.p_pos[0]
            green_circle = plt.Circle(green_pos, env.rad[0] if hasattr(env, 'rad') else 0.05,
                                    color='green', alpha=0.8)
            ax.add_patch(green_circle)

            # Red agent (agent 1)
            red_pos = state.p_pos[1]
            red_circle = plt.Circle(red_pos, env.rad[1] if hasattr(env, 'rad') else 0.05,
                                  color='red', alpha=0.8)
            ax.add_patch(red_circle)

            # Draw velocity vectors if available
            if hasattr(state, 'p_vel') and len(state.p_vel) >= 2:
                green_vel = state.p_vel[0]
                red_vel = state.p_vel[1]

                # Scale velocity for visualization
                vel_scale = 0.05
                if np.linalg.norm(green_vel) > 0.01:
                    ax.arrow(green_pos[0], green_pos[1],
                            green_vel[0] * vel_scale, green_vel[1] * vel_scale,
                            head_width=0.02, head_length=0.02, fc='darkgreen', ec='darkgreen', alpha=0.7)
                if np.linalg.norm(red_vel) > 0.01:
                    ax.arrow(red_pos[0], red_pos[1],
                            red_vel[0] * vel_scale, red_vel[1] * vel_scale,
                            head_width=0.02, head_length=0.02, fc='darkred', ec='darkred', alpha=0.7)

        # Set axis properties
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add step information
        if i >= final_state_idx:
            # For final frame and extended frames, use the final state information
            step_info = f"Step: {final_state_idx}"

            # Use final actions and rewards (grab the last available ones)
            if len(trajectory['action_strings']) > 0:
                action_strs = trajectory['action_strings'][-1]
                step_info += f" | Actions: Green={action_strs['green']}, Red={action_strs['red']}"

            if len(trajectory['rewards']) > 0:
                rewards = trajectory['rewards'][-1]
                step_info += f" | Rewards: Green={rewards['green']:.2f}, Red={rewards['red']:.2f}"
        else:
            # For regular frames, use current state information
            step_info = f"Step: {state_idx}"

            if state_idx < len(trajectory['action_strings']):
                action_strs = trajectory['action_strings'][state_idx]
                step_info += f" | Actions: Green={action_strs['green']}, Red={action_strs['red']}"

            if state_idx < len(trajectory['rewards']):
                rewards = trajectory['rewards'][state_idx]
                step_info += f" | Rewards: Green={rewards['green']:.2f}, Red={rewards['red']:.2f}"

        ax.set_title(step_info, fontsize=12)

        # Add outcome text on final extended frames - positioned outside arena
        if i >= len(trajectory['states']):
            # Determine text color based on winner
            if outcome['winner'] == 'green':
                text_color = 'green'
                bg_color = 'lightgreen'
            elif outcome['winner'] == 'red':
                text_color = 'red'
                bg_color = 'lightcoral'
            else:
                text_color = 'black'
                bg_color = 'lightgray'

            # Position text in a visible area within the view
            ax.text(0.4, -0.4, outcome['outcome'],
                   fontsize=16, fontweight='bold',
                   ha='center', va='center',
                   color=text_color,
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor=bg_color, alpha=0.9, edgecolor=text_color))

        # Convert to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        frames.append(frame)

        plt.close(fig)

    return frames

def save_gif(frames, output_path, fps=5):
    """Save frames as animated GIF."""
    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000/fps),  # Duration in milliseconds
        loop=0
    )

def analyze_rollout_outcome(trajectory):
    """Analyze the outcome of a rollout."""
    final_rewards = trajectory['rewards'][-1] if trajectory['rewards'] else {'green': 0, 'red': 0}
    final_dones = trajectory['dones'][-1] if trajectory['dones'] else {'__all__': False}

    # Determine winner
    if final_rewards['green'] > final_rewards['red']:
        winner = 'green'
        outcome = 'Green Wins'
    elif final_rewards['red'] > final_rewards['green']:
        winner = 'red'
        outcome = 'Red Wins'
    else:
        winner = 'draw'
        outcome = 'Draw/Timeout'

    episode_length = len(trajectory['states']) - 1

    return {
        'winner': winner,
        'outcome': outcome,
        'episode_length': episode_length,
        'final_rewards': final_rewards,
        'terminated_early': final_dones['__all__']
    }

def generate_multiple_rollouts(agent1_name, agent2_name, num_rollouts=10, output_dir='rollout_gifs', learned_algorithms=None):
    """Generate multiple rollout GIFs between two agents (scripted or learned)."""
    learned_algorithms = learned_algorithms or []

    print(f"🎬 Generating {num_rollouts} rollout GIFs: {agent1_name} vs {agent2_name}")

    # Load learned agents if specified
    learned_agents = {}
    for algorithm in learned_algorithms:
        if algorithm in [agent1_name, agent2_name]:
            try:
                learned_agents[algorithm] = load_learned_agent(algorithm)
            except Exception as e:
                print(f"❌ Failed to load {algorithm}: {e}")
                return []

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create environment with fixed starting positions
    env = SimpleSumoMPE(random_spawn=False)

    # Generate rollouts
    key = jax.random.PRNGKey(42)
    outcomes = []

    for i in range(num_rollouts):
        print(f"  🎯 Generating rollout {i+1}/{num_rollouts}...")

        # Generate rollout
        key, subkey = jax.random.split(key)
        trajectory = run_rollout(env, agent1_name, agent2_name, subkey, learned_agents=learned_agents)

        # Analyze outcome
        outcome = analyze_rollout_outcome(trajectory)
        outcomes.append(outcome)

        # Create frames
        frames = create_rollout_frames(trajectory, env, agent1_name, agent2_name)

        # Save GIF
        gif_filename = f"{agent1_name}_vs_{agent2_name}_rollout_{i+1:02d}.gif"
        gif_path = output_path / gif_filename
        save_gif(frames, gif_path)

        print(f"    ✅ Saved: {gif_path}")
        print(f"    📊 Outcome: {outcome['outcome']} (Length: {outcome['episode_length']} steps)")

    # Print summary statistics
    print(f"\n📈 ROLLOUT SUMMARY: {agent1_name} vs {agent2_name}")
    print("=" * 60)

    agent1_wins = sum(1 for o in outcomes if o['winner'] == 'green')
    agent2_wins = sum(1 for o in outcomes if o['winner'] == 'red')
    draws = sum(1 for o in outcomes if o['winner'] == 'draw')

    print(f"🏆 {agent1_name} (Green) wins: {agent1_wins}/{num_rollouts} ({agent1_wins/num_rollouts*100:.1f}%)")
    print(f"🏆 {agent2_name} (Red) wins: {agent2_wins}/{num_rollouts} ({agent2_wins/num_rollouts*100:.1f}%)")
    print(f"🤝 Draws/Timeouts: {draws}/{num_rollouts} ({draws/num_rollouts*100:.1f}%)")

    avg_length = np.mean([o['episode_length'] for o in outcomes])
    print(f"📏 Average episode length: {avg_length:.1f} steps")

    early_terminations = sum(1 for o in outcomes if o['terminated_early'])
    print(f"⚡ Early terminations: {early_terminations}/{num_rollouts} ({early_terminations/num_rollouts*100:.1f}%)")

    return outcomes

def main():
    parser = argparse.ArgumentParser(description='Generate agent rollout GIFs (scripted or learned)')
    parser.add_argument('--agent1', type=str, default='seek',
                       help='First agent type (scripted: noop, random, seek, dodge, guardian; learned: FSPPPO, IPPO, SPPPO)')
    parser.add_argument('--agent2', type=str, default='noop',
                       help='Second agent type (scripted: noop, random, seek, dodge, guardian; learned: FSPPPO, IPPO, SPPPO)')
    parser.add_argument('--num-rollouts', type=int, default=10,
                       help='Number of rollouts to generate')
    parser.add_argument('--output-dir', type=str, default='rollout_gifs',
                       help='Output directory for GIF files')
    parser.add_argument('--learned-algorithms', nargs='*', default=[],
                       choices=['FSPPPO', 'IPPO', 'SPPPO'],
                       help='Specify which agents are learned algorithms (default: all are scripted)')

    args = parser.parse_args()

    # Validate agent types
    scripted_agents = ['noop', 'random', 'seek', 'dodge', 'guardian']
    learned_agents = ['FSPPPO', 'IPPO', 'SPPPO']

    for agent_name in [args.agent1, args.agent2]:
        if agent_name not in scripted_agents and agent_name not in learned_agents:
            print(f"❌ Unknown agent type: {agent_name}")
            print(f"Available scripted agents: {', '.join(scripted_agents)}")
            print(f"Available learned agents: {', '.join(learned_agents)}")
            return

    # Generate rollouts
    outcomes = generate_multiple_rollouts(
        args.agent1,
        args.agent2,
        args.num_rollouts,
        args.output_dir,
        args.learned_algorithms
    )

    print(f"\n🎉 Generated {args.num_rollouts} rollout GIFs!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎬 Files: {args.agent1}_vs_{args.agent2}_rollout_XX.gif")

if __name__ == "__main__":
    main()

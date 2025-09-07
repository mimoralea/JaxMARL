#!/usr/bin/env python3
"""
Run BR Matchups (Consolidated)

Convenient entry-point that runs Best-Response evaluations against scripted and
learned opponents. This script now inlines the evaluator previously located in
`baselines/br_evaluation.py` so that a single file handles both the orchestration
and the per-episode evaluation.

Defaults:
- episodes per matchup: 100
- br-seeds: 0,1,2
- evaluation-seed: 0
- output base: experiments/results/br_eval/

Usage examples:
  python -m baselines.run_br_matchups --episodes 100 --br-seeds 0,1,2
  python -m baselines.run_br_matchups --only-scripted --br-seeds 0 --episodes 50
  python -m baselines.run_br_matchups --only-learned --opponent-seeds 0 --episodes 50
"""

import json
import os
import glob
import math
from datetime import datetime
from pathlib import Path
import argparse
import pandas as pd
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from baselines import scripted_behaviors

class TournamentPlayer:
    """Represents a player in the evaluation."""
    def __init__(
        self,
        name: str,
        player_type: str,
        checkpoint_path: str = None,
        algorithm: str = None,
        seed: int = None,
        run_id: str = None,
        checkpoint_step: int = None,
        timesteps: int = None,
    ):
        self.name = name
        self.player_type = player_type  # 'checkpoint' or 'scripted'
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self.seed = seed
        # Provenance fields
        self.run_id = run_id
        self.checkpoint_step = checkpoint_step
        self.timesteps = timesteps
        # Runtime fields
        self.params = None
        self.apply_fn = None


class BREvaluator:
    def __init__(
        self,
        env_name: str = "MPE_simple_sumo_v3",
        episodes: int = 100,
        output_dir: str = "experiments/results/br_eval",
        max_episode_steps: int = 100,
        seed: int = 0,
        include_scripted: bool = True,
        include_learned: bool = True,
    ):
        self.env_name = env_name
        self.episodes = episodes
        self.max_episode_steps = max_episode_steps
        self.seed = int(seed)
        self.include_scripted = include_scripted
        self.include_learned = include_learned

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(output_dir)
        self.output_dir = self.base_output_dir / f"run_{self.run_timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.env = make(self.env_name)
        self.env = LogWrapper(self.env)
        self.num_agents = len(self.env.agents)

        self.scripted_behaviors = ["noop", "random", "seek", "guardian", "dodge"]
        self.baseline_algorithms = ["ippo", "spppo", "fspppo"]

    # ---------- Checkpoint discovery helpers (seed-aware) ----------
    def _seed_suffix(self) -> str:
        return f"seed{self.seed}"

    def find_latest_br_checkpoint(self, behavior: str):
        br_base_dir = f"/workspace/code/src/JaxMARL/checkpoints/br/{behavior}"
        if not os.path.exists(br_base_dir):
            raise ValueError(f"BR checkpoint directory not found: {br_base_dir}")

        run_dirs = [d for d in os.listdir(br_base_dir) if d.startswith("run_") and d.endswith(self._seed_suffix())]
        if not run_dirs:
            raise ValueError(f"No run directories found in {br_base_dir} for {self._seed_suffix()}")
        run_dirs.sort(reverse=True)
        latest_run = run_dirs[0]

        main_agent_dir = os.path.join(br_base_dir, latest_run, "main_agent")
        if not os.path.exists(main_agent_dir):
            raise ValueError(f"Main agent directory not found: {main_agent_dir}")

        checkpoints = [d for d in os.listdir(main_agent_dir) if d.isdigit()]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {main_agent_dir}")

        latest_checkpoint = max(checkpoints, key=int)
        br_path = os.path.join(main_agent_dir, latest_checkpoint)

        print(f"Using latest BR checkpoint for {behavior}: {br_path}")
        return br_path, int(latest_checkpoint), latest_run

    def find_matched_br_checkpoint(self, behavior: str, target_timesteps: int):
        br_base_dir = f"/workspace/code/src/JaxMARL/checkpoints/br/{behavior}"
        if not os.path.exists(br_base_dir):
            raise ValueError(f"BR checkpoint directory not found: {br_base_dir}")

        run_dirs = [d for d in os.listdir(br_base_dir) if d.startswith("run_") and d.endswith(self._seed_suffix())]
        if not run_dirs:
            raise ValueError(f"No run directories found in {br_base_dir} for {self._seed_suffix()}")
        run_dirs.sort(reverse=True)
        latest_run = run_dirs[0]

        main_agent_dir = os.path.join(br_base_dir, latest_run, "main_agent")
        if not os.path.exists(main_agent_dir):
            raise ValueError(f"Main agent directory not found: {main_agent_dir}")

        checkpoints = [d for d in os.listdir(main_agent_dir) if d.isdigit()]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {main_agent_dir}")

        best_checkpoint = None
        best_diff = float('inf')
        for checkpoint in checkpoints:
            checkpoint_timesteps = int(checkpoint) * 16 * 128
            diff = abs(checkpoint_timesteps - target_timesteps)
            if diff < best_diff:
                best_diff = diff
                best_checkpoint = checkpoint

        if best_checkpoint is None:
            best_checkpoint = max(checkpoints, key=int)

        br_path = os.path.join(main_agent_dir, best_checkpoint)
        print(f"Using matched BR checkpoint for {behavior}: {br_path}")
        return br_path, int(best_checkpoint), latest_run

    def find_latest_baseline_checkpoint(self, algorithm: str, seed: int = None):
        baseline_base_dir = f"/workspace/code/src/JaxMARL/checkpoints/{algorithm}"
        if not os.path.exists(baseline_base_dir):
            return None

        # Use provided seed (opponent seed) if specified; otherwise use evaluator seed
        seed_suffix = f"seed{seed}" if seed is not None else self._seed_suffix()
        run_dirs = [d for d in os.listdir(baseline_base_dir) if d.startswith("run_") and d.endswith(seed_suffix)]
        if not run_dirs:
            return None
        run_dirs.sort(reverse=True)

        for run_dir in run_dirs:
            main_dir = os.path.join(baseline_base_dir, run_dir, "main")
            if not os.path.exists(main_dir):
                continue
            checkpoints = [d for d in os.listdir(main_dir) if d.isdigit()]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=int)
                baseline_path = os.path.join(main_dir, latest_checkpoint)
                print(f"Using latest {algorithm.upper()} checkpoint: {baseline_path}")
                return baseline_path, int(latest_checkpoint), run_dir
        return None

    # ---------- Player loading ----------
    def load_checkpoint_player(self, player: TournamentPlayer):
        if player.params is not None:
            return
        try:
            # Use shared ActorCritic (baselines.algorithms.ppo)
            try:
                from baselines.algorithms.ppo import ActorCritic
            except ImportError:
                from ..algorithms.ppo import ActorCritic
            network = ActorCritic(self.env.action_space(self.env.agents[0]).n, activation="tanh")

            checkpointer = ocp.PyTreeCheckpointer()
            restored = None
            try:
                restored = checkpointer.restore(player.checkpoint_path)
                if 'model' in restored:
                    player.params = restored['model']['params']
                elif 'params' in restored:
                    player.params = {'params': restored['params']}
                else:
                    player.params = restored
            except Exception:
                train_state_path = os.path.join(player.checkpoint_path, 'train_state')
                if os.path.exists(train_state_path):
                    restored = checkpointer.restore(train_state_path)
                    if 'params' in restored:
                        player.params = restored['params']
                    elif 'train_state' in restored:
                        player.params = restored['train_state']['params']
                    else:
                        player.params = restored
                else:
                    parent_path = os.path.dirname(player.checkpoint_path)
                    restored = checkpointer.restore(parent_path)
                    if 'params' in restored:
                        player.params = {'params': restored['params']}
                    elif 'train_state' in restored:
                        player.params = restored['train_state']['params']
                    else:
                        player.params = restored

            player.apply_fn = jax.jit(network.apply)
            print(f"Loaded {player.name} from {player.checkpoint_path}")
        except Exception as e:
            print(f"Failed to load {player.name}: {e}")
            player.params = None
            player.apply_fn = None

    # ---------- Episode/run helpers ----------
    def get_scripted_action(self, obs, behavior, rng_key):
        return scripted_behaviors.get_scripted_action(obs, behavior, rng_key)

    def run_single_episode(self, player1: TournamentPlayer, player2: TournamentPlayer, behavior: str, rng_key, episode_id: int):
        rng_key, reset_key = jax.random.split(rng_key)
        obs, state = self.env.reset(reset_key)
        episode_rewards = {agent: 0.0 for agent in self.env.agents}
        steps = 0
        reward_log = {agent: [] for agent in self.env.agents}
        done_log = []

        done_all = False
        while not done_all:
            actions = {}
            for agent, player in zip(self.env.agents, [player1, player2]):
                if player.player_type == 'scripted':
                    rng_key, action_key = jax.random.split(rng_key)
                    actions[agent] = self.get_scripted_action(obs[agent], behavior, action_key)
                else:
                    if player.params is None:
                        self.load_checkpoint_player(player)
                    if player.apply_fn is None:
                        raise ValueError(f"Apply function not set for {player.name}")
                    rng_key, action_key = jax.random.split(rng_key)
                    network_output = player.apply_fn(player.params, obs[agent])
                    if isinstance(network_output, tuple):
                        pi, _ = network_output
                        actions[agent] = pi.sample(seed=action_key)
                    else:
                        actions[agent] = jax.random.categorical(action_key, network_output)

            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, _ = self.env.step(step_key, state, actions)

            for agent in self.env.agents:
                reward_log[agent].append(float(rewards[agent]))
                episode_rewards[agent] += float(rewards[agent])
            done_flag = bool(dones.get('__all__', False))
            done_log.append(done_flag)
            steps += 1
            if done_flag:
                done_all = True

        green_reward = episode_rewards["green"]
        red_reward = episode_rewards["red"]
        winner = "green" if green_reward > red_reward else "red" if red_reward > green_reward else "draw"

        if episode_id < 1:
            print(f"Episode {episode_id} Debug for {behavior}: G={green_reward}, R={red_reward}, steps={steps}")

        return {
            "episode_id": episode_id,
            "winner": winner,
            "green_player": player1.name,
            "red_player": player2.name,
            "behavior": behavior,
            "green_reward": green_reward,
            "red_reward": red_reward,
            "steps": steps,
            # Provenance - green (BR)
            "green_player_type": player1.player_type,
            "green_algorithm": player1.algorithm,
            "green_checkpoint_path": player1.checkpoint_path,
            "green_checkpoint_step": player1.checkpoint_step,
            "green_run_id": player1.run_id,
            "green_seed": player1.seed,
            "green_timesteps": player1.timesteps,
            # Provenance - red (opponent)
            "red_player_type": player2.player_type,
            "red_algorithm": player2.algorithm,
            "red_checkpoint_path": player2.checkpoint_path,
            "red_checkpoint_step": player2.checkpoint_step,
            "red_run_id": player2.run_id,
            "red_seed": player2.seed,
            "red_timesteps": player2.timesteps,
            # Evaluation run id
            "eval_run_id": self.run_timestamp,
        }

    def _evaluate_matchup(self, player1, opponent_player, behavior_label, rng_key):
        results = []
        green_wins = red_wins = draws = 0
        total_green_reward = 0.0
        total_red_reward = 0.0

        for episode in range(self.episodes):
            result = self.run_single_episode(player1, opponent_player, behavior_label, rng_key, episode)
            results.append(result)

            if result["winner"] == "green":
                green_wins += 1
            elif result["winner"] == "red":
                red_wins += 1
            else:
                draws += 1

            total_green_reward += result["green_reward"]
            total_red_reward += result["red_reward"]

            if (episode + 1) % 10 == 0:
                print(f"Completed {episode + 1}/{self.episodes} episodes for {behavior_label}")

        print(f"Summary {player1.name} vs {behavior_label}: G_wins={green_wins}, R_wins={red_wins}, draws={draws}")

        # Per-matchup metadata JSON
        metadata = {
            "eval_run_id": self.run_timestamp,
            "env_name": self.env_name,
            "episodes": self.episodes,
            "max_episode_steps": self.max_episode_steps,
            "behavior_label": behavior_label,
            "results_csv": str(self.output_dir / "evaluation_results.csv"),
            "green": {
                "name": player1.name,
                "player_type": player1.player_type,
                "algorithm": player1.algorithm,
                "checkpoint_path": player1.checkpoint_path,
                "checkpoint_step": player1.checkpoint_step,
                "run_id": player1.run_id,
                "seed": player1.seed,
                "timesteps": player1.timesteps,
            },
            "red": {
                "name": opponent_player.name,
                "player_type": opponent_player.player_type,
                "algorithm": opponent_player.algorithm,
                "checkpoint_path": opponent_player.checkpoint_path,
                "checkpoint_step": opponent_player.checkpoint_step,
                "run_id": opponent_player.run_id,
                "seed": opponent_player.seed,
                "timesteps": opponent_player.timesteps,
            },
            "summary": {
                "green_wins": green_wins,
                "red_wins": red_wins,
                "draws": draws,
                "avg_green_reward": total_green_reward / self.episodes,
                "avg_red_reward": total_red_reward / self.episodes,
            },
        }
        metadata_file = self.output_dir / f"matchup_{player1.name}_vs_{behavior_label}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Matchup metadata saved to: {metadata_file}")

        return results

    def _save_results(self, all_results):
        import pandas as pd  # ensure available if used standalone
        results_df = pd.DataFrame(all_results)
        results_file = self.output_dir / "evaluation_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nAll results saved to: {results_file}")

    # ---------- Summaries ----------
    @staticmethod
    def _compute_seed_summary_df(all_results_df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Compute per-seed, per-behavior summary from per-episode results."""
        import pandas as pd
        if all_results_df.empty:
            return pd.DataFrame(columns=[
                "behavior", "episodes", "win_rate", "loss_rate", "draw_rate",
                "avg_green_reward", "avg_red_reward"
            ])
        grouped = all_results_df.groupby("behavior")
        rows = []
        for behavior, df in grouped:
            episodes = len(df)
            win_rate = (df["winner"] == "green").mean()
            loss_rate = (df["winner"] == "red").mean()
            draw_rate = (df["winner"] == "draw").mean()
            avg_green_reward = df["green_reward"].mean()
            avg_red_reward = df["red_reward"].mean()
            rows.append({
                "behavior": behavior,
                "episodes": episodes,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "draw_rate": draw_rate,
                "avg_green_reward": avg_green_reward,
                "avg_red_reward": avg_red_reward,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _t_critical(df: int) -> float:
        """Approximate two-sided 95% t critical value for small df."""
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
            30: 2.042,
        }
        if df in table:
            return table[df]
        # interpolate crudely between known points
        keys = sorted(table.keys())
        if df < keys[0]:
            return table[keys[0]]
        if df > keys[-1]:
            return 1.96
        for i in range(len(keys) - 1):
            if keys[i] <= df <= keys[i+1]:
                x0, y0 = keys[i], table[keys[i]]
                x1, y1 = keys[i+1], table[keys[i+1]]
                # linear interpolation
                return y0 + (y1 - y0) * (df - x0) / (x1 - x0)
        return 1.96

    @classmethod
    def _aggregate_across_seeds(cls, seed_summaries: dict) -> 'pd.DataFrame':
        """Aggregate per-seed summaries into across-seed means and 95% CIs per behavior."""
        import pandas as pd
        # seed_summaries: {seed: DataFrame}
        all_behaviors = set()
        for df in seed_summaries.values():
            all_behaviors.update(df["behavior"].tolist())
        rows = []
        for behavior in sorted(all_behaviors):
            # collect per-seed metrics for this behavior
            metrics = {"win_rate": [], "loss_rate": [], "draw_rate": [],
                       "avg_green_reward": [], "avg_red_reward": []}
            seeds_present = []
            for seed, df in seed_summaries.items():
                row = df[df["behavior"] == behavior]
                if not row.empty:
                    seeds_present.append(seed)
                    for k in metrics.keys():
                        metrics[k].append(float(row.iloc[0][k]))
            n = len(seeds_present)
            if n == 0:
                continue
            dfree = max(n - 1, 1)
            tcrit = cls._t_critical(dfree)
            agg = {"behavior": behavior, "n_seeds": n}
            for k, vals in metrics.items():
                if len(vals) == 0:
                    continue
                mean = sum(vals) / len(vals)
                if len(vals) > 1:
                    # sample std
                    m = mean
                    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
                    se = math.sqrt(var / len(vals))
                    ci = tcrit * se
                else:
                    ci = float("nan")
                agg[k] = mean
                agg[f"{k}_ci95"] = ci
            rows.append(agg)
        import pandas as pd
        return pd.DataFrame(rows)


def _parse_seed_list(arg_val: str | None) -> list[int] | None:
    if arg_val is None:
        return None
    arg_val = arg_val.strip()
    if arg_val == "":
        return None
    parts = [p.strip() for p in arg_val.split(',') if p.strip() != ""]
    return [int(p) for p in parts]


def _determine_opponent_seed_for_index(opponent_seeds: list[int] | None, index: int, br_seed: int) -> tuple[str, int | None]:
    """Return (mode, seed) where mode in {match, fixed, zipped} and seed may be None for scripted."""
    if opponent_seeds is None:
        return ("match", br_seed)
    if len(opponent_seeds) == 1:
        return ("fixed", opponent_seeds[0])
    # zipped mode requires equal length; checked by caller
    return ("zipped", opponent_seeds[index])


def main():
    parser = argparse.ArgumentParser(description="Run BR matchups (multi-seed)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per matchup")
    parser.add_argument("--output-dir", type=str, default="experiments/results/br_eval", help="Base output directory for results")
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--br-seeds", type=str, default="0,1,2", help="Comma-separated BR checkpoint seeds to evaluate (e.g., 0,1,2)")
    parser.add_argument("--opponent-seeds", type=str, default="0", help="Learned opponent seeds. Default: fixed seed 0. Options: single value=fixed; list (same length as BR seeds)=zipped; omit to match BR seeds")
    parser.add_argument("--evaluation-seed", type=int, default=0, help="RNG seed for evaluation sampling")
    parser.add_argument("--only-scripted", action="store_true", help="Evaluate only scripted opponents")
    parser.add_argument("--only-learned", action="store_true", help="Evaluate only learned opponents")
    args = parser.parse_args()

    include_scripted = True
    include_learned = True
    if args.only_scripted:
        include_learned = False
    if args.only_learned:
        include_scripted = False

    br_seeds = _parse_seed_list(args.br_seeds) or [0, 1, 2]
    opponent_seeds = _parse_seed_list(args.opponent_seeds)
    if opponent_seeds is not None and len(opponent_seeds) > 1 and len(opponent_seeds) != len(br_seeds):
        raise ValueError("When providing multiple --opponent-seeds, it must have the same length as --br-seeds (zipped mode)")

    # Create run root dir
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir) / f"run_{run_timestamp}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries: dict[int, pd.DataFrame] = {}
    opponents_mode = "match" if opponent_seeds is None else ("fixed" if len(opponent_seeds) == 1 else "zipped")

    for idx, br_seed in enumerate(br_seeds):
        _, opp_seed = _determine_opponent_seed_for_index(opponent_seeds, idx, br_seed)

        evaluator = BREvaluator(
            episodes=args.episodes,
            output_dir=str(base_output_dir),
            max_episode_steps=args.max_episode_steps,
            seed=br_seed,
            include_scripted=include_scripted,
            include_learned=include_learned,
        )
        evaluator.output_dir = base_output_dir / f"seed_{br_seed}"
        evaluator.output_dir.mkdir(parents=True, exist_ok=True)

        results = evaluator.run(evaluation_seed=args.evaluation_seed, opponent_seed=opp_seed)
        seed_results_df = pd.DataFrame(results)
        seed_summary_df = evaluator._compute_seed_summary_df(seed_results_df)
        seed_summary_file = evaluator.output_dir / "seed_summary.csv"
        seed_summary_df.to_csv(seed_summary_file, index=False)
        seed_summaries[br_seed] = seed_summary_df

    seeds_meta = {
        "br_seeds": br_seeds,
        "opponent_seeds_mode": opponents_mode,
        "opponent_seeds": opponent_seeds,
        "evaluation_seed": args.evaluation_seed,
    }
    with open(base_output_dir / "seeds.json", "w") as f:
        json.dump(seeds_meta, f, indent=2)

    aggregated_df = BREvaluator._aggregate_across_seeds(seed_summaries)
    aggregated_df.to_csv(base_output_dir / "aggregated_summary.csv", index=False)
    with open(base_output_dir / "aggregated_summary.json", "w") as f:
        json.dump(aggregated_df.to_dict(orient="records"), f, indent=2)

    print(f"\nAggregated summary written to: {base_output_dir}")


if __name__ == "__main__":
    main()

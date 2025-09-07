#!/usr/bin/env python3
"""
Batch training script for Best-Response (BR) agents.
Trains BR agents against all discovered learned opponents (IPPO, SPPPO, FSPPPO).
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Any

# Import opponent discovery utilities
try:
    from baselines.BR.opponent_discovery import discover_learned_opponents
except ImportError:
    from BR.opponent_discovery import discover_learned_opponents


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def create_learned_br_training_jobs(
    training_seed: int = 0
) -> List[Dict[str, Any]]:
    """
    Create BR training job configurations for all discovered learned opponents.
    
    Args:
        training_seed: Training seed to discover opponents for
        
    Returns:
        List of training job configurations
    """
    # Discover all available learned opponents
    opponents = discover_learned_opponents(
        training_seed=training_seed,
        latest_only=True
    )
    
    training_jobs = []
    
    for opponent in opponents:
        # Create job config for this opponent
        job_config = {
            "opponent_type": opponent.algorithm,
            "opponent_checkpoint": opponent.checkpoint_path,
            "opponent_name": opponent.name,
            "opponent_description": f"{opponent.algorithm} seed={opponent.seed}",
            "training_seed": training_seed,
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "is_scripted": False
        }
        
        training_jobs.append(job_config)
    
    logging.info(f"Created {len(training_jobs)} learned opponent jobs for discovery seed {training_seed}")
    return training_jobs


def create_scripted_br_training_jobs(
    training_seed: int = 0
) -> List[Dict[str, Any]]:
    """
    Create BR training job configurations for all scripted opponents.
    
    Args:
        training_seed: Training seed to use
        
    Returns:
        List of training job configurations
    """
    # Define all available scripted opponents
    scripted_opponents = [
        "noop",      # Does nothing
        "random",    # Takes random actions
        "seek",      # Moves toward opponent
        "guardian",  # Moves toward center
        "dodge",     # Moves away from opponent
    ]
    
    training_jobs = []
    
    for opponent in scripted_opponents:
        # Create job config for this scripted opponent
        job_config = {
            "opponent_type": opponent,
            "opponent_checkpoint": None,  # No checkpoint for scripted opponents
            "opponent_name": f"scripted_{opponent}",
            "opponent_description": f"Scripted {opponent} behavior",
            "training_seed": training_seed,
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "is_scripted": True
        }
        
        training_jobs.append(job_config)
    
    logging.info(f"Created {len(training_jobs)} scripted opponent jobs")
    return training_jobs


def run_br_training_job(job_config: Dict[str, Any], num_seeds: int) -> bool:
    """
    Run a single BR training job using subprocess.
    
    Args:
        job_config: Training job configuration
        num_seeds: Number of parallel seeds to train
        
    Returns:
        True if successful, False otherwise
    """
    try:
        opponent_name = job_config["opponent_name"]
        opponent_type = job_config["opponent_type"]
        opponent_checkpoint = job_config["opponent_checkpoint"]
        training_seed = job_config["training_seed"]  # opponent discovery seed
        is_scripted = job_config.get("is_scripted", False)
        
        logging.info(f"Starting BR training vs {opponent_name} ({opponent_type}) - {num_seeds} seeds in parallel")
        
        # Build command to run BR training with full timesteps
        # Hydra requires arguments in format key=value (no spaces)
        cmd = [
            "python", "-m", "baselines.BR.train",
            "--config-name=br_ff_mpe"
        ]
        
        # Add all parameters as separate arguments with proper prefix
        hydra_args = [
            f"SEED={training_seed}",
            f"NUM_SEEDS={num_seeds}",
            f"RUN_NAME=br_vs_{opponent_name.replace('_', '-')}",
        ]
        
        # Add total timesteps override if provided
        if job_config.get('total_timesteps_override'):
            hydra_args.append(f"TOTAL_TIMESTEPS={job_config['total_timesteps_override']}")
        
        # Add appropriate parameters based on opponent type
        if is_scripted:
            # For scripted opponents, use the opponent type directly
            hydra_args.append(f"TARGET_ALGORITHM={opponent_type.lower()}")  # Use lowercase for scripted types
        else:
            # For learned opponents, include the checkpoint path
            hydra_args.append(f"TARGET_ALGORITHM={opponent_type.lower()}")  # Use lowercase for algorithm names
            # For Hydra, we need to properly format the path
            # Use quotes around the path to handle spaces and special characters
            # Escape any special characters in the path
            hydra_args.append(f"OPPONENT_CHECKPOINT_PATH={opponent_checkpoint}")
        
        # Add all Hydra args to command
        cmd.extend(hydra_args)
        
        # Print the command for debugging
        cmd_str = ' '.join(cmd)
        logging.info(f"Running command: {cmd_str}")
        
        # Run the training command
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logging.info(f"✅ BR training vs {opponent_name} completed successfully")
            return True
        else:
            logging.error(f"❌ BR training vs {opponent_name} failed:")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"❌ BR training vs {opponent_name} timed out")
        return False
    except Exception as e:
        opponent_name = job_config.get("opponent_name", "unknown")
        logging.error(f"❌ BR training vs {opponent_name} failed: {e}")
        return False


def main():
    """Main batch training function."""
    parser = argparse.ArgumentParser(
        description="Batch train BR agents against all learned and scripted opponents"
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=0,
        help="Baseline seed to discover opponent checkpoints from"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Directory containing learned opponent checkpoints"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of parallel BR seeds to train per opponent"
    )
    parser.add_argument(
        "--total-timesteps",
        type=str,
default=None,
        help="Override total timesteps from config (e.g., 1e8). If not specified, uses config value."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show training jobs without executing them"
    )
    parser.add_argument(
        "--skip-learned",
        action="store_true",
        help="Skip training against learned opponents"
    )
    parser.add_argument(
        "--skip-scripted",
        action="store_true",
        help="Skip training against scripted opponents"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("BATCH BR TRAINING")
    logging.info("=" * 60)
    logging.info(f"Opponent discovery seed: {args.training_seed}")
    logging.info(f"Checkpoints directory: {args.checkpoints_dir}")
    logging.info(f"Parallel seeds per opponent: {args.num_seeds}")
    if args.total_timesteps:
        logging.info(f"Total timesteps override: {args.total_timesteps}")
    else:
        logging.info(f"Using config timesteps (no override specified)")
    logging.info("=" * 60)
    
    # Create training jobs
    all_training_jobs = []
    
    # Add learned opponent jobs if not skipped
    if not args.skip_learned:
        try:
            learned_jobs = create_learned_br_training_jobs(
                training_seed=args.training_seed
            )
            all_training_jobs.extend(learned_jobs)
            
            # Show learned job summary
            if learned_jobs:
                logging.info(f"\nDiscovered {len(learned_jobs)} learned opponents:")
                for i, job in enumerate(learned_jobs, 1):
                    opponent_name = job["opponent_name"]
                    opponent_type = job["opponent_type"]
                    description = job["opponent_description"]
                    logging.info(f"  {i:2d}. {opponent_name} ({opponent_type}) - {description}")
            else:
                logging.warning("No learned opponents found")
                
        except Exception as e:
            logging.error(f"Failed to create learned opponent jobs: {e}")
    else:
        logging.info("Skipping learned opponents as requested")
    
    # Add scripted opponent jobs if not skipped
    if not args.skip_scripted:
        try:
            scripted_jobs = create_scripted_br_training_jobs(
                training_seed=args.training_seed
            )
            all_training_jobs.extend(scripted_jobs)
            
            # Show scripted job summary
            if scripted_jobs:
                logging.info(f"\nAdded {len(scripted_jobs)} scripted opponents:")
                for i, job in enumerate(scripted_jobs, 1):
                    opponent_name = job["opponent_name"]
                    opponent_type = job["opponent_type"]
                    description = job["opponent_description"]
                    logging.info(f"  {i:2d}. {opponent_name} ({opponent_type}) - {description}")
            
        except Exception as e:
            logging.error(f"Failed to create scripted opponent jobs: {e}")
    else:
        logging.info("Skipping scripted opponents as requested")
    
    # Check if we have any jobs to run
    if not all_training_jobs:
        logging.warning("No training jobs created - no opponents found or all types skipped")
        return 1
    
    if args.dry_run:
        logging.info(f"\nDry run complete - would run {len(all_training_jobs)} training jobs")
        return 0
    
    # Execute training jobs
    logging.info(f"\nStarting batch BR training for {len(all_training_jobs)} opponents...")
    
    successful_jobs = 0
    failed_jobs = 0
    
    # Run jobs sequentially
    for i, job_config in enumerate(all_training_jobs, 1):
        opponent_name = job_config["opponent_name"]
        opponent_type = job_config["opponent_type"]
        is_scripted = job_config.get("is_scripted", False)
        job_type = "scripted" if is_scripted else "learned"
        
        logging.info(f"\n[{i}/{len(all_training_jobs)}] Training BR vs {opponent_name} ({job_type}, {args.num_seeds} seeds)")
        
        # Add total timesteps override if specified
        if args.total_timesteps:
            job_config['total_timesteps_override'] = args.total_timesteps
            logging.info(f"Using override {args.total_timesteps} timesteps for {job_config['opponent_name']}")
        else:
            logging.info(f"Using config timesteps for {job_config['opponent_name']}")
        
        success = run_br_training_job(
            job_config,
            num_seeds=args.num_seeds,
        )
        if success:
            successful_jobs += 1
        else:
            failed_jobs += 1
    
    # Final summary
    logging.info("\n" + "=" * 60)
    logging.info("BATCH BR TRAINING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Total jobs: {len(all_training_jobs)}")
    logging.info(f"Successful: {successful_jobs}")
    logging.info(f"Failed: {failed_jobs}")
    logging.info("=" * 60)
    
    return 0 if failed_jobs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Sequential Training Script for All Baseline Algorithms.

This script trains IPPO, SPPPO, and FSPPPO sequentially with consistent configurations
to generate fresh checkpoints for comprehensive tournament evaluation.

Features:
- Sequential training of all three baseline algorithms
- Consistent training parameters across algorithms
- Automatic checkpoint generation at regular intervals
- Progress tracking and logging
- Configurable training duration and checkpoint frequency

Usage:
    python -m baselines.train_all_baselines --config train_all_config.yaml
    python -m baselines.train_all_baselines --quick-test  # For testing
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def get_timestamp() -> str:
    """Get current timestamp for run identification."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_training_config(
    algorithm: str,
    base_config: Dict,
    run_timestamp: str,
    quick_test: bool = False
) -> Dict:
    """Create training configuration for specific algorithm."""
    
    # Base configuration for all algorithms
    config = {
        "ENV_NAME": "MPE_simple_sumo_v3",
        "ENV_KWARGS": {"random_spawn": True},
        "ACTIVATION": "tanh",
        "SEED": 42,
        "NUM_SEEDS": 5,  # Multiple seeds for statistical robustness
        
        # Training parameters
        "LR": 2.5e-4,
        "ANNEAL_LR": True,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 0.5,
        
        # Checkpoint configuration
        "CHECKPOINT_FREQ": 100,  # Save every 100 iterations
        "SAVE_AT_END": True,
        "MAX_CHECKPOINTS_TO_KEEP": 20,
        
        # Logging
        "WANDB_MODE": "disabled",
        "LOG_EVERY": 10,
    }
    
    # Override with base config if provided
    if base_config:
        config.update(base_config)
    
    # Quick test configuration
    if quick_test:
        config.update({
            "TOTAL_TIMESTEPS": 50000,  # ~39 iterations
            "NUM_SEEDS": 2,
            "NUM_ENVS": 4,
            "CHECKPOINT_FREQ": 20,
            "MAX_CHECKPOINTS_TO_KEEP": 5,
        })
    else:
        # Full training configuration
        config.update({
            "TOTAL_TIMESTEPS": 2e6,  # 2M timesteps (~976 iterations)
        })
    
    # Algorithm-specific configurations
    if algorithm == "IPPO":
        config.update({
            "ENTITY": "jaxmarl",  # Required for wandb (even when disabled)
            "PROJECT": "jaxmarl-mpe",  # Required for wandb
        })
    elif algorithm == "FSPPPO":
        config.update({
            "OPPONENT_SAMPLING_FREQ": 200,  # Sample new opponent every 200 iterations
            "SELF_PLAY_PROBABILITY": 0.3,
            "RECENCY_BIAS_ALPHA": 0.6,
        })
    
    return config


def save_config_file(config: Dict, filepath: str) -> None:
    """Save configuration to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def run_training(
    algorithm: str,
    config_dict: Dict,
    run_timestamp: str,
    log_file: Optional[str] = None
) -> bool:
    """Run training for specific algorithm."""
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {algorithm} Training")
    print(f"{'='*60}")
    print(f"Log file: {log_file or 'stdout'}")
    
    # Determine the training script path (use standalone training scripts)
    script_configs = {
        "IPPO": {
            "script": "baselines.IPPO.train",
            "standalone": True  # Uses internal config, no external config needed
        },
        "SPPPO": {
            "script": "baselines.SPPPO.train",
            "standalone": True  # Uses internal config, no external config needed
        },
        "FSPPPO": {
            "script": "baselines.FSPPPO.train",
            "standalone": True  # Uses internal config, no external config needed
        },
    }
    
    if algorithm not in script_configs:
        print(f"❌ Unknown algorithm: {algorithm}")
        return False
    
    script_info = script_configs[algorithm]
    
    # Build command (standalone scripts don't need external config files)
    cmd = ["python", "-m", script_info["script"]]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Note: {algorithm} uses standalone training script with internal config")
    
    # Run training
    start_time = time.time()
    
    try:
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd="/share/code/src/JaxMARL"
                )
        else:
            result = subprocess.run(
                cmd,
                text=True,
                cwd="/share/code/src/JaxMARL"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # No cleanup needed for standalone scripts
        
        if result.returncode == 0:
            print(f"✅ {algorithm} training completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            return True
        else:
            print(f"❌ {algorithm} training failed with return code {result.returncode}")
            if log_file:
                print(f"Check log file for details: {log_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {algorithm} training: {e}")
        # No cleanup needed for standalone scripts
        return False


def check_checkpoints(algorithm: str, run_timestamp: str) -> List[str]:
    """Check what checkpoints were created for an algorithm."""
    
    checkpoint_patterns = {
        "IPPO": f"checkpoints/ippo/run_{run_timestamp}_seed*/agent_*/*/",
        "SPPPO": f"checkpoints/spppo/run_{run_timestamp}_seed*/*/",
        "FSPPPO": f"checkpoints/fspppo/run_{run_timestamp}_seed*/main_agent/step_*/",
    }
    
    import glob
    pattern = checkpoint_patterns.get(algorithm, "")
    checkpoints = glob.glob(pattern)
    
    print(f"📁 {algorithm} checkpoints found: {len(checkpoints)}")
    if checkpoints:
        for cp in sorted(checkpoints)[:5]:  # Show first 5
            print(f"   {cp}")
        if len(checkpoints) > 5:
            print(f"   ... and {len(checkpoints) - 5} more")
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Train all baseline algorithms sequentially")
    parser.add_argument("--config", help="Base configuration YAML file")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with reduced parameters")
    parser.add_argument("--algorithms", nargs="+", default=["IPPO", "SPPPO", "FSPPPO"], 
                       help="Algorithms to train")
    parser.add_argument("--output-dir", default="training_runs", help="Output directory for configs and logs")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just check existing checkpoints")
    
    args = parser.parse_args()
    
    # Load base configuration if provided
    base_config = {}
    if args.config:
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate run timestamp
    run_timestamp = get_timestamp()
    
    print("🎯 Sequential Baseline Algorithm Training")
    print("=" * 60)
    print(f"Run timestamp: {run_timestamp}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Quick test mode: {args.quick_test}")
    print(f"Output directory: {output_dir}")
    
    # Training results
    results = {}
    
    if not args.skip_training:
        # Train each algorithm sequentially
        for algorithm in args.algorithms:
            print(f"\n🔄 Preparing {algorithm} training...")
            
            # Create algorithm-specific config
            config = create_training_config(
                algorithm=algorithm,
                base_config=base_config,
                run_timestamp=run_timestamp,
                quick_test=args.quick_test
            )
            
            # Save config file for reference
            config_file = output_dir / f"{algorithm.lower()}_config_{run_timestamp}.yaml"
            save_config_file(config, str(config_file))
            
            # Set up log file
            log_file = output_dir / f"{algorithm.lower()}_training_{run_timestamp}.log"
            
            # Run training
            success = run_training(
                algorithm=algorithm,
                config_dict=config,
                run_timestamp=run_timestamp,
                log_file=str(log_file)
            )
            
            results[algorithm] = {
                "success": success,
                "config_file": str(config_file),
                "log_file": str(log_file),
            }
            
            if not success:
                print(f"⚠️  {algorithm} training failed, continuing with next algorithm...")
    
    # Check generated checkpoints
    print(f"\n📊 Checkpoint Summary")
    print("=" * 40)
    
    for algorithm in args.algorithms:
        checkpoints = check_checkpoints(algorithm, run_timestamp)
        if algorithm in results:
            results[algorithm]["checkpoints"] = len(checkpoints)
    
    # Generate summary
    print(f"\n📋 Training Summary")
    print("=" * 40)
    
    if not args.skip_training:
        for algorithm, result in results.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            checkpoint_count = result.get("checkpoints", 0)
            print(f"{algorithm:8} {status:10} {checkpoint_count:3d} checkpoints")
    
    # Generate next steps
    print(f"\n🎯 Next Steps")
    print("=" * 30)
    print("1. Run tournament evaluation:")
    print(f"   python -m baselines.tournament_eval --config tournament_config.yaml")
    print("\n2. Update tournament_config.yaml with new checkpoint paths:")
    
    for algorithm in args.algorithms:
        checkpoints = check_checkpoints(algorithm, run_timestamp)
        if checkpoints:
            example_checkpoint = sorted(checkpoints)[0]
            print(f"   - \"{algorithm}:{example_checkpoint}\"")
    
    print(f"\n3. Analyze results and generate research artifacts")
    
    # Save run summary
    summary_file = output_dir / f"training_summary_{run_timestamp}.yaml"
    summary = {
        "run_timestamp": run_timestamp,
        "algorithms": args.algorithms,
        "quick_test": args.quick_test,
        "results": results,
    }
    
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    print(f"\n💾 Training summary saved to: {summary_file}")
    
    # Exit code based on results
    if not args.skip_training:
        failed_algorithms = [alg for alg, result in results.items() if not result["success"]]
        if failed_algorithms:
            print(f"\n⚠️  Some algorithms failed: {failed_algorithms}")
            sys.exit(1)
        else:
            print(f"\n🎉 All algorithms trained successfully!")
    
    print(f"\n🚀 Ready for comprehensive tournament evaluation!")


if __name__ == "__main__":
    main()

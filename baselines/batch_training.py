#!/usr/bin/env python3
"""Batch Training Script for All Baseline Algorithms.

This script trains IPPO, SPPPO, and FSPPPO sequentially using their standalone
training scripts. It focuses purely on training and checkpoint generation,
with evaluation handled separately by run_comprehensive_evaluation.py.

Features:
- Sequential training of all baseline algorithms
- Uses existing standalone training scripts (train_ippo.py, train_spppo.py, train_fspppo.py)
- Progress tracking and logging
- Checkpoint discovery and validation
- Clean separation of training and evaluation concerns

Usage:
    python -m baselines.batch_training --algorithms IPPO SPPPO FSPPPO
    python -m baselines.batch_training --algorithms IPPO  # Train only IPPO
    python -m baselines.batch_training --quick-test       # Quick validation
"""

import argparse
import datetime
import subprocess
import sys
import time
import glob
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def get_timestamp() -> str:
    """Get current timestamp for run identification."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_algorithm_training(algorithm: str, log_file: Optional[str] = None) -> bool:
    """Run training for a specific algorithm using its standalone script with standardized config."""
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {algorithm} Training")
    print(f"{'='*60}")
    
    # Map algorithms to their training scripts and standardized configs
    training_configs = {
        "IPPO": {
            "script": "baselines.IPPO.ippo_ff_mpe",
            "config_name": "ippo_batch_training"
        },
        "SPPPO": {
            "script": "baselines.SPPPO.spppo_ff_mpe", 
            "config_name": "spppo_batch_training"
        },
        "FSPPPO": {
            "script": "baselines.FSPPPO.fspppo_ff_mpe",
            "config_name": "fspppo_batch_training"
        },
    }
    
    if algorithm not in training_configs:
        print(f"❌ Unknown algorithm: {algorithm}")
        return False
    
    config = training_configs[algorithm]
    cmd = [
        "python", "-m", config["script"],
        "--config-name", config["config_name"]
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Config: {config['config_name']}.yaml (standardized for fair comparison)")
    print(f"Log file: {log_file or 'stdout'}")
    
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
        return False


def discover_recent_checkpoints(algorithm: str, hours_back: int = 2) -> List[str]:
    """Discover recently created checkpoints for an algorithm."""
    
    # Checkpoint discovery patterns for different algorithms
    # All follow: checkpoints/{algorithm}/run_*_seed*/{agent_type}/{checkpoint_files}
    patterns = {
        "IPPO": "checkpoints/ippo/run_*_seed*/agent_*/*/",  # Two agents: agent_0, agent_1
        "SPPPO": "checkpoints/spppo/run_*_seed*/shared_agent/*/",  # One shared agent
        "FSPPPO": "checkpoints/fspppo/run_*_seed*/main_agent/step_*/",  # One main agent
    }
    
    pattern = patterns.get(algorithm, "")
    if not pattern:
        return []
    
    # Find all matching checkpoints
    all_checkpoints = glob.glob(pattern)
    
    # Filter by modification time (recent checkpoints)
    recent_checkpoints = []
    cutoff_time = time.time() - (hours_back * 3600)
    
    for checkpoint in all_checkpoints:
        try:
            if Path(checkpoint).stat().st_mtime > cutoff_time:
                recent_checkpoints.append(checkpoint)
        except OSError:
            continue
    
    return sorted(recent_checkpoints)


def validate_training_results(algorithms: List[str]) -> Dict[str, Dict]:
    """Validate that training produced expected checkpoints."""
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n📁 Validating {algorithm} checkpoints...")
        
        # Discover recent checkpoints
        recent_checkpoints = discover_recent_checkpoints(algorithm)
        
        # Count checkpoints
        checkpoint_count = len(recent_checkpoints)
        
        # Determine success based on checkpoint count
        success = checkpoint_count > 0
        
        results[algorithm] = {
            "success": success,
            "checkpoint_count": checkpoint_count,
            "checkpoints": recent_checkpoints[:5],  # Show first 5
        }
        
        # Print results
        status = "✅ SUCCESS" if success else "❌ NO CHECKPOINTS"
        print(f"  {algorithm}: {status} - {checkpoint_count} checkpoints found")
        
        if recent_checkpoints:
            print("  Recent checkpoints:")
            for cp in recent_checkpoints[:3]:  # Show first 3
                print(f"    {cp}")
            if len(recent_checkpoints) > 3:
                print(f"    ... and {len(recent_checkpoints) - 3} more")
        
    return results


def save_training_summary(
    algorithms: List[str], 
    results: Dict[str, Dict], 
    run_timestamp: str,
    output_dir: str
) -> str:
    """Save a summary of the training session."""
    
    summary = {
        "run_timestamp": run_timestamp,
        "training_date": datetime.datetime.now().isoformat(),
        "algorithms_trained": algorithms,
        "results": results,
        "total_checkpoints": sum(r["checkpoint_count"] for r in results.values()),
        "successful_algorithms": [alg for alg, r in results.items() if r["success"]],
        "failed_algorithms": [alg for alg, r in results.items() if not r["success"]],
    }
    
    # Save summary
    summary_file = Path(output_dir) / f"batch_training_summary_{run_timestamp}.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    return str(summary_file)


def main():
    parser = argparse.ArgumentParser(description="Batch training for baseline algorithms")
    parser.add_argument(
        "--algorithms", 
        nargs="+", 
        default=["IPPO", "SPPPO", "FSPPPO"],
        choices=["IPPO", "SPPPO", "FSPPPO"],
        help="Algorithms to train sequentially"
    )
    parser.add_argument(
        "--output-dir", 
        default="batch_training_logs", 
        help="Output directory for logs and summaries"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true", 
        help="Quick test mode (same as normal, but indicates testing)"
    )
    parser.add_argument(
        "--skip-validation", 
        action="store_true", 
        help="Skip checkpoint validation after training"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate run timestamp
    run_timestamp = get_timestamp()
    
    print("🎯 Batch Training for Baseline Algorithms")
    print("=" * 60)
    print(f"Run timestamp: {run_timestamp}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Quick test mode: {args.quick_test}")
    print(f"Output directory: {output_dir}")
    
    # Training results
    training_results = {}
    
    # Train each algorithm sequentially
    for algorithm in args.algorithms:
        print(f"\n🔄 Preparing {algorithm} training...")
        
        # Set up log file
        log_file = output_dir / f"{algorithm.lower()}_training_{run_timestamp}.log"
        
        # Run training
        success = run_algorithm_training(
            algorithm=algorithm,
            log_file=str(log_file)
        )
        
        training_results[algorithm] = {
            "success": success,
            "log_file": str(log_file),
        }
        
        if not success:
            print(f"⚠️  {algorithm} training failed, continuing with next algorithm...")
    
    # Validate training results
    if not args.skip_validation:
        print(f"\n📊 Validating Training Results")
        print("=" * 40)
        
        validation_results = validate_training_results(args.algorithms)
        
        # Merge training and validation results
        for algorithm in args.algorithms:
            if algorithm in training_results and algorithm in validation_results:
                training_results[algorithm].update(validation_results[algorithm])
    
    # Generate summary
    print(f"\n📋 Batch Training Summary")
    print("=" * 40)
    
    successful_algorithms = []
    failed_algorithms = []
    
    for algorithm, result in training_results.items():
        training_success = result.get("success", False)
        checkpoint_count = result.get("checkpoint_count", 0)
        
        if training_success and checkpoint_count > 0:
            status = "✅ SUCCESS"
            successful_algorithms.append(algorithm)
        else:
            status = "❌ FAILED"
            failed_algorithms.append(algorithm)
        
        print(f"{algorithm:8} {status:10} {checkpoint_count:3d} checkpoints")
    
    # Save training summary
    summary_file = save_training_summary(
        algorithms=args.algorithms,
        results=training_results,
        run_timestamp=run_timestamp,
        output_dir=str(output_dir)
    )
    
    print(f"\n💾 Training summary saved to: {summary_file}")
    
    # Generate next steps
    print(f"\n🎯 Next Steps")
    print("=" * 30)
    
    if successful_algorithms:
        print("1. Run comprehensive evaluation:")
        print("   python -m baselines.run_comprehensive_evaluation --auto-discover")
        print("\n2. Or run evaluation with specific timestamp:")
        print(f"   python -m baselines.run_comprehensive_evaluation --run-timestamp {run_timestamp}")
        
        print(f"\n3. Successful algorithms ready for evaluation:")
        for alg in successful_algorithms:
            print(f"   ✅ {alg}")
    
    if failed_algorithms:
        print(f"\n⚠️  Failed algorithms (check logs):")
        for alg in failed_algorithms:
            log_file = training_results[alg].get("log_file", "")
            print(f"   ❌ {alg}: {log_file}")
    
    # Exit code based on results
    if failed_algorithms:
        print(f"\n⚠️  Some algorithms failed: {failed_algorithms}")
        if not successful_algorithms:
            print("❌ No algorithms trained successfully!")
            sys.exit(1)
        else:
            print(f"✅ {len(successful_algorithms)} algorithms trained successfully")
    else:
        print(f"\n🎉 All {len(successful_algorithms)} algorithms trained successfully!")
    
    print(f"\n🚀 Ready for comprehensive evaluation!")


if __name__ == "__main__":
    main()

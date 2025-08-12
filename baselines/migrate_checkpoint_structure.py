#!/usr/bin/env python3
"""
Migration script to standardize checkpoint folder structure across all baseline algorithms.

This script migrates existing checkpoints from the old structure to the new standardized structure:

Old Structure:
- IPPO: checkpoints/ippo/run_*_seed*/agent_0/, agent_1/
- SPPPO: checkpoints/spppo/run_*_seed*/shared_agent/
- FSPPPO: checkpoints/fspppo/run_*_seed*/main_agent/

New Standardized Structure:
- IPPO: checkpoints/ippo/run_*_seed*/main/, opponent/
- SPPPO: checkpoints/spppo/run_*_seed*/main/
- FSPPPO: checkpoints/fspppo/run_*_seed*/main/

Usage:
    python migrate_checkpoint_structure.py [--dry-run] [--checkpoint-dir CHECKPOINT_DIR]
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def find_checkpoint_runs(checkpoint_dir: Path, algorithm: str) -> List[Path]:
    """Find all checkpoint runs for a specific algorithm."""
    algo_dir = checkpoint_dir / algorithm
    if not algo_dir.exists():
        return []

    runs = []
    for item in algo_dir.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            runs.append(item)

    return sorted(runs)


def migrate_ippo_checkpoints(run_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Migrate IPPO checkpoints from agent_0/agent_1 to main/opponent structure.

    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    successful = 0
    failed = 0

    agent_0_dir = run_dir / "agent_0"
    agent_1_dir = run_dir / "agent_1"
    main_dir = run_dir / "main"
    opponent_dir = run_dir / "opponent"

    # Check if old structure exists
    if not (agent_0_dir.exists() and agent_1_dir.exists()):
        print(f"  No IPPO migration needed for {run_dir.name}")
        return successful, failed

    # Check if new structure already exists
    if main_dir.exists() or opponent_dir.exists():
        print(f"  IPPO migration already done for {run_dir.name}")
        return successful, failed

    try:
        print(f"  Migrating IPPO: {run_dir.name}")
        print(f"    agent_0 -> main")
        print(f"    agent_1 -> opponent")

        if not dry_run:
            # Rename directories
            agent_0_dir.rename(main_dir)
            agent_1_dir.rename(opponent_dir)

        successful += 2

    except Exception as e:
        print(f"  ERROR migrating IPPO {run_dir.name}: {e}")
        failed += 2

    return successful, failed


def migrate_spppo_checkpoints(run_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Migrate SPPPO checkpoints from shared_agent to main structure.

    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    successful = 0
    failed = 0

    shared_agent_dir = run_dir / "shared_agent"
    main_dir = run_dir / "main"

    # Check if old structure exists
    if not shared_agent_dir.exists():
        print(f"  No SPPPO migration needed for {run_dir.name}")
        return successful, failed

    # Check if new structure already exists
    if main_dir.exists():
        print(f"  SPPPO migration already done for {run_dir.name}")
        return successful, failed

    try:
        print(f"  Migrating SPPPO: {run_dir.name}")
        print(f"    shared_agent -> main")

        if not dry_run:
            # Rename directory
            shared_agent_dir.rename(main_dir)

        successful += 1

    except Exception as e:
        print(f"  ERROR migrating SPPPO {run_dir.name}: {e}")
        failed += 1

    return successful, failed


def migrate_fspppo_checkpoints(run_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Migrate FSPPPO checkpoints from main_agent to main structure.

    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    successful = 0
    failed = 0

    main_agent_dir = run_dir / "main_agent"
    main_dir = run_dir / "main"

    # Check if old structure exists
    if not main_agent_dir.exists():
        print(f"  No FSPPPO migration needed for {run_dir.name}")
        return successful, failed

    # Check if new structure already exists
    if main_dir.exists():
        print(f"  FSPPPO migration already done for {run_dir.name}")
        return successful, failed

    try:
        print(f"  Migrating FSPPPO: {run_dir.name}")
        print(f"    main_agent -> main")

        if not dry_run:
            # Rename directory
            main_agent_dir.rename(main_dir)

        successful += 1

    except Exception as e:
        print(f"  ERROR migrating FSPPPO {run_dir.name}: {e}")
        failed += 1

    return successful, failed


def migrate_algorithm_checkpoints(
    checkpoint_dir: Path,
    algorithm: str,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Migrate all checkpoints for a specific algorithm.

    Returns:
        Tuple of (successful_migrations, failed_migrations)
    """
    print(f"\nMigrating {algorithm.upper()} checkpoints...")

    runs = find_checkpoint_runs(checkpoint_dir, algorithm)
    if not runs:
        print(f"  No {algorithm.upper()} checkpoint runs found")
        return 0, 0

    total_successful = 0
    total_failed = 0

    for run_dir in runs:
        if algorithm == "ippo":
            successful, failed = migrate_ippo_checkpoints(run_dir, dry_run)
        elif algorithm == "spppo":
            successful, failed = migrate_spppo_checkpoints(run_dir, dry_run)
        elif algorithm == "fspppo":
            successful, failed = migrate_fspppo_checkpoints(run_dir, dry_run)
        else:
            print(f"  Unknown algorithm: {algorithm}")
            continue

        total_successful += successful
        total_failed += failed

    print(f"  {algorithm.upper()} migration summary: {total_successful} successful, {total_failed} failed")
    return total_successful, total_failed


def main():
    parser = argparse.ArgumentParser(
        description="Migrate checkpoint folder structure to standardized format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually moving files"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Base checkpoint directory (default: checkpoints)"
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory does not exist: {checkpoint_dir}")
        return 1

    print(f"Checkpoint Migration Script")
    print(f"===========================")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Dry run: {'Yes' if args.dry_run else 'No'}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be moved ***")

    # Migrate each algorithm
    algorithms = ["ippo", "spppo", "fspppo"]
    total_successful = 0
    total_failed = 0

    for algorithm in algorithms:
        successful, failed = migrate_algorithm_checkpoints(
            checkpoint_dir, algorithm, args.dry_run
        )
        total_successful += successful
        total_failed += failed

    # Summary
    print(f"\nMigration Summary")
    print(f"=================")
    print(f"Total successful migrations: {total_successful}")
    print(f"Total failed migrations: {total_failed}")

    if total_failed > 0:
        print(f"\nWARNING: {total_failed} migrations failed. Please check the errors above.")
        return 1

    if total_successful == 0:
        print("\nNo migrations were needed. All checkpoints are already using the standardized structure.")
    else:
        if not args.dry_run:
            print(f"\nMigration completed successfully! {total_successful} directories were renamed.")
        else:
            print(f"\nDry run completed. {total_successful} directories would be renamed.")

    return 0


if __name__ == "__main__":
    exit(main())

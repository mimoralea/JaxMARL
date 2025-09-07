"""
Opponent Discovery Module for Best-Response (BR) Training

This module automatically discovers available learned opponents (IPPO, SPPPO, FSPPPO)
from checkpoint directories, using the same logic as the tournament evaluation system.
"""

import os
import glob
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpponentInfo:
    """Information about a discovered opponent."""
    name: str
    algorithm: str
    checkpoint_path: str
    seed: int
    step: Optional[str] = None
    
    def __str__(self):
        return f"{self.name} ({self.algorithm}, seed={self.seed})"


class OpponentDiscovery:
    """Discovers available learned opponents for BR training."""
    
    def __init__(self, training_seed: int = 0):
        """Initialize opponent discovery.
        
        Args:
            training_seed: Training seed to use for checkpoint discovery
        """
        self.training_seed = training_seed
        
    def discover_all_opponents(self, latest_only: bool = True) -> List[OpponentInfo]:
        """Discover all available learned opponents.
        
        Args:
            latest_only: If True, only return the most recent checkpoint for each algorithm
            
        Returns:
            List of discovered opponents
        """
        opponents = []
        
        # Discover each algorithm type
        opponents.extend(self._discover_ippo_opponents(latest_only))
        opponents.extend(self._discover_spppo_opponents(latest_only))
        opponents.extend(self._discover_fspppo_opponents(latest_only))
        
        print(f"Discovered {len(opponents)} learned opponents:")
        for opponent in opponents:
            print(f"  - {opponent}")
            
        return opponents
    
    def _discover_ippo_opponents(self, latest_only: bool = True) -> List[OpponentInfo]:
        """Discover IPPO opponents."""
        # Try both experiments and checkpoints directories
        patterns = [
            f"experiments/checkpoints/ippo/run_*_seed{self.training_seed}/main/*",
            f"checkpoints/ippo/run_*_seed{self.training_seed}/main/*"
        ]
        
        ippo_paths = []
        for pattern in patterns:
            ippo_paths.extend(glob.glob(pattern))
        
        # Filter out temporary/incomplete checkpoints
        ippo_paths = [path for path in ippo_paths if not path.endswith('.orbax-checkpoint-tmp-0')]
            
        if not ippo_paths:
            return []
            
        if latest_only:
            # Find the most recent checkpoint
            latest_path = max(ippo_paths, key=os.path.getmtime)
            return [self._create_ippo_opponent_info(latest_path)]
        else:
            # Return all checkpoints
            return [self._create_ippo_opponent_info(path) for path in ippo_paths]
    
    def _create_ippo_opponent_info(self, checkpoint_path: str) -> OpponentInfo:
        """Create OpponentInfo for IPPO checkpoint."""
        parts = checkpoint_path.split('/')
        run_seed = parts[-3]
        seed_match = re.search(r'seed(\d+)', run_seed)
        seed = int(seed_match.group(1)) if seed_match else 0
        step = os.path.basename(checkpoint_path)
        name = f"IPPO_seed{seed}_step{step}"
        
        return OpponentInfo(
            name=name,
            algorithm="IPPO",
            checkpoint_path=os.path.abspath(checkpoint_path),
            seed=seed,
            step=step
        )
    
    def _discover_spppo_opponents(self, latest_only: bool = True) -> List[OpponentInfo]:
        """Discover SPPPO opponents."""
        # Try both experiments and checkpoints directories
        patterns = [
            f"experiments/checkpoints/spppo/run_*_seed{self.training_seed}/main/*",
            f"checkpoints/spppo/run_*_seed{self.training_seed}/main/*"
        ]
        
        spppo_paths = []
        for pattern in patterns:
            spppo_paths.extend(glob.glob(pattern))
        
        # Filter out temporary/incomplete checkpoints
        spppo_paths = [path for path in spppo_paths if not path.endswith('.orbax-checkpoint-tmp-0')]
            
        if not spppo_paths:
            return []
            
        if latest_only:
            # Find the most recent checkpoint
            latest_path = max(spppo_paths, key=os.path.getmtime)
            return [self._create_spppo_opponent_info(latest_path)]
        else:
            # Return all checkpoints
            return [self._create_spppo_opponent_info(path) for path in spppo_paths]
    
    def _create_spppo_opponent_info(self, checkpoint_path: str) -> OpponentInfo:
        """Create OpponentInfo for SPPPO checkpoint."""
        parts = checkpoint_path.split('/')
        run_seed = parts[-3]
        seed_match = re.search(r'seed(\d+)', run_seed)
        seed = int(seed_match.group(1)) if seed_match else 0
        step = os.path.basename(checkpoint_path)
        name = f"SPPPO_seed{seed}_step{step}"
        
        return OpponentInfo(
            name=name,
            algorithm="SPPPO",
            checkpoint_path=os.path.abspath(checkpoint_path),
            seed=seed,
            step=step
        )
    
    def _discover_fspppo_opponents(self, latest_only: bool = True) -> List[OpponentInfo]:
        """Discover FSPPPO opponents."""
        # Try both experiments and checkpoints directories
        patterns = [
            f"experiments/checkpoints/fspppo/run_*_seed{self.training_seed}/main/*",
            f"checkpoints/fspppo/run_*_seed{self.training_seed}/main/*"
        ]
        
        fspppo_paths = []
        for pattern in patterns:
            fspppo_paths.extend(glob.glob(pattern))
        
        # Filter out temporary/incomplete checkpoints
        fspppo_paths = [path for path in fspppo_paths if not path.endswith('.orbax-checkpoint-tmp-0')]
            
        if not fspppo_paths:
            return []
            
        if latest_only:
            # Prefer the highest numeric step directory rather than modification time
            numeric_dirs = []
            for path in fspppo_paths:
                if os.path.isdir(path):
                    base = os.path.basename(path)
                    if base.isdigit():
                        try:
                            step_num = int(base)
                            numeric_dirs.append((step_num, path))
                        except ValueError:
                            continue
            if numeric_dirs:
                # Pick the path with the largest numeric step
                step_num, latest_path = max(numeric_dirs, key=lambda t: t[0])
                return [self._create_fspppo_opponent_info(latest_path)]
            else:
                return []
        else:
            # Return all directory checkpoints
            return [self._create_fspppo_opponent_info(path) 
                   for path in fspppo_paths if os.path.isdir(path)]
    
    def _create_fspppo_opponent_info(self, checkpoint_path: str) -> OpponentInfo:
        """Create OpponentInfo for FSPPPO checkpoint."""
        parts = checkpoint_path.split('/')
        run_seed = parts[-3]
        seed_match = re.search(r'seed(\d+)', run_seed)
        seed = int(seed_match.group(1)) if seed_match else 0
        step = os.path.basename(checkpoint_path)
        name = f"FSPPPO_seed{seed}_step{step}"
        return OpponentInfo(
            name=name,
            algorithm="FSPPPO",
            checkpoint_path=os.path.abspath(checkpoint_path),
            seed=seed,
            step=step
        )
    
    def discover_opponents_by_algorithm(self, algorithm: str, latest_only: bool = True) -> List[OpponentInfo]:
        """Discover opponents for a specific algorithm.
        
        Args:
            algorithm: Algorithm name ("IPPO", "SPPPO", or "FSPPPO")
            latest_only: If True, only return the most recent checkpoint
            
        Returns:
            List of discovered opponents for the specified algorithm
        """
        algorithm = algorithm.upper()
        
        if algorithm == "IPPO":
            return self._discover_ippo_opponents(latest_only)
        elif algorithm == "SPPPO":
            return self._discover_spppo_opponents(latest_only)
        elif algorithm == "FSPPPO":
            return self._discover_fspppo_opponents(latest_only)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: IPPO, SPPPO, FSPPPO")
    
    def get_opponent_summary(self) -> Dict[str, int]:
        """Get a summary of available opponents by algorithm.
        
        Returns:
            Dictionary mapping algorithm names to opponent counts
        """
        summary = {}
        
        for algorithm in ["IPPO", "SPPPO", "FSPPPO"]:
            opponents = self.discover_opponents_by_algorithm(algorithm, latest_only=False)
            summary[algorithm] = len(opponents)
            
        return summary


def discover_learned_opponents(training_seed: int = 0, latest_only: bool = True) -> List[OpponentInfo]:
    """Convenience function to discover all learned opponents.
    
    Args:
        training_seed: Training seed to use for checkpoint discovery
        latest_only: If True, only return the most recent checkpoint for each algorithm
        
    Returns:
        List of discovered opponents
    """
    discovery = OpponentDiscovery(training_seed)
    return discovery.discover_all_opponents(latest_only)


if __name__ == "__main__":
    # Test the opponent discovery
    print("Testing opponent discovery...")
    
    # Test with different training seeds
    for seed in [0, 1, 2]:
        print(f"\n=== Training Seed {seed} ===")
        discovery = OpponentDiscovery(seed)
        opponents = discovery.discover_all_opponents(latest_only=True)
        
        if not opponents:
            print("No opponents found for this seed")
        
        # Show summary
        summary = discovery.get_opponent_summary()
        print(f"Summary: {summary}")

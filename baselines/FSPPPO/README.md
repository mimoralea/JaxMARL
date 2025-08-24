# FSPPPO: Fictitious Self-Play PPO

## Overview

This implementation provides **True Fictitious Self-Play PPO** for multi-agent reinforcement learning. The key principle is that each agent learns by playing against its own historical versions, not against other agents or external populations.

## Self-Play Architecture

### 🎯 **TRUE SELF-PLAY ONLY**

This codebase implements **pure Fictitious Self-Play** with the following strict constraints:

- ✅ **Same-seed sampling only**: Each seed samples opponents only from its own checkpoint history
- ✅ **Current-run sampling only**: Only checkpoints from the current training run are used
- ❌ **NO cross-seed sampling**: seed0 cannot sample from seed1's history
- ❌ **NO cross-run sampling**: Cannot sample from previous training runs
- ❌ **NO population-based training**: This is not a curriculum or population method

### Why This Design?

**Fictitious Self-Play** means an agent learns by playing against its own past policies. This is fundamentally different from:

- **Population-based training**: Agents learn against diverse populations
- **Cross-seed sampling**: Agents sample from other agents' histories
- **Multi-run curricula**: Using checkpoints from previous training sessions

Our implementation ensures each agent develops its own independent learning trajectory by facing only its own historical opponents.

## Usage

### Multi-Seed Training
```bash
# Train 4 independent agents, each with its own self-play history
python -m baselines.FSPPPO.train_fspppo NUM_SEEDS=4 TOTAL_TIMESTEPS=1000000
```

### Configuration Parameters
```yaml
# Opponent sampling frequency (every N training iterations)
OPPONENT_SAMPLING_FREQ: 200

# Mix ratio: probability of self-play vs historical opponent
SELF_PLAY_PROBABILITY: 0.5

# Recency bias: higher values favor more recent opponents
RECENCY_BIAS_ALPHA: 0.8

# Checkpoint saving frequency
CHECKPOINT_FREQ: 50
```

## Directory Structure

Each seed maintains its own independent checkpoint history (standardized):
```
checkpoints/fspppo/
├── run_20250727_223608_seed0/
│   └── main/
│       ├── 50/
│       ├── 100/
│       └── 150/
├── run_20250727_223608_seed1/
│   └── main/
│       ├── 50/
│       ├── 100/
│       └── 150/
└── ...
```

Notes:
- Step directories are numeric (e.g., `50`, `100`, `150`) and correspond to training update steps.
- Agent directory is `main/` for the primary agent.

Backward compatibility:
- Older runs may appear as `main_agent/step_000050/` (with `step_` prefix and zero-padding). Current code and tooling use the standardized `main/<step>/` layout.

**Key Point**: seed0 can ONLY sample from `run_20250727_223608_seed0/` directory. It cannot access seed1's checkpoints or any other run's checkpoints.

## Implementation Details

### Non-JIT Opponent Sampling
- All opponent sampling logic runs outside JAX JIT compilation
- Training runs in chunks with opponent resampling between chunks
- Avoids JAX tracing issues with Python control flow and I/O

### Sequential Multi-Seed Execution
- Seeds are trained sequentially (not in parallel) when opponent sampling is enabled
- Each seed gets full CPU/GPU resources during its training phase
- Ensures robust checkpoint I/O without JAX compilation conflicts

## Comparison with Other Methods

| Method | Cross-Seed | Cross-Run | Use Case |
|--------|------------|-----------|----------|
| **FSPPPO (this)** | ❌ No | ❌ No | True self-play |
| **Population Training** | ✅ Yes | ✅ Yes | Diverse opponents |
| **SPPPO** | N/A | N/A | Pure self-play (no history) |

## Future Extensions

While the current implementation focuses on true self-play, the architecture could be extended for:
- Population-based training (with cross-seed sampling)
- Multi-run curricula (with cross-run sampling)
- Hybrid approaches (mixing self-play with population methods)

However, these would be fundamentally different algorithms and should be implemented as separate methods to maintain clarity.

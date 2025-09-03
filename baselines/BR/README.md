# Best-Response (BR) Agent Training

This module implements Best-Response (BR) agent training for JaxMARL. BR agents are trained to exploit specific target opponents, revealing brittleness and vulnerabilities in learned and scripted policies.

## Overview

**Purpose**: BR agents are specialized exploiters designed to defeat specific target opponents. They are used for brittleness analysis, not as general-purpose tournament participants.

**Key Features**:
- Train against any scripted behavior (seek, dodge, guardian, noop, random)
- Train against any learned policy checkpoint (IPPO, SPPPO, FSPPPO)
- Uses shared PPO implementation for consistency
- Follows same code structure as other baselines
- Integrates with analysis pipeline for exploitability assessment

## Usage

### Basic Training

Train a BR agent against a scripted opponent:
```bash
cd /path/to/JaxMARL
conda activate jaxmarl
python -m baselines.BR.train OPPONENT_TYPE=scripted OPPONENT_NAME=seek
```

Train a BR agent against a learned policy:
```bash
python -m baselines.BR.train OPPONENT_TYPE=learned OPPONENT_PATH=/path/to/checkpoint
```

### Batch Training

Train BR agents against all tournament participants:
```bash
python -m baselines.BR.batch_train_br
```

## Configuration

Main configuration file: `config/br_ff_mpe.yaml`

Key parameters:
- `OPPONENT_TYPE`: "scripted" or "learned"
- `OPPONENT_NAME`: Name of scripted behavior (if scripted)
- `OPPONENT_PATH`: Path to checkpoint (if learned)
- `TOTAL_TIMESTEPS`: Training duration (default: 5M)
- `SEEDS`: List of training seeds

## Workflow Integration

BR training fits into the research workflow:

1. **batch_train**: Train baseline algorithms (IPPO, SPPPO, FSPPPO)
2. **train_br**: Train BR agents against all baselines + scripted behaviors
3. **run_tournament**: Evaluate baselines against each other (no BR agents)
4. **run_analysis**: Analyze tournament + BR exploitation results

## Output Structure

```
experiments/checkpoints/br/
├── run_YYYYMMDD_HHMMSS/
│   ├── seed_0/
│   │   └── main/
│   │       └── step_XXXXXX/
│   ├── seed_1/
│   └── seed_2/
```

## Research Applications

**Brittleness Assessment**: Measure how much each algorithm's performance drops when facing its specialized BR agent.

**Vulnerability Analysis**: Identify specific weaknesses in learned policies that BR agents exploit.

**Robustness Ranking**: Compare exploitability across different training methods.

**USD Motivation**: Demonstrate that even "robust" algorithms like FSPPPO can be exploited, motivating need for better training methods.

## Implementation Details

- **Architecture**: Uses shared `ActorCritic` network from `baselines.algorithms.ppo`
- **Training**: Similar to FSPPPO (main agent trainable, opponent fixed)
- **Opponent Loading**: Supports both scripted behaviors and learned checkpoints
- **Checkpointing**: Standard Orbax checkpoint management
- **Multi-seed**: Trains multiple seeds for statistical significance

## Notes

- BR agents are **NOT** included as tournament participants
- Each BR agent is specialized for one target opponent
- BR vs BR matchups are meaningless (not evaluated)
- Results integrated into analysis pipeline for exploitability metrics

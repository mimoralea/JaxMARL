#!/usr/bin/env python3
import os
import sys
import argparse
import orbax.checkpoint as ocp


def read_metadata_from_step_dir(step_dir: str):
    agent_dir = os.path.dirname(step_dir)
    step = int(os.path.basename(step_dir))
    cm = ocp.CheckpointManager(
        directory=os.path.abspath(agent_dir),
        checkpointers={'metadata': ocp.StandardCheckpointer()},
        options=ocp.CheckpointManagerOptions(create=False),
    )
    restored = cm.restore(step)
    md = restored.get('metadata', {})
    return md


def main():
    parser = argparse.ArgumentParser(description='Inspect BR checkpoint metadata')
    parser.add_argument('paths', nargs='+', help='Paths to BR main_agent step directories (e.g., .../main_agent/48828)')
    args = parser.parse_args()

    for p in args.paths:
        print(f"\n=== Inspecting: {p} ===")
        if not os.path.isdir(p):
            print('Missing directory')
            continue
        try:
            md = read_metadata_from_step_dir(p)
            if isinstance(md, dict):
                # Print specific fields if present
                for k in ['opponent_algo_code','opponent_seed','opponent_step','agent_id','is_final','training_step','step']:
                    if k in md:
                        print(f"{k}: {md[k]}")
                # Print all keys
                print('All keys:', sorted(md.keys()))
            else:
                print('Metadata is not a dict:', type(md))
        except Exception as e:
            print('Failed to read metadata:', e)


if __name__ == '__main__':
    main()

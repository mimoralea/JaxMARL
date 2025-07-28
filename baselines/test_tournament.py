#!/usr/bin/env python3
"""Simple test script to verify unified tournament system works with Orbax checkpoints."""

import sys
from pathlib import Path

# Add baselines to path
sys.path.append(str(Path(__file__).parent))

from tournament_eval import run_single_match

def test_ippo_vs_scripted():
    """Test IPPO checkpoint loading against scripted baseline."""
    
    # Use existing IPPO checkpoint
    ippo_checkpoint = "IPPO:checkpoints/ippo/run_20250718_233033_seed0/agent_0/4882.0/"
    
    print("Testing IPPO vs scripted baseline...")
    print(f"IPPO checkpoint: {ippo_checkpoint}")
    
    try:
        result = run_single_match(
            env_name="MPE_simple_sumo_v3",
            env_kwargs={"random_spawn": True},
            green_spec=ippo_checkpoint,
            red_spec="seek",
            match_seed=42,
            save_gif=False
        )
        
        print("\n✅ Test successful!")
        print(f"Green (IPPO): {result['green_reward']:.2f}")
        print(f"Red (seek): {result['red_reward']:.2f}")
        print(f"Winner: {result['winner'].upper()}")
        print(f"Episode length: {result['episode_length']} steps")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scripted_vs_scripted():
    """Test scripted baseline vs scripted baseline."""
    
    print("\nTesting scripted vs scripted baseline...")
    
    try:
        result = run_single_match(
            env_name="MPE_simple_sumo_v3",
            env_kwargs={"random_spawn": True},
            green_spec="seek",
            red_spec="centaur",
            match_seed=42,
            save_gif=False
        )
        
        print("\n✅ Test successful!")
        print(f"Green (seek): {result['green_reward']:.2f}")
        print(f"Red (centaur): {result['red_reward']:.2f}")
        print(f"Winner: {result['winner'].upper()}")
        print(f"Episode length: {result['episode_length']} steps")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Testing Unified Tournament System")
    print("=" * 50)
    
    # Test 1: Scripted vs Scripted (should always work)
    success1 = test_scripted_vs_scripted()
    
    # Test 2: IPPO vs Scripted (tests Orbax loading)
    success2 = test_ippo_vs_scripted()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Tournament system is working correctly.")
        print("\nNext steps:")
        print("1. Train SPPPO and FSPPPO with consistent checkpoints")
        print("2. Run full tournament evaluation")
        print("3. Generate analysis and visualizations")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

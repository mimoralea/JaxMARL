#!/usr/bin/env python3
"""
Test script to validate --training-seeds parsing logic without JAX dependencies.
"""

def parse_training_seeds(value):
    """Parse training seeds from CLI argument."""
    # Handle both string and integer inputs
    if isinstance(value, int):
        return [value]
    
    value_str = str(value)
    if ',' in value_str:
        # Multiple seeds: "0,1,2" or 0,1,2
        return [int(s.strip()) for s in value_str.split(',')]
    else:
        # Single seed: "0" or 0
        return [int(value_str)]

def test_training_seeds_parsing():
    """Test various training seeds parsing scenarios."""
    
    print("🧪 Testing --training-seeds parsing logic")
    print("=" * 50)
    
    test_cases = [
        ("0", [0]),
        ("1", [1]),
        ("0,1,2", [0, 1, 2]),
        ("0, 1, 2", [0, 1, 2]),  # with spaces
        ("5,7,9", [5, 7, 9]),
        ("42", [42]),
        ("0,1,2,3,4", [0, 1, 2, 3, 4]),
        # Test integer inputs (no quotes needed)
        (0, [0]),
        (1, [1]),
        (42, [42]),
    ]
    
    all_passed = True
    
    for input_value, expected in test_cases:
        try:
            result = parse_training_seeds(input_value)
            if result == expected:
                print(f"✅ '{input_value}' -> {result}")
            else:
                print(f"❌ '{input_value}' -> {result} (expected {expected})")
                all_passed = False
        except Exception as e:
            print(f"❌ '{input_value}' -> ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Training seeds parsing works correctly.")
    else:
        print("💥 Some tests failed!")
    
    print("\nExample usage:")
    print("  --training-seeds '0'       # Single seed")
    print("  --training-seeds '0,1,2'   # Multiple seeds (default)")
    print("  --training-seeds '5,7,9'   # Custom seeds")
    
    return all_passed

if __name__ == "__main__":
    test_training_seeds_parsing()

#!/usr/bin/env python3
import sys
sys.path.append('/root/autodl-tmp/self_train_verl')

from reward_fuction.caller import compute_score

test_data_source = "hiyouga/geometry3k"
test_solution_str = '{"question": "What is 2+2? A) 3 B) 4 C) 5 D) 6", "answer": "B"}'
test_ground_truth = 'B'

print("Testing compute_score function...")
print(f"Input data_source: {test_data_source}")
print(f"Input solution_str: {test_solution_str}")
print(f"Input ground_truth: {test_ground_truth}")

try:
    result = compute_score(test_data_source, test_solution_str, test_ground_truth)
    print(f"Output result: {result}")
    print(f"Result type: {type(result)}")
    if isinstance(result, list):
        print(f"List length: {len(result)}")
        for i, item in enumerate(result):
            print(f"  Item {i}: {item}, type: {type(item)}")
    else:
        print(f"ERROR: Expected list, got {type(result)}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
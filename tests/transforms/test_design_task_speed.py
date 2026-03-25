"""Speed tests for design tasks.

This module tests the execution speed of all design tasks using complex protein fixtures.
Each task is run 10 times and timing statistics (average and max) are collected and displayed.
"""

import atexit
import time
from collections import defaultdict

from tests.transforms.design_task_test_utils import (
    apply_design_task,
    parametrize_complex_sweep,
)

# Store timing results for all tasks
_timing_results = defaultdict(list)


def print_timing_results():
    """Print timing results table."""
    if not _timing_results:
        return

    # Calculate statistics
    results = []
    for task_name, timings in sorted(_timing_results.items()):
        avg_time = sum(timings) / len(timings)
        max_time = max(timings)
        min_time = min(timings)
        results.append(
            {
                "task": task_name,
                "avg": avg_time,
                "max": max_time,
                "min": min_time,
            }
        )

    # Print table
    print("\n" + "=" * 80)
    print("Design Task Speed Test Results")
    print("=" * 80)
    print(f"{'Task Name':<40} {'Avg (s)':>12} {'Max (s)':>12} {'Min (s)':>12}")
    print("-" * 80)

    for result in results:
        print(f"{result['task']:<40} " f"{result['avg']:>12.4f} " f"{result['max']:>12.4f} " f"{result['min']:>12.4f}")

    print("=" * 80)
    print(f"Total tasks tested: {len(results)}")
    print("Runs per task: 10")
    print("=" * 80 + "\n")


# Register the print function to run on exit
atexit.register(print_timing_results)


@parametrize_complex_sweep()
def test_design_task_speed(task_class, atom_array_fixture):
    """Test the execution speed of each design task.

    Runs each task 10 times and records timing information.
    Results are collected in _timing_results and printed at the end.
    """
    task_name = task_class.__name__
    timings = []

    # Run the task 10 times
    for _run_idx in range(10):
        start_time = time.perf_counter()

        # Apply the design task
        _data = apply_design_task(atom_array_fixture, task_class)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        timings.append(elapsed_time)

    # Store results
    _timing_results[task_name] = timings

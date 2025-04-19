import re
from collections import OrderedDict
import argparse

def parse_task_and_success_rates(directory, filename, unseen_task_ids=None):
    # Compile patterns for task lines and success-rate lines.
    task_pattern = re.compile(r'^Task:\s+(.*)$')
    task_rate_pattern = re.compile(
        r'^INFO:.*Current task success rate:\s+([\d.]+)$',
        re.IGNORECASE
    )
    
    # Use an OrderedDict to store tasks in the order they are found.
    results = OrderedDict()  # { task_string: [list_of_rates] }
    current_task = None
    log_path = f"{directory.rstrip('/')}/{filename}"
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if this line declares a new task.
            task_match = task_pattern.match(line)
            if task_match:
                current_task = task_match.group(1)
                if current_task not in results:
                    results[current_task] = []
                continue
            
            # Check if this line contains a success-rate.
            rate_match = task_rate_pattern.match(line)
            if rate_match and current_task is not None:
                rate_str = rate_match.group(1)
                results[current_task].append(float(rate_str))
    
    unseen_sum = 0.0
    unseen_count = 0
    seen_sum = 0.0
    seen_count = 0
    
    for index, (task, rates) in enumerate(results.items()):
        assert rates, f"No rates found for task: {task}"
        assert len(rates) == 1, f"Expected one rate per task, got {len(rates)} for task: {task}"
        avg_rate = rates[0]

        if unseen_task_ids is not None and index in unseen_task_ids:
            unseen_sum += avg_rate
            unseen_count += 1
        else:
            seen_sum += avg_rate
            seen_count += 1
    
    unseen_avg = unseen_sum / unseen_count if unseen_count > 0 else 0.0
    seen_avg = seen_sum / seen_count if seen_count > 0 else 0.0
    total_avg = (unseen_sum + seen_sum) / (unseen_count + seen_count) if (unseen_count + seen_count) > 0 else 0.0
    
    print(f"\nResults for: {filename}")
    if unseen_task_ids is not None:
        print(f"  Average success rate for unseen tasks: {unseen_avg:.4f}")
    print(f"  Average success rate for seen tasks: {seen_avg:.4f}")
    print(f"  Overall average success rate: {total_avg:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse per-task and overall success rates from log files in a directory"
    )
    parser.add_argument(
        '-d', '--directory', required=True,
        help='Path to the directory containing log files'
    )
    args = parser.parse_args()
    directory = args.directory

    # Define log files and their corresponding unseen task IDs
    configs = [
        ('spatial.log', [3, 8]),
        ('object.log', [5, 7]),
        ('goal.log', [1, 8]),
        ('10.log', [0, 6]),
    ]

    for filename, unseen_ids in configs:
        parse_task_and_success_rates(directory, filename, unseen_ids)

if __name__ == '__main__':
    main()

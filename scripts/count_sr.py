import re
from collections import OrderedDict

def parse_task_and_success_rates(filename, unseen_task_ids=None):
    # Compile patterns for task lines and success-rate lines.
    task_pattern = re.compile(r'^Task:\s+(.*)$')
    task_rate_pattern = re.compile(
        r'^INFO:.*Current task success rate:\s+([\d.]+)$',
        re.IGNORECASE
    )
    
    # Use an OrderedDict to store tasks in the order they are found.
    results = OrderedDict()  # { task_string: [list_of_rates] }
    current_task = None
    
    with open(filename, 'r') as f:
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
                # print(current_task)
                # print(line)
                rate_str = rate_match.group(1)  # e.g., "0.92"
                results[current_task].append(float(rate_str))
    
    # Print out per-task success rates.
    # print("Per-task success rates:")
    # for task, rates in results.items():
    #     if rates:
    #         print(f"Task: {task}, found success rates: {rates}")
    #     else:
    #         print(f"Task: {task}, no success rates found.")
    
    # Now compute the overall average success rates for unseen and seen tasks.
    unseen_sum = 0.0
    unseen_count = 0
    seen_sum = 0.0
    seen_count = 0
    
    # Iterate in the order tasks were added. Task IDs are assigned using enumerate (starting at 0).
    for index, (task, rates) in enumerate(results.items()):
        assert rates
        # import ipdb; ipdb.set_trace()
        assert len(rates) == 1
        avg_rate = sum(rates) / len(rates)

        
        # Use the order index as the task ID.
        if unseen_task_ids is not None and index in unseen_task_ids:
            # import ipdb; ipdb.set_trace()
            unseen_sum += avg_rate
            unseen_count += 1
        else:
            seen_sum += avg_rate
            seen_count += 1
    
    unseen_avg = unseen_sum / unseen_count if unseen_count > 0 else 0.0
    seen_avg = seen_sum / seen_count if seen_count > 0 else 0.0
    
    print("\nComputed Averages:")
    if unseen_task_ids is not None:
        print(f"Average success rate for unseen tasks: {unseen_avg:.4f}")
    print(f"Average success rate for seen tasks: {seen_avg:.4f}")

if __name__ == '__main__':
    # directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/"
    # directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_eval/"
    # directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split/"
    # directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv3_low_mem_finetune_sample2_actionssample32_random_select_without_delta_eval/"
    # directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split"
    # directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv4_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split_run2"
    directory = "/home/dingj0b/code/openpi/logs/pi0_libero_incontextv6_low_mem_finetune_sample2_actionssample32_random_select_without_delta_train_split"

    log_file = f"{directory}/spatial.log"
    # Provide the unseen task IDs as integers, corresponding to the task order (starting at 1)
    unseen_task_ids = [3, 8]  # For example: tasks 2, 4, and 6 are considered unseen.
    print("spatial task")
    parse_task_and_success_rates(log_file, unseen_task_ids)

    log_file = f"{directory}/object.log"
    # Provide the unseen task IDs as integers, corresponding to the task order (starting at 1)
    unseen_task_ids = [5, 7]  # For example: tasks 2, 4, and 6 are considered unseen.
    print("object task")
    parse_task_and_success_rates(log_file, unseen_task_ids)

    log_file = f"{directory}/goal.log"
    # Provide the unseen task IDs as integers, corresponding to the task order (starting at 1)
    unseen_task_ids = [1, 8]  # For example: tasks 2, 4, and 6 are considered unseen.
    print("goal task")
    parse_task_and_success_rates(log_file, unseen_task_ids)

    log_file = f"{directory}/10.log"
    # Provide the unseen task IDs as integers, corresponding to the task order (starting at 1)
    unseen_task_ids = [0, 6]  # For example: tasks 2, 4, and 6 are considered unseen.
    print("object task")
    parse_task_and_success_rates(log_file, unseen_task_ids)

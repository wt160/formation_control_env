import subprocess
import time
import os

# Script to run (must be executable and in PATH or provide full path)
TRAINING_SCRIPT = "ppo_train_bitmap_new.py" # Or "python ppo_train_bitmap.py" if not executable

# --- Define Experiment Configurations ---
# Each dictionary contains the arguments for one run of ppo_train_bitmap.py
experiment_configurations = [
    {
        "experiment_name": "ppo_bitmap_lr3e4_std0.2_rwg1_rwc-0.5",
        "train_env_type": "bitmap",
        "policy_filename": "ppo_bitmap_empty.pth", # Optional: if you have one
        "critic_filename": "ppo_bitmap_empty_critic.pth", 
        "output_policy_filename": "ppo_bitmap_clutter.pth",
        "output_critic_filename": "ppo_bitmap_clutter_critic.pth",
        "steps_per_epoch": 700,
        "epochs": 50000, # Shorter for example
        "device": "cpu", # Assign specific GPU if available
        "learning_rate": 3e-4,
        "action_std_init": 0.2,
        "reward_weight_goal": 1.0,
        "reward_weight_collision": -0.5,
        "num_envs": 20,
        "seed": 0,
        "has_laser": True,
        "train_map_directory": "train_maps_0_clutter",
        "use_leader_laser_only": False,
    },
    # {5_rwc-0.8_seed1",
    #     "train_env_type": "bitmap",
    #     # "policy_filename": "", # Start from scratch
    #     "output_policy_filename_suffix": "ppo_bitmap_final.pth",
    #     "steps_per_epoch": 500,
    #     "epochs": 50,
    #     "device": "cuda:1", # Assign another GPU if available, or use "cpu"
    #     "learning_rate": 1e-4,
    #     "action_std_init": 0.4,
    #     "reward_weight_goal": 1.5,
    #     "reward_weight_collision": -0.8,
    #     "num_envs": 8,
    #     "seed": 1,
    # },
    # {
    #     "experiment_name": "ppo_clutter_default_params_seed2",
    #     "train_env_type": "clutter", # Assuming you have a "clutter" scenario
    #     "output_policy_filename_suffix": "ppo_clutter_final.pth",
    #     "steps_per_epoch": 300,
    #     "epochs": 60,
    #     "device": "cpu",
    #     "learning_rate": 5e-4,
    #     "action_std_init": 0.5,
    #     "reward_weight_goal": 1.0,
    #     "reward_weight_collision": -1.0,
    #     "num_envs": 4,
    #     "seed": 2,
    # },
    # Add more configurations as needed
]

# --- Concurrency Settings ---
MAX_CONCURRENT_PROCESSES = 18
processes = []

for i, config in enumerate(experiment_configurations):
    while len(processes) >= MAX_CONCURRENT_PROCESSES:
        for p_idx, p in enumerate(processes):
            if p.poll() is not None:
                processes.pop(p_idx)
                break
        else:
            time.sleep(5)

    cmd = ["python", TRAINING_SCRIPT]
    for arg_name_key, arg_val in config.items():
        # Determine the command line flag format
        # If your ppo_train_bitmap_new.py defines flags like --train_env_type (with underscore)
        # then you should ensure those specific keys are not hyphenated.
        # Otherwise, for consistency, convert other multi-word keys from underscore to hyphen for CLI.

        # --- ADJUST THIS LOGIC BASED ON YOUR ppo_train_bitmap_new.py DEFINITIONS --- 
        if arg_name_key in ["critic_filename", "output_critic_filename", "train_map_directory", "train_env_type", "action_std_init", "reward_weight_goal", "reward_weight_collision", "policy_filename", "output_policy_filename", "experiment_name", "steps_per_epoch", "num_envs", "learning_rate", "has_laser", "use_leader_laser_only"]: # Add other keys that are defined with underscores in ppo_train_bitmap_new.py
            param_name_cli = f"--{arg_name_key}" # Use underscore directly
        elif arg_name_key in []: # Add keys defined with hyphens in ppo_train_bitmap_new.py
             param_name_cli = f"--{arg_name_key.replace('_', '-')}" # Convert to hyphen
        else: # Default to hyphenating for other keys, or make a clear choice
            param_name_cli = f"--{arg_name_key.replace('_', '-')}"


        # Handle boolean flags (action='store_true' or 'store_false')
        if isinstance(arg_val, bool):
            if arg_val:  # If True, add the flag
                cmd.append(param_name_cli)
            # If False for a 'store_true' action, we omit the flag,
            # and argparse in the target script will use its default (False).
        else:
            # Skip policy_filename if its value is empty or None (optional)
            if arg_name_key == "policy_filename" and not arg_val:
                continue
            cmd.append(param_name_cli)
            cmd.append(str(arg_val))

    print(f"({i+1}/{len(experiment_configurations)}) Launching experiment: {config.get('experiment_name', 'N/A')}")
    print(f"Command: {' '.join(cmd)}")

    log_dir = os.path.join("experiment_logs", config.get("experiment_name", f"exp_{i}"))
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = open(os.path.join(log_dir, "stdout.log"), "w")
    stderr_log = open(os.path.join(log_dir, "stderr.log"), "w")

    process = subprocess.Popen(cmd, stdout=stdout_log, stderr=stderr_log)
    processes.append(process)
    print(f"Launched PID: {process.pid}. Logs in: {log_dir}")
    time.sleep(2)

print("\nWaiting for all launched experiments to complete...")
for p_idx, p in enumerate(processes):
    p.wait()
    return_code = p.returncode
    # Correctly get the config name for the finished process
    # This assumes processes finish in the order they were launched, which might not be true
    # if MAX_CONCURRENT_PROCESSES < len(experiment_configurations).
    # A more robust way would be to store (process, config_name) tuples.
    # For now, this is a simplified approach:
    try:
        # Find the config associated with the process if needed, or use index if order is maintained
        # This part of logging which config finished might be tricky if they complete out of order.
        # The original code for getting config_name was okay assuming sequential completion or if order is preserved.
        # We'll keep the original logic for simplicity here.
        original_index = -1
        for k_idx, cfg in enumerate(experiment_configurations):
            # This is a placeholder for a better way to map process to config if they finish out of order
            # For now, we'll assume the print order of launch is sufficient for human tracking with PIDs
            pass

        # Simplified: refer to the experiment name from the config used to launch this process
        # This requires more careful tracking if you want to print the exact config name upon completion
        # when MAX_CONCURRENT_PROCESSES is involved. Let's assume the PID and log file are primary for tracking.
        # The provided error output doesn't give us the config name directly for failed process.

        if return_code == 0:
            print(f"Experiment with PID {p.pid} finished successfully.") # Simpler message
        else:
            print(f"Experiment with PID {p.pid} FAILED with return code {return_code}.")
            # Find the log directory associated with this PID is harder without storing map.
            # User should check the log directory based on the experiment name they know was launched with this PID.
    except IndexError:
        print(f"Process PID {p.pid} finished, but could not retrieve its original config name easily.")


print("\nAll experiments have concluded.")
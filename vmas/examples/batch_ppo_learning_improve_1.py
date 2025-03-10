# train_env_type = sys.argv[1]
# policy_filename = sys.argv[2]
# output_policy_filename = sys.argv[3]
# steps_per_epoch = int(sys.argv[4])
# # Set device
# device = sys.argv[5]

import subprocess




# List of argument sets to test
argument_sets = [
    # {"train_env_type": "clutter", "policy_filename": "best_imitation_model_clutter_11.pth", "output_policy_filename": "best_ppo_clutter_2.pth", "steps_per_epoch":"500", "device":"cuda"},
    {"train_env_type": "door", "policy_filename": "best_imitation_model_door_improve.pth", "output_policy_filename": "best_ppo_door_improve.pth", "steps_per_epoch":"300", "device":"cpu"},
    # {"train_env_type": "tunnel", "policy_filename": "best_imitation_model_tunnel_1.pth", "output_policy_filename": "best_ppo_tunnel_0.pth", "steps_per_epoch":"500", "device":"cpu"},
    # {"train_env_type": "narrow", "policy_filename": "best_imitation_model_narrow_improve.pth", "output_policy_filename": "best_ppo_narrow_improve.pth", "steps_per_epoch":"300", "device":"cpu"},
    
    # {"data_filename": "collected_data_narrow_7.pkl", "policy_filename": "best_imitation_model_narrow_noise_7.pth"},
    # {"data_filename": "collected_data_tunnel_0.pkl", "policy_filename": "best_imitation_model_tunnel_0.pth"},
    # {"data_filename": "collected_data_door_0.pkl", "policy_filename": "best_imitation_model_door_0.pth"},
    # {"data_filename": "collected_data_clutter_7.pkl", "policy_filename": "best_imitation_model_clutter_7.pth"},
]

# Loop through each set of arguments and call imitation_train.py
for args in argument_sets:
    env_type = args["train_env_type"]
    policy_filename = args["policy_filename"]
    output_policy_filename = args["output_policy_filename"]
    steps_per_epoch = args["steps_per_epoch"]
    device = args["device"]
    print(f"Running imitation_train.py with arguments: {env_type}, {policy_filename}, {output_policy_filename}, {steps_per_epoch}, {device}")

    # Call the script with subprocess
    subprocess.run(
        ["python", "ppo_train.py", env_type, policy_filename, output_policy_filename, steps_per_epoch, device],
        check=True  # Ensures the process raises an error if it fails
    )


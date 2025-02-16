import subprocess

# List of argument sets to test
argument_sets = [
    # {"data_filename": "collected_data_real_door_no_action.pkl", "policy_filename": "best_imitation_model_door_real_no_action.pth"},

    {"data_filename": "collected_data_door_narrow_0_no_action.pkl", "policy_filename": "best_imitation_model_door_narrow_noise_0_no_action.pth"},
    {"data_filename": "collected_data_tunnel_0_no_action.pkl", "policy_filename": "best_imitation_model_tunnel_0_no_action.pth"},
    # {"data_filename": "collected_data_door_0_no_action.pkl", "policy_filename": "best_imitation_model_door_0_no_action.pth"},
    {"data_filename": "collected_data_clutter_11_no_action.pkl", "policy_filename": "best_imitation_model_clutter_0_no_action.pth"},
]

# Loop through each set of arguments and call imitation_train.py
for args in argument_sets:
    data_filename = args["data_filename"]
    policy_filename = args["policy_filename"]

    print(f"Running imitation_train.py with arguments: {data_filename}, {policy_filename}")

    # Call the script with subprocess
    subprocess.run(
        ["python", "gnn_train_graph_obs_add_noise_cuda1_no_action.py", data_filename, policy_filename],
        check=True  # Ensures the process raises an error if it fails
    )
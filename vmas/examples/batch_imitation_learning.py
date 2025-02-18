import subprocess

# List of argument sets to test
argument_sets = [
    {"data_filename": "collected_data_tunnel_1.pkl", "policy_filename": "best_imitation_model_tunnel_1.pth"},
    {"data_filename": "collected_data_narrow_7.pkl", "policy_filename": "best_imitation_model_narrow_7.pth"},

    {"data_filename": "collected_data_door_narrow_0.pkl", "policy_filename": "best_imitation_model_door_narrow_0.pth"},
    # {"data_filename": "collected_data_tunnel_0.pkl", "policy_filename": "best_imitation_model_tunnel_0.pth"},
    {"data_filename": "collected_data_door_0.pkl", "policy_filename": "best_imitation_model_door_0.pth"},
    {"data_filename": "collected_data_clutter_11.pkl", "policy_filename": "best_imitation_model_clutter_11.pth"},
]

# Loop through each set of arguments and call imitation_train.py
for args in argument_sets:
    data_filename = args["data_filename"]
    policy_filename = args["policy_filename"]

    print(f"Running imitation_train.py with arguments: {data_filename}, {policy_filename}")

    # Call the script with subprocess
    subprocess.run(
        ["python", "gnn_train_graph_obs_add_noise_cuda1.py", data_filename, policy_filename],
        check=True  # Ensures the process raises an error if it fails
    )
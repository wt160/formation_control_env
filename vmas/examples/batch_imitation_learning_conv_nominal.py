import subprocess

# List of argument sets to test
argument_sets = [
    # {"data_filename": "collected_data_narrow_conv.pkl", "policy_filename": "best_imitation_model_narrow_conv.pth"},

    # {"data_filename": "collected_data_door_narrow_0.pkl", "policy_filename": "best_imitation_model_door_narrow_conv.pth"},
    # {"data_filename": "collected_data_tunnel_0.pkl", "policy_filename": "best_imitation_model_tunnel_0.pth"},
    {"data_filename": "collected_data_door_conv_nominal.pkl", "policy_filename": "best_imitation_model_door_conv_nominal.pth"},
    # {"data_filename": "collected_data_clutter_improve.pkl", "policy_filename": "best_imitation_model_clutter_conv.pth"},
    # {"data_filename": "collected_data_tunnel_conv.pkl", "policy_filename": "best_imitation_model_tunnel_conv.pth"},
]

# Loop through each set of arguments and call imitation_train.py
for args in argument_sets:
    data_filename = args["data_filename"]
    policy_filename = args["policy_filename"]

    print(f"Running imitation_train.py with arguments: {data_filename}, {policy_filename}")

    # Call the script with subprocess
    subprocess.run(
        ["python", "gnn_train_graph_obs_add_noise_cuda1_conv_nominal.py", data_filename, policy_filename],
        check=True  # Ensures the process raises an error if it fails
    )
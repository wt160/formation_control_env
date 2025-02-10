import os
import glob
import pickle
import sys

def merge_pickled_data(input_pattern='collected_data_*.pkl', output_file='collected_data_merged.pkl'):
    """
    Merges multiple pickle files containing collected data into a single pickle file.

    Args:
        input_pattern (str): Glob pattern to match input pickle files.
        output_file (str): Filename for the merged output pickle file.
    """
    # Find all files matching the input pattern
    pickle_files = glob.glob(input_pattern)
    
    if not pickle_files:
        print(f"No files found matching pattern '{input_pattern}'.")
        return

    print(f"Found {len(pickle_files)} files to merge.")

    # Open the output file in binary write mode using the highest protocol
    try:
        with open(output_file, 'wb') as fout:
            # Initialize the pickle writer with the highest protocol
            pickle.dump([], fout, protocol=pickle.HIGHEST_PROTOCOL)  # Initialize as empty list
            fout.seek(0)  # Reset file pointer to the beginning
            merged_data = []  # Temporary list to hold merged data

            for idx, file in enumerate(pickle_files, 1):
                try:
                    with open(file, 'rb') as fin:
                        data = pickle.load(fin)
                        if isinstance(data, list):
                            merged_data.extend(data)
                            print(f"File {idx}/{len(pickle_files)} '{file}' loaded with {len(data)} records.")
                            del data  # Free memory
                        else:
                            print(f"Warning: Data in '{file}' is not a list. Skipping this file.")
                except Exception as e:
                    print(f"Error loading '{file}': {e}")
            
            print(f"Total records after merging: {len(merged_data)}.")
            
            # Save the merged data using the highest protocol
            pickle.dump(merged_data, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Merged data saved to '{output_file}'.")
    except Exception as e:
        print(f"Error handling the output file '{output_file}': {e}", file=sys.stderr)

if __name__ == "__main__":
    # Change the input_pattern and output_file as needed
    merge_pickled_data(
        input_pattern='collected_data_*.pkl',  # Pattern to match input files
        output_file='collected_data_merged.pkl'  # Output file name
    )

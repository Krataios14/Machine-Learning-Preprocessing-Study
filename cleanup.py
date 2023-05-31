import os
import shutil

# Define directories to clean
directories = ['./data/processed_data', './results']

# Iterate over each directory
for directory in directories:
    # Check if directory exists
    if os.path.exists(directory):
        # Iterate over each file in directory
        for filename in os.listdir(directory):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            try:
                # Remove the file
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # If the file is a directory, recursively remove it
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        print(f'Cleaned directory: {directory}')
    else:
        print(f'The directory {directory} does not exist')

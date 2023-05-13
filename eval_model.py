import sys

if __name__ == "__main__":
    # Check if the command line argument is provided
    if len(sys.argv) < 2:
        print("Please provide a checkpoint location as a command line argument.")
    else:
        # Extract the checkpoint location from the command line argument
        checkpoint_location = sys.argv[1]

        # Use the checkpoint_location variable as needed
        print("Checkpoint location:", checkpoint_location)

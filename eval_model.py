import sys
from ray.rllib.algorithms.ppo import PPO
from RLWithBushMostellerEnv import test_n_games

NUM_TEST_GAMES = 1

if __name__ == "__main__":
    # Check if the command line argument is provided
    if len(sys.argv) < 2:
        print("Please provide a checkpoint location as a command line argument.")
    else:
        # Extract the checkpoint location from the command line argument
        path_to_checkpoint = sys.argv[1]

        # Use the checkpoint_location variable as needed
        my_new_ppo = PPO.from_checkpoint(path_to_checkpoint)

        results, final_states = test_n_games(my_new_ppo, NUM_TEST_GAMES)


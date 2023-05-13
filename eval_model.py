import sys
import os
from ray.rllib.algorithms.ppo import PPO
from RLWithBushMostellerEnv import run_n_rl_games
from BushMostellerEnvNoRL import run_n_baseline_games
from datetime import datetime
import pickle


NUM_TEST_GAMES = 10000

if __name__ == "__main__":
    # Check if the command line argument is provided
    if len(sys.argv) < 3:
        print("Please provide two command line arguments: path_to_checkpoint and reward_function")
    else:
        # Extract the checkpoint location from the command line argument
        path_to_checkpoint = sys.argv[1]
        reward_function = sys.argv[2]

        # Use the checkpoint_location variable as needed
        my_new_ppo = PPO.from_checkpoint(path_to_checkpoint)

        rl_game_rewards, rl_game_states = run_n_rl_games(my_new_ppo, reward_function, NUM_TEST_GAMES)
        baseline_game_states = run_n_baseline_games(NUM_TEST_GAMES)
        file_location = f'results/results_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.pkl'

        if not os.path.exists('./results'):
            os.makedirs('./results')

        with open(file_location, 'wb') as pickle_file:
            pickle.dump({
                'rl_game_rewards': rl_game_rewards,
                'rl_game_states': rl_game_states,
                'baseline_game_states': baseline_game_states
            }, pickle_file)

        print(f'Results saved in: {file_location}')




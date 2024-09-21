import sys
import os
from ray.rllib.algorithms.ppo import PPO
from RLWithBushMostellerEnv import run_n_rl_games, TMAX_OPTIONS
from BushMostellerEnvNoRL import run_n_baseline_games
from datetime import datetime
import pickle
import numpy as np

# prop results: results/results_20230513121319691750.pkl
# sum results: results/results_sum_20230513200538466486.pkl

NUM_TEST_GAMES = 1000

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

        rl_game_rewards_by_tmax = {}
        rl_game_states_by_tmax = {}
        baseline_game_states_by_tmax = {}

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_location = f'results/results_dqn_{reward_function}_{timestamp}'

        if not os.path.exists('./results'):
            os.makedirs('./results')
    

        for tmax in TMAX_OPTIONS:
            rl_game_rewards, rl_game_states = run_n_rl_games(my_new_ppo, reward_function, NUM_TEST_GAMES, tmax)
            baseline_game_states = run_n_baseline_games(NUM_TEST_GAMES, tmax)

            rl_game_rewards_by_tmax[tmax] = rl_game_rewards
            rl_game_states_by_tmax[tmax] = rl_game_states
            baseline_game_states_by_tmax[tmax] = baseline_game_states

        with open(f'{file_location}.pkl', 'wb') as pickle_file:
            pickle.dump({
                'rl_game_rewards': rl_game_rewards_by_tmax,
                'rl_game_states': rl_game_states_by_tmax,
                'baseline_game_states': baseline_game_states_by_tmax
            }, pickle_file)
        
        with open(f'{file_location}.txt', 'w') as results_txt:
            result_strings = []

            for tmax, rewards in rl_game_rewards_by_tmax.items():
                # Calculate the mean of the array
                mean_reward = np.mean(rewards)
                # Create the string for each tmax value
                result_string = f"tmax: {tmax}, mean: {mean_reward:.2f}"
                # Append the result string to the list
                result_strings.append(result_string)

                # Join all strings into a single string if needed
                final_output = "\n".join(result_strings)
                results_txt.write(final_output)
                print(f'Results for tmax = {tmax}: {mean_reward}')


        print(f'Results saved in: {file_location}')




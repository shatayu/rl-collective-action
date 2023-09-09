# Create a PPO algorithm object using a config object ..
from ray.rllib.algorithms.ppo import PPOConfig
from RLWithBushMostellerEnv import RLWithBushMostellerWholeGameEnv
import os
import sys
from datetime import datetime

NUM_TRAINING_ITERATIONS = 1000

# prop game: /home/shatayu/ray_results/PPO_RLWithBushMostellerWholeGameEnv_2023-05-12_23-16-30p84cj234/checkpoint_001000
# sum model:  /home/shatayu/ray_results/PPO_RLWithBushMostellerWholeGameEnv_2023-05-13_11-04-37_jmf78e3/checkpoint_001000


if __name__ == "__main__":
    # Check if the command line argument is provided
    if len(sys.argv) < 2 or sys.argv[1] not in ('prop', 'sum'):
        print("Please provide the reward function ('prop' or 'sum')")
    else:
        # Extract the checkpoint location from the command line argument
        reward_function = sys.argv[1]

        my_ppo = ( 
            PPOConfig()
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            .environment(
                env=RLWithBushMostellerWholeGameEnv,
                env_config={
                    'num_rounds_hidden': 0,
                    'reward_function': reward_function
                },
                auto_wrap_old_gym_envs=False
            )
            .build()
        )


        for i in range(NUM_TRAINING_ITERATIONS):
            my_ppo.train()

            if (i + 1) % 10 == 0:
                path_to_checkpoint = my_ppo.save()
                print("*********************************")
                print(
                    "An Algorithm checkpoint has been created inside directory: "
                    f"'{path_to_checkpoint}'."
                )
                print("---------------------------------")

        # Let's terminate the algo for demonstration purposes.
        final_checkpoint = my_ppo.save('./models/')
        renamed_checkpoint = f'models/model_{reward_function}_{final_checkpoint.split("/")[-1]}_{datetime.now().strftime("%Y%m%d%H%M%S%f")}'

        os.rename(final_checkpoint, renamed_checkpoint)

        my_ppo.stop()
        print(f'Final checkpoint (reward function: {reward_function}): {renamed_checkpoint}')
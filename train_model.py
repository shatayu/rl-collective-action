# Create a PPO algorithm object using a config object ..
from ray.rllib.algorithms.ppo import PPOConfig
from RLWithBushMostellerEnv import RLWithBushMostellerWholeGameEnv

NUM_TRAINING_ITERATIONS = 1000
REWARD_FUNCTION = 'sum'

# prop game: /home/shatayu/ray_results/PPO_RLWithBushMostellerWholeGameEnv_2023-05-12_23-16-30p84cj234/checkpoint_001000

my_ppo = ( 
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(
        env=RLWithBushMostellerWholeGameEnv,
        env_config={
            'num_rounds_hidden': 0,
            'reward_function': REWARD_FUNCTION
        },
        auto_wrap_old_gym_envs=False
    )
    .build()
)


final_checkpoint = ""
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
        final_checkpoint = path_to_checkpoint

# Let's terminate the algo for demonstration purposes.
my_ppo.stop()
print(f'Final checkpoint (reward function: {REWARD_FUNCTION}): {final_checkpoint}')

# Test the environment out
# env = RLWithBushMostellerWholeGameEnv({
#     'num_rounds_hidden': 0,
#     'reward_function': 'proportion'
# })

# obs = env.reset()
# reward = 0
# terminated = False
# truncated = False

# action = my_ppo.compute_single_action(obs)
# print(action)

# obs, reward, terminated, truncated, info = env.step(action)

# # Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# # thathas the exact same state as the old one, from which the checkpoint was
# # created in the first place:
# from ray.rllib.algorithms.ppo import PPO

# my_new_ppo = PPO.from_checkpoint(path_to_checkpoint)

# print("Finished!")

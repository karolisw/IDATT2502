import os, os.path
import gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# The path must be the path to the sb3_logs directory
def eval_all_trained_models(path, env, nr_episodes):
    runs = os.scandir(path)
    for run in runs:
        path_to_model = str(path) + str(run) + "/models"
        #We are now inside the folder containing all the tructuated runs
        if(os.path.isdir(path_to_model)): # the run has a models directory
            eval_trained_model(path, env, nr_episodes)
        else:
            print("Could not find model in path: {}".format(path_to_model))

# The path must be the path to the sb3_logs director
def test_all_trained_models(path, env, nr_episodes):
    runs = os.scandir(path)
    for run in runs:
        path_to_model = str(path) + str(run) + "/models"
        #We are now inside the folder containing all the tructuated runs
        if(os.path.isdir(path_to_model)): # the run has a models directory
            test_trained_agent(path_to_model, env, nr_episodes)
        else:
            print("Could not find model in path: {}".format(path_to_model))

# Just to see how well the model performs with current parameters
def eval_trained_model(path_to_model, env, nr_episodes):
    if (os.path.exists(path_to_model) and os.path.isfile(path_to_model)):
        gym_env = Monitor(gym.make(env))#, render_mode="human")
        model = PPO.load(path=path_to_model, env=gym_env)
        mean_reward, std_reward = evaluate_policy(model, gym_env, n_eval_episodes=nr_episodes)
        print("Mean reward for model with path '{}' is:{}\nStd reward for model is: {}\n".format(path_to_model,mean_reward, std_reward))
    else:
        raise Exception("Could not find model with path: ", path_to_model)


# For use in final comparison between this trained model and other programs model
def test_trained_agent(path_to_model, env, nr_steps):

    if (os.path.exists(path_to_model) and os.path.isfile(path_to_model)):
            gym_env = gym.make(env )#, render_mode="rgb_array")
            # Record video every 10th episode
            #gym_env = RecordVideo(gym_env, 'video', episode_trigger = lambda x: x % 10 == 0) #TODO this should be on (update sb3 first)
            model = PPO.load(path=path_to_model, env=gym_env)

            all_rewards = []

            obs = gym_env.reset()
            done_testing = False
            
            for step in range(nr_steps):

                episode_rewards = []
                while not (done_testing):
                    action, _states = model.predict(obs)
                    #action, _states = model.predict(obs, deterministic=True)

                    obs, rewards, done, info = gym_env.step(action)
                    if (done):
                        done_testing = True
                    episode_rewards.append(rewards)   # Add reward to the list
                    #gym_env.render()
                    print("nr rewards: ",len(all_rewards))

                # Add episode rewards to all rewards
                all_rewards.append(episode_rewards)
                print("The mean reward in step {} is {}".format(step, np.mean(np.array(episode_rewards))))

            # When training is done, we calculate the mean of all rewards
            all_rewards_np = np.array(all_rewards)

            # Returns a dictionary that holds the mean and max reward
            return {'Mean_reward':np.mean(all_rewards_np),
                    'Max reward':np.max(all_rewards_np) }

    # Do not train if no model :p
    else:
        raise Exception("Could not find model with path: ", path_to_model)    

# Get two models
# Evaluate the model
eval_trained_model("logs/best_models/1.zip", "procgen:procgen-bigfish-v0", 10)
eval_trained_model("logs/best_models/best_model.zip", "procgen:procgen-bigfish-v0", 10)

# Test the agent
mean_max_reward1 = test_trained_agent("logs/best_models/1.zip","procgen:procgen-bigfish-v0", 10)
mean_max_reward2 = test_trained_agent("logs/best_models/best_model.zip","procgen:procgen-bigfish-v0", 10)

print('Mean reward for model 1: ', mean_max_reward1['Mean_reward'])
print('Mean reward for model 2: ', mean_max_reward2['Mean_reward'])
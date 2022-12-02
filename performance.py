import os, os.path
import gym
import time
import numpy as np
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# For video-recording
from stable_baselines3.common.vec_env import VecVideoRecorder

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


eval_dir ="logs/eval_results/" + '_' + time.strftime("%d-%m-%Y_%H-%M-%S") 


# The path must be the path to the sb3_logs directory
def eval_all_trained_models(path, env, nr_episodes):
    runs = os.listdir(path)
    for run in runs: 
        # We are now at the folders containing the runs and proceed to find the models
        path_to_models = str(path) +"/"+ str(run) + "/models"
        models = os.listdir(path_to_models)
        for model in models:
            model_path = path_to_models + "/" + model
            if(os.path.isfile(model_path)): # should be zip file containing one of many models for the run
                print("Evaluating model at path: {}".format(model_path))
                eval_trained_model(model_path, env, nr_episodes)
            else:
                print("Could not find model in path: {}".format(model_path))



# Just to see how well the model performs with current parameters
def eval_trained_model(path_to_model, env, nr_episodes):
    if (os.path.exists(path_to_model) and os.path.isfile(path_to_model)):
        try:
            # If value-error, we use procgen-env instead
            gym_env = ProcgenEnv(num_envs=32,
                start_level=0, 
                num_levels=0,
                distribution_mode="easy", 
                env_name="bigfish",
                render_mode="rgb_array")
            gym_env = VecVideoRecorder(venv=gym_env, video_folder="video", record_video_trigger=lambda x: x % 2000 == 0, video_length=200) # type: ignore
            gym_env = VecMonitor(gym_env, filename=eval_dir) # type: ignore
            model = PPO.load(path=path_to_model, env=gym_env, custom_objects={"clip_range" : 0.2})
        except ValueError:
            gym_env = Monitor(gym.make(env, render_mode="human"))
            model = PPO.load(path=path_to_model, env=gym_env, custom_objects={"clip_range" : 0.2})

        # In both cases
        mean_reward, std_reward = evaluate_policy(model, gym_env, n_eval_episodes=nr_episodes)
        print("Mean reward for model with path '{}' is:{}\nStd reward for model is: {}\n".format(path_to_model,mean_reward, std_reward))
        
        # Also write the result to file
        with open("logs/eval_results/general_performance.txt", "a") as f:
            f.write("Mean reward for model with path '{}' is:{}\nStd reward for model is: {}\n".format(path_to_model,mean_reward, std_reward))
        return mean_reward

    else:
        raise Exception("Could not find model with path: ", path_to_model)

# Rewards is the list of all rewards 
def create_reward_fig(rewards, path):
        
        # For x axis
        count_rewards = len(rewards)
        print("reward count: ", count_rewards)

        # from 1 to count, with count count mod 100 points
        x = np.linspace(0, count_rewards, count_rewards)

        #if(count_rewards > 1):
       #     x = np.linspace(0, count_rewards, count_rewards % 100)

        y = rewards

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")

        # save image to path
        time_now = time.strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig(path + '@_time_' + time_now, format="png")
        #plt.show()
        #plt.close()
         
        with SummaryWriter(path) as reward_writer:
            reward_writer.add_figure("reward/episode", figure = plt.gcf())    

        return plt.gcf() # return figure

# For videoing
eval_trained_model("logs/sb3_logs/best_models_PBT/199.zip","procgen:procgen-bigfish-v0", 3)
eval_trained_model("logs/sb3_logs/best_models_regular/baseline3_3.zip","procgen:procgen-bigfish-v0", 3)

'''
# Testing the two best performing models from each project

pbt_mod_rewards= []
reg_mod_rewards= []

for i in range(20):
    pbt_mod_rewards.append(eval_trained_model("logs/sb3_logs/best_models_PBT/199.zip","procgen:procgen-bigfish-v0", 10))

# Now we plot the testing
create_reward_fig(pbt_mod_rewards,"logs/sb3_logs/best_models_PBT/plot")

for i in range(20):
    reg_mod_rewards.append(eval_trained_model("logs/sb3_logs/best_models_regular/baseline3_3.zip","procgen:procgen-bigfish-v0", 10))

# Plotting once more
create_reward_fig(reg_mod_rewards,"logs/sb3_logs/best_models_regular/plot")

pbt_mean = np.mean(np.array(pbt_mod_rewards))
reg_mean = np.mean(np.array(reg_mod_rewards))
print("PBT models average reward: {}".format(pbt_mean))
print("Regular models average reward: {}".format(reg_mean))


# Baseline3_3.zip performs best
'''
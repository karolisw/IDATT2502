import gym
import procgen
from procgen import ProcgenEnv
from gym import spaces, utils
import os
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import pybullet_envs

    #VecExtractDictObs,
    #VecMonitor,
    #VecFrameStack,
    #VecNormalize

from gym_minigrid.wrappers import * # Do not remove - necessary to use OneHotPartialWrapper and FlatObsWrapper (both for MiniGrid below)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecExtractDictObs,VecMonitor, VecVideoRecorder


def env_create(env_id="CartPole-v1", idx=0, seed=141, vec_env=False, capture_video=False, run_name="Test"):   
    if env_id[0:8] == "MiniGrid":
        print("=="*10+"MiniGrid"+"=="*10)
        env = gym.make(env_id)
        env.seed(seed)
        env = Monitor(env)
        env = OneHotPartialObsWrapper(env)
        #env = RGBImgPartialObsWrapper(env)
        env = FlatObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        return env
    if env_id[0:6] == "dm2gym":
        print("=="*10+"DM_Control"+"=="*10)
        env = gym.make(env_id, environment_kwargs={'flat_observation': True})
        env.seed(seed)
        env = DummyVecEnv([lambda:env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    if env_id[-12:] == "BulletEnv-v0":
        #env = DummyVecEnv([lambda: gym.make(env_id)])
        print("=="*10+"Bullet"+"=="*10)
        env = make_vec_env(env_id, n_envs=1, seed=seed)
        # Automatically normalize the input features and reward
        env = VecNormalize(venv=env, norm_obs=True, norm_reward=True, clip_obs=10.)
    if env_id[0:7] == "BigFish" or env_id[0:7] == "bigfish":
        print("=="*10+"BigFish"+"=="*10)
        #env = gym.make("procgen:procgen-bigfish-v0", render_mode="human")
        env = gym.make("procgen:procgen-bigfish-v0", num_levels=1, start_level=0,#render_mode="human", 
                        center_agent=False,distribution_mode="easy")
        #env = procgen.ProcgenEnv(env_name='bigfish', num_envs=1)
        #env = ProcgenEnv(num_envs=32, env_name='bigfish', distribution_mode='easy')
        #env = VecExtractDictObs(env, "rgb") # To use only part of the observation
        # Action space should be normalized
        #action_space = spaces.Box(low=1, high=1, shape=(4), dtype="float32")

        # Wrap with a VecMonitor to collect stats and avoid errors
        #env = VecMonitor(venv=env)

        #env = VecVideoRecorder(venv=env, video_folder="logs/videos", record_video_trigger=lambda x: x == 0, video_length=100, name_prefix="random-agent-{}".format(env_id))

        #env = VecNormalize(venv=env, ob=False)
        #env = gym.make('procgen-bigfish-v0')
        #env.seed(seed)
        #env = Monitor(env, filename="monitor/monitor.csv")
        #env = gym.wrappers.RecordEpisodeStatistics(env) # COuld be removed -> not tested yet
        #env = DummyVecEnv([lambda:env]) # Slowed down the performance greatly
        #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)  # Also slowed down the reward
    else:
        env = gym.make(env_id)
        env.seed(seed)
        
    #env = gym.wrappers.RecordEpisodeStatistics(env)
    
    if capture_video:
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)
    return env

 

def env_create_sb(env_id="CartPole-v1", idx=0, seed=141, n_envs=1, capture_video=False, log_dir="./tmp"):
    
    def thunk():
        env = make_vec_env(env_id, n_envs=n_envs, monitor_dir=log_dir)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def eval_agent(model, env):
    obs = env.reset()
    r = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        r.append(rewards)
        if dones:
            #r = info['episode']['r']
            break
    return np.sum(r)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, idx: int = 0):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_dir = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.idx = idx

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  _model_name = "ID_" + str(self.idx) + "_Best_Model"    
                  save_path = os.path.join(self.save_dir, _model_name)
                  self.model.save(save_path)

        return True
import gym
from typing import Optional


from stable_baselines3 import SAC
from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: (Optional[BaseCallback]) Callback that will be called
        when an event is triggered.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(EventCallback, self).__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if callback is not None:
            self.callback.parent = self

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback()
        return True


################################ CHECKPOINT CALLBACK ##########################


# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = SAC("MlpPolicy", "Pendulum-v1")
model.learn(2000, callback=checkpoint_callback)


################################ EVAL CALLBACK ################################

# Separate evaluation env
eval_env = gym.make("Pendulum-v1")
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)

model = SAC("MlpPolicy", "Pendulum-v1")
model.learn(5000, callback=eval_callback)


################################ PROGRESSBAR CALLBACK ##########################

model = PPO("MlpPolicy", "Pendulum-v1")
# Display progress bar using the progress bar callback
# this is equivalent to model.learn(100_000, callback=ProgressBarCallback())
model.learn(100_000, progress_bar=True)



################## STOP TRAINING ON NO IMPROVEMENT CALLBACK #####################

# Separate evaluation env
eval_env = gym.make("Pendulum-v1")
# Stop training if there is no improvement after more than 3 evaluations
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

model = SAC("MlpPolicy", "Pendulum-v1", learning_rate=1e-3, verbose=1)
# Almost infinite number of timesteps, but the training will stop early
# as soon as the the number of consecutive evaluations without model
# improvement is greater than 3
model.learn(int(1e10), callback=eval_callback)
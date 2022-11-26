import argparse
import os
import time
import numpy as np
from utils.mpi_utils import MPI_Tool
from stable_baselines3.common.evaluation import evaluate_policy
from utils.rl_tools import env_create_sb, env_create, eval_agent, SaveOnBestTrainingRewardCallback
# from pbt_toy import pbt_engine
from mpi4py import MPI
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
mpi_tool = MPI_Tool()
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
#from tensorboardX import SummaryWriter
tmp_path = "logs/sb3_logs/truct" + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
models_dir = tmp_path +"/models"

if not (os.path.exists(tmp_path)):
    os.makedirs(tmp_path)

if not (os.path.exists(models_dir)):
    os.makedirs(models_dir)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--tb-writer", type=bool, default=False,
        help="if toggled, Tensorboard summary writer is enabled")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HumanoidBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=141,
        help="seed of the experiment")
    parser.add_argument("--num-agents", type=int, default=20,
        help="number of agents")
    parser.add_argument("--total-generations", type=int, default=200,
        help="total generations of the experiments")
    parser.add_argument("--agent-training-steps", type=int, default=10000,
        help="total generations of the experiments")
    
    parser.add_argument("--learning-rate-range", type=tuple, default=(1e-4, 1e-3),
        help="the range of leanring rates among different agents")
    parser.add_argument("--gamma-range", type=tuple, default=(0.8, 0.99),
        help="the range of discount factors among different agents")
    args = parser.parse_args()

    return args



'''
##############
## CALLBACK ##
##############

print('CREATING CALLBACK...')   

# Saving model to same directory as Monitor directory every 'check_freq' steps
best_reward_callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir = tmp_path)#=models_dir)  

'''
class rl_agent():
    def __init__(self, idx, env_name, learning_rate, gamma, log_dir = tmp_path, seed=141) -> None:
        self.idx = idx
        self.seed = seed
        self.score = 0 # For now just use reward per episode 
        self.length = 0 # For now just use length per episode 

        if env_name[0:8] == "MiniGrid":
            self.env = env_create(env_name, idx)
            #self.model = DQN("MlpPolicy", env = self.env, verbose=0, create_eval_env= False)
            self.model =  PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
        elif env_name[0:7] == "BigFish" or env_name[0:7] == "bigfish":          
            self.env = env_create(env_name, idx) 
            self.model = PPO("CnnPolicy", env=self.env, verbose=0,n_steps=256, n_epochs=3, batch_size=2048,
                        gae_lambda= 0.95, clip_range= 0.2, vf_coef= 0.5, ent_coef= 0.01, max_grad_norm=0.5, normalize_advantage=True) 
        elif env_name[0:11] == "LunarLander":
            self.env = env_create(env_name, idx) 
            self.model =  PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
        elif env_name[0:5] == "nasim": 
            self.env = env_create(env_name, idx)
            #self.model = DQN("MlpPolicy", env = self.env, verbose=0, create_eval_env= False)
            self.model =  PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=False)
        elif env_name[0:6] == "dm2gym":
            self.env = env_create(env_name, idx)
            #self.model = DQN("MlpPolicy", env = self.env, verbose=0, create_eval_env=True)
            self.model = PPO("MultiInputPolicy", env=self.env, verbose=0, create_eval_env=True)
        else:
            #self.model = DQN("MlpPolicy", env = env_name, verbose=0, create_eval_env=True)
            self.model =  PPO("MlpPolicy", env=env_name, verbose=0, create_eval_env=True)
        self.model.gamma = gamma
        self.model.learning_rate = learning_rate 
        self.log_dir = os.path.join(log_dir, str(idx))
        new_logger = configure(tmp_path, ["tensorboard"]) #removed csv!
        self.model.set_logger(new_logger)

    def step(self, traing_step=2000, callback=None, vanilla=False, rmsprop=False, Adam=False):
        """one episode of RL"""
        self.model.learn(total_timesteps=traing_step)

    def exploit(self, best_params):

        self.model.set_parameters(best_params) 
        

    def explore(self):
        """
        perturb hyperparaters with noise from a normal distribution
        """
        
        # LR 0.95 decay
        self.model.learning_rate=self.model.learning_rate*np.random.triangular(0.9, 0.95, 1.2)

        if self.model.gamma*np.random.uniform(0.9, 1.1)>=0.99:
            self.model.gamma = 0.99
        elif self.model.gamma*np.random.uniform(0.9, 1.1)<=0.8:
            self.model.gamma = 0.8
        else:
            self.model.gamma = self.model.gamma*np.random.uniform(0.9, 1.1)


    def eval(self, vanilla=True, return_episode_rewards=False):

        # Evaluate the agent

        if vanilla:
            if return_episode_rewards == True:
                eps_reward, eps_length = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=5, return_episode_rewards=True)
                mean_reward = np.mean(eps_reward)
                mean_length = np.mean(eps_length)
                self.length = mean_length
            else:
                mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=5)
        else:
            mean_reward = eval_agent(self.model, self.model.get_env())

        self.score =  mean_reward
        

    def update(self):
        """
        Just update the
        """

def workers_init(args):
    workers = []
    for idx in range(args.num_agents):
        # get learning rate, uniformly sampled on log scale
        _l_lb = np.log10(args.learning_rate_range[0])
        _l_ub = np.log10(args.learning_rate_range[1])
        if _l_ub >= _l_lb:       
            _lr = 10 ** np.random.uniform(low=_l_lb, high=_l_ub)
        else:
            raise Exception('Error in Learning Rate Range: Low bound shoud less that the Upper bound')
        
        # get discount factor, uniformly sampled 
        _g_lb = np.log10(args.gamma_range[0])
        _g_ub = np.log10(args.gamma_range[1])
        if _g_ub >= _g_lb:       
            _g = np.random.uniform(low=_g_lb, high=_g_ub)
        else:
            raise Exception('Error in Gamma Range: Low bound shoud less that the Upper bound')
        
        workers.append(rl_agent(idx=idx, env_name=args.env_id, learning_rate=_lr, gamma=_g)) 
    return workers

class base_population(object):
    def __init__(self):
        self.agents_pool = []

    def create(self, agent_list):
        self.agents_pool = agent_list

    def get_scores(self):
        return [worker.score for worker in self.agents_pool]
        # return score

     # Hoping to be able to call on this when we need to save the best model of the population
    def get_best_model(self):
        _best_id = self.get_best_agent()
        return self.agents_pool[_best_id] 

    def get_best_agent(self):
        return self.get_scores().index(max(self.get_scores()))

    def get_best_score(self):
        # return max(self.get_scores())
        _best_id = self.get_best_agent()
        return self.agents_pool[_best_id].score
    

    def get_best_results(self):
        # return max(self.get_scores())
        _best_id = self.get_best_agent()
        return [self.agents_pool[_best_id].score, self.agents_pool[_best_id].length] 

    def get_best_agent_params(self):
        _best_id = self.get_best_agent()
        _best_agent = self.agents_pool[_best_id]
        params = _best_agent.model.get_parameters()

        return params

    @property
    def size(self):
        return int(len(self.agents_pool))


class base_engine(object):
    def __init__(self, total_population_size, tb_logger=False):
        self.total_population_size = total_population_size
        self.best_score_population = 0
        if mpi_tool.is_master & (tb_logger):
            self.tb_writer = SummaryWriter(log_dir=tmp_path)
        else:
            self.tb_writer = False

    def create_local(self, pbt_population):
        if pbt_population.size == 0:
            self.population = []
            self.best_params_population = []
        else:    
            self.population = pbt_population
            self.best_params_population = self.population.get_best_agent_params()
        

    def run(self, steps=3, exploit=False, explore=False, agent_training_steps=2000, return_episode_rewards=True):
        if not mpi_tool.is_master:
            print("Agents number: {} at rank {} on node {}".format(self.population.size, mpi_tool.rank, str(mpi_tool.node)))
        
        for i in range(steps):

            if mpi_tool.is_master:
                # Master is the centre controll, with no RL agent
                top=round(self.total_population_size*0.50)
                bottom=round(self.total_population_size*0.50)
                exchanged_vector = np.arange(self.total_population_size)
                #print(exchanged_vector)
            else:
                exchanged_vector = np.arange(self.total_population_size) 
                #print(exchanged_vector)

         
            for worker in self.population.agents_pool:
                worker.step(traing_step=agent_training_steps, vanilla=True)  # one step of GD
                worker.eval(return_episode_rewards=return_episode_rewards)
                
            # Update best score to the whole population
            if return_episode_rewards:
                best_results_to_sent = self.population.get_best_results()
            else:
                best_score_to_sent = self.population.get_best_score()
                
            best_params_to_sent = self.population.get_best_agent_params()
        
            if return_episode_rewards:
                #print(best_results_to_sent)
                best_score_to_sent, best_length_to_sent = best_results_to_sent[0], best_results_to_sent[1]
                best_scores = mpi_tool.gather(best_score_to_sent, root=0)
                best_length = mpi_tool.gather(best_length_to_sent, root=0)
            else:
                best_scores = mpi_tool.gather(best_score_to_sent, root=0)
            #print((best_scores, mpi_tool.rank))
            #mpi_tool.barrier()

            if i % 1 == 0 and i!=0:
                if mpi_tool.is_master:
                    """
                    scores: np.array([15, 10, 2, 8])
                    score_poistion: np.argsort(x) == array([2, 3, 1, 0])
                    exchanged_vector: [0,1,2,3]-->[0,1,0,3]
                    """
                    if return_episode_rewards:
                            #print(best_results.shape)
                            #best_scores, best_length = best_results
                        if best_scores is not None:
                            score_poistion = np.argsort(best_scores) 
                    else:
                        if best_scores is not None:
                            score_poistion = np.argsort(best_scores)
                    

                    if best_scores is not None:
                        for low_idx in score_poistion[:bottom]: 
                            exchanged_vector[score_poistion[low_idx]] = np.random.choice(score_poistion[-top:])

                    self.best_score_population = best_scores[score_poistion[-1]]
                    self.best_episode_length_population = best_length[score_poistion[-1]]
                    self.best_rank = score_poistion[-1]

                exchanged_vector = mpi_tool.bcast(exchanged_vector, root=0)

            
            #print((exchanged_vector, mpi_tool.rank))
            mpi_tool.barrier()
            #data = mpi_tool.rank
            for rec_idx in range(self.total_population_size):
                if rec_idx != exchanged_vector[rec_idx]:
                    #print(rec_idx)
                    #print(exchanged_vector[rec_idx])
                    #print(best_params_to_sent)
                    if mpi_tool.rank == exchanged_vector[rec_idx]:
                        MPI.COMM_WORLD.send(best_params_to_sent, dest=rec_idx, tag=rec_idx)
                    elif mpi_tool.rank == rec_idx:
                        best_params_to_sent=MPI.COMM_WORLD.recv(source=exchanged_vector[rec_idx], tag=rec_idx)

            #print(data, mpi_tool.rank)
            mpi_tool.barrier()
            if i % 1 == 0 and i!=0:
                for worker in self.population.agents_pool:
                    if explore and exploit:
                        #if worker.score <= rec_best_score:
                        
                        worker.exploit(best_params= best_params_to_sent)
                        worker.explore()
                    else:
                        pass
            
            
            mpi_tool.barrier()
            if mpi_tool.is_master:
                #self.best_score_population = rec_best_score
                # if return_episode_rewards:
                #     self.best_length_population = rec_best_length
                # self.best_params_population = best_params_population
                
                if (i+1) % 1 == 0 and i!=0:
                    if self.tb_writer:
                        self.tb_writer.add_scalar('Score/PBT_Results', self.best_score_population, i)
                    if return_episode_rewards:
                        if self.tb_writer:
                            self.tb_writer.add_scalar('Length/PBT_Results', self.best_episode_length_population, i)
                        print("At iteration {} the Best Pop Score is {} Best Length is {} on rank {}".format(i, self.best_score_population, self.best_episode_length_population, self.best_rank ))
                        best_agent = self.population.get_best_model()
                   
                        # saving
                        best_agent.model.save("{}/{}".format(models_dir, i))
                    else:
                        print("At iteration {} the Best Pop Score is {} on rank {}".format(i, self.best_score_population, self.best_rank))
                        best_agent = self.population.get_best_model()
                  
                        # saving
                        best_agent.model.save("{}/{}".format(models_dir, i))

def main():

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    workers = workers_init(args)
    writer = args.tb_writer
    
    num_generations = args.total_generations
    agent_training_steps = args.agent_training_steps

 
    local_size, local_agent_inds = mpi_tool.split_size(len(workers))
    
    if local_size > 1:
        raise Exception('Updates! Each rank should only one single agent')
    else:
        print("Agent Number of {} at rank {}".format(local_agent_inds, mpi_tool.rank))

    # Initializing a local population
    print("{} at rank {}".format(local_agent_inds, mpi_tool.rank))

    pbt_population = base_population()
    pbt_population.create(agent_list=[workers[i] for i in local_agent_inds])

    # Initializing a local engin
    pbt_engine = base_engine(total_population_size=args.num_agents, tb_logger=writer)
    pbt_engine.create_local(pbt_population=pbt_population)

    run1 = pbt_engine.run(steps=num_generations,exploit=True, explore=True,agent_training_steps=agent_training_steps)
    if mpi_tool.is_master:
        if writer:
            pbt_engine.tb_writer.close()

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    if mpi_tool.is_master:
        print("Total Run Time: {}".format(time_elapsed))

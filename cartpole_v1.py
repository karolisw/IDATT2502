import gym
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import math, time
import warnings
import tqdm


'''
Rules: 
The only forces that can be applied are +1 and -1, which translates to a movement of either left or right. 
If the cart moves more than 2.4 units from the center the episode is over. 
If the angle is moved more than 15 degrees from the vertical the episode is over. 
The reward is +1 for every timestamp that the episode is not over.
'''

environment = gym.make('CartPole-v1')

#Q-table
#Actions: left - right
print("Environment action space: ", environment.action_space.n)

class CartPoleAgent:
    def __init__(self, env, episodes):
        self.env = env
        # One table: actions * possible angles, second: actions*pole_vels
        self.n_bins = (6, 12)
        self.q_table = np.zeros(self.n_bins + (env.action_space.n, ))
        self.lower_bounds = [env.observation_space.low[2], -math.radians(50)]
        self.upper_bounds = [env.observation_space.high[2], math.radians(50)]
        self.lr = 0.1
        self.min_lr_rate = 0.1
        self.min_explo_rate = 0.1 # Epsilon
        self.decay = 24
        self.discount = 0.95 # Gamma -> should maybe be 0.9
        self.episodes = episodes

    #Gets dsicrete value since the original state is contionus (time)
    def get_discrete(self, state):
        _, __, angle, pole_vel = state
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int, est.transform([[angle, pole_vel]])[0]))

    # Returns the policy used for getting the next action (highest value in q table next)
    def get_policy(self, state):
        return np.argmax(self.q_table[state])

    #Q function used to update the prev q-value
    def update_q_value(self, reward, state, action, new_state):
        return self.lr * (reward + self.discount * np.max(self.q_table[new_state]) - self.q_table[state][action])

    # Adaptive lr. Used to adapt the learning rate
    def get_lr(self, n):
        return max(self.min_lr_rate, min(1.0, 1.0 - math.log10((n+1) / self.decay)))

    #D ecaying explo rate/epsilon
    def get_explo_rate(self, n):
        return max(self.min_explo_rate, min(1.0, 1.0 - math.log10((n+1) / self.decay)))

    def train(self):
        scores = []
        for e in tqdm.tqdm(range(self.episodes)):
            current_state = self.get_discrete(self.env.reset())
            self.lr = self.get_lr(e)
            done = False

            #Tracks how many inputs it survives
            score = 0
            while not done:
                action = self.get_policy(current_state)

                #Random action (exploration). Every x amount of times the agent should do a random action
                if np.random.random() <= self.get_explo_rate(e):
                    action = self.env.action_space.sample()

                #After each step an observable state, reward and if it is done is returned
                self.env.render()
                obs, reward, done, _ = self.env.step(action)
                #Discretes the new state
                new_state = self.get_discrete(obs)

                #Gets new q-value
                #learnt_value = self.update_q_value(reward, current_state, action, new_state)
                self.q_table[current_state][action] += self.update_q_value(reward, current_state, action, new_state)
                #Gets old val from q-table

                current_state = new_state
                #For each loop it statys above 12 degs, it adds to the score
                score += 1

            scores.append(score)

        #print(scores)
        print("Training done!")
        return scores

warnings.filterwarnings('ignore')
model = CartPoleAgent(environment, episodes=1000) # Change to 10000
scores = model.train()


#print(model.get_discrete_state())


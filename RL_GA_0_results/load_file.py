from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path
import pickle

import imp
from platform import architecture
import re
from this import d
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
from gym_derk import ActionKeys
import numpy as np
from scipy.special import softmax
import gym
import math
import os.path
import random


path = './again_derk.pkl'

# TO DO:
# fix counts
# fix importing from toml config file
# increase the size of the arenas to pop_size / 6 done
# mix players in the arenas
# add alternate play

reward_function = {
'damageEnemyStatue':5,
'damageEnemyUnit':5,
'killEnemyStatue':5,
'killEnemyUnit':5,
'healFriendlyStatue':5,
'healTeammate1':2,
'healTeammate2':2,
'timeSpentHomeBase':0,
'timeSpentHomeTerritory':0,
'timeSpentAwayTerritory':0,
'timeSpentAwayBase':0,
'damageTaken':-5,
'friendlyFire':-5,
'healEnemy':-5,
'fallDamageTaken':0,
'statueDamageTaken':-5,
'manualBonus':0,
'victory':0,
'loss':0,
'tie':0,
'teamSpirit':0,
'timeScaling':0,
}

actions_map = [
    [0,0,0,0,0], # nothing
    [1,0,0,0,0], # move forward
    [-1,0,0,0,0], # move backward
    [0,1,0,0,0], # turn right
    [0,-1,0,0,0], # turn left
    [0,0,1,0,0], # chase focus
    [0,0,0,1,0], # cast 1
    [0,0,0,2,0], # cast 2
    [0,0,0,3,0], # cast 3
    [0,0,0,0,1], # focus 1
    [0,0,0,0,2], # focus friend 2
    [0,0,0,0,3], # focus friend 3
    [0,0,0,0,4], # focus enemy statue
    [0,0,0,0,5], # focus enemy 1
    [0,0,0,0,6], # focus enemy 2
    [0,0,0,0,7], # focus emeny 3
    ]

pop_size=3
# how many parallel arenas to play
n_arenas = 1

env = DerkEnv(turbo_mode=True,n_arenas=n_arenas,reward_function=reward_function,
   home_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#00ff00', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#ff0000', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],}
   ],
#    away_team=[
#       { 'primaryColor': '#ff00ff', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
#       { 'primaryColor': '#00ff00', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
#       { 'primaryColor': '#ff0000','slots': ['Cleavers', 'Shell', 'ParalyzingDart'],}
#    ],
   away_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#00ff00', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#ff0000', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],}
   ],
   
)
# dumb semi-deterministic policy
class Derk_Player:
    # create new player given some weights/state. 
    # Random to set initial players to random
    # n_states is the number of different states the derkling is able to learn
    # state is the mapping function from observations to states. It also serves to discretize and compress the input
    # q_table is the optional environment q table to inherit
    # a_table is the optional action discretization table to inherit
    def __init__(self,state=None,random_state=False,
                 q_table=None,n_states=100,
                 alpha = 0.5, discount = 0.5,):
        
        # init the "comprehension" of the state
        if state is np.ndarray :
            # state is a matrix that maps observations to states
            self.state = np.copy(state)
        else:
            # state is a matrix of genes that encode the mapping from observations to states
            self.state = np.zeros((1,len(ObservationKeys)))
            if random_state:
                self.state += np.random.choice([-1,0,1],size=(1,len(ObservationKeys)))
    
        # init the q table
        if q_table is np.ndarray:
            self.q_table = np.copy(q_table)
        else:
            self.q_table = np.zeros((n_states,len(actions_map) ))
            
        # init Q learning hyperparameters
        self.alpha = alpha
        self.discount = discount
        self.n_states = n_states
        self.current_state = 0
        self.current_actions = np.zeros((1,len(ActionKeys)))
        self.current_action = 0
    # maps observations into states by the inner matrix ( genes ).
    def observation_to_state(self, observations):
        state = np.matmul(self.state,observations)
        # import pdb; pdb.set_trace()
        return int(state*self.q_table.shape[1]) % self.q_table.shape[1]
    # compute best actions for a given state, using E-policy
    def actions(self, observations,eps):
        actions = [0,0,0,0,0]
        best_action = 0
        p = np.random.rand(1,1)
        # random action
        if p < eps:
            action = np.random.randint(0,len(actions_map))
            actions = actions_map[action]
        # follow greedy policy
        else:
            best_action = np.asarray(np.argmax(self.q_table[self.current_state])).reshape(1)
            actions = actions_map[best_action[0]]
        self.current_actions = actions
        self.current_action = best_action
        return actions
    # def action_compression(self,actions):
    #     action = np.matmul(self.state,actions)
    #     return int(actions*self.a_compression.size[1]) % self.a_compression_size[1]
    def update_policy(self,observations,reward):
        # save previous state
        previous_state = self.current_state
        # compute the mapping to the current state
        self.current_state = self.observation_to_state(observations)
        # maximum q value for the new current state
        max_q = np.max(self.q_table[self.current_state])
        # split this part after actions are performed
        current_q_value = self.q_table[previous_state,self.current_action]
        self.q_table[previous_state,self.current_action] += self.alpha * (reward + self.discount * max_q - current_q_value)

    # mutate this player
    def mutate(self, mutation_rate=0.1, mutation_width=0.1):
        # mutate comprehension of the observations
        mutation_count = np.random.poisson(mutation_rate*len(ObservationKeys))
        for i in range(mutation_count):
            gene_pos_x = (np.random.randint(0,self.state.shape[0]))
            gene_pos_y = (np.random.randint(0,self.state.shape[1]))
            # import pdb; pdb.set_trace()
            self.state[gene_pos_x,gene_pos_y] += np.random.normal(self.state[gene_pos_x,gene_pos_y],size=1, scale=mutation_width) 
        mutation_count = np.random.poisson(mutation_rate*len(ObservationKeys)*self.n_states)
        for i in range(mutation_count):
            gene_pos_x = (np.random.randint(0,self.q_table.shape[0]))
            gene_pos_y = (np.random.randint(0,self.q_table.shape[1]))
            # import pdb; pdb.set_trace()
            self.q_table[gene_pos_x,gene_pos_y] += np.random.normal(self.q_table[gene_pos_x,gene_pos_y],size=1, scale=mutation_width) 
         
         # mutate hyperparamters of Q learning, alpha and discount
        p = np.random.random()
        if p<mutation_rate:
            self.alpha += np.random.normal(self.alpha,scale=0.1)
        p = np.random.random()
        if p<mutation_rate:
            self.discount += np.random.normal(self.discount,scale=0.1)
        # if unlikely scenario, add or remove rows from the q table/a table
        # p = np.random.random((1,1))
        # if p<mutation_rate:
        #     np.append(self.s_table,np.zeros((self.q_table.shape[0],1)))
    # creates a new player by crossing this player with another player
    def crossover(self, lover, cross_over_rate=0.5, mutation_rate=0.1):
        crossover_index = int(len(ObservationKeys)*cross_over_rate)
        child_state = np.concatenate(
            (self.state[crossover_index:],
            lover.state[:crossover_index]))
        child_q_table = np.concatenate(
            (self.q_table[crossover_index:],
            lover.q_table[:crossover_index]))
        # child_alpha = (self.alpha+lover.alpha)/2
        # child_discount = (self.discount+lover.discount)/2
        # child_state = (child_state- np.min(child_state)) / (np.max(child_state) - np.min(child_state))
        # child_q_table = (child_q_table - np.min(child_q_table)) / (np.max(child_q_table) - np.min(child_q_table))
        child_alpha = (self.alpha)
        child_discount = (self.discount)
        return Derk_Player(state=child_state,q_table=child_q_table,alpha=child_alpha,discount=child_discount)
    
# initial population
players_home = [Derk_Player(random_state=False) for i in range(pop_size)]
players_away = [Derk_Player() for j in range(pop_size)]

player_home = pickle.load(open(path,'rb'))
for e in range(100):
  observation_n = env.reset()
  count = 0
  while True:
    action_n_home = [players_home[i].actions(observation_n[i],0) for i in range(pop_size)]
    action_n_away = [players_away[j].actions(observation_n[j+pop_size],0) for j in range(pop_size)]
    # import pdb; pdb.set_trace()
    # receive the rewards and update policies
    observation_n, reward_n, done_n, info = env.step([*action_n_home, *action_n_away])
    if all(done_n):
            print("Episode {} finished".format(e))
            break
    
env.close()
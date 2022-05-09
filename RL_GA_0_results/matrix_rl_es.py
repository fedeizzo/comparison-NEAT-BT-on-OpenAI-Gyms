# try to do a simple evolutionary algorithm to solve  this problem
# the goal is to find the best solution to the problem
# Each player has a set of genes that are used to determine a mapping from observations to states
# Using that function, then a reinforcement learning procedure is used to train the q function

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

# to save winning model
import pickle
# to flush directly in log files
import functools

from torch import alpha_dropout, rand
print = functools.partial(print, flush=True)


tournament_size_home = 5 # size of the tournament for reproductive selection. Within each tournament, the best reproduction_parents are selected for reproduction
mutation_rate_home = 0.05 # percentage of genes that on average will be mutated
mutation_width_home = 10 # the width of the mutation
reproduction_parents_home = 2 # number of parents to combine to generate a child
# the selection is done in a lamba + m
offsprings_count_home = 200 # number of offspring to generate at each generation. They replace the worst offsprint_count derklings in the population

tournament_size_away = 5 # size of the tournament for reproductive selection. Within each tournament, the best reproduction_parents are selected for reproduction
mutation_rate_away = 0.05 # percentage of genes that on average will be mutated
mutation_width_away = 10 # the width of the mutation
reproduction_parents_away = 2 # number of parents to combine to generate a child
# the selection is done in a lamba + m
offsprings_count_away = 200 # number of offspring to generate at each generation. They replace the worst offsprint_count derklings in the population

pop_size = 600 # number of derklings in the population, for each team, required to be multiple of 3.
#Every 3 derklings play a game in a single arena 

# TO DO:
# fix counts
# fix importing from toml config file
# increase the size of the arenas to pop_size / 6 done
# mix players in the arenas
# add alternate play

reward_function = {
'damageEnemyStatue':5,
'damageEnemyUnit':5,
'killEnemyStatue':20,
'killEnemyUnit':20,
'healFriendlyStatue':3,
'healTeammate1':2,
'healTeammate2':2,
'timeSpentHomeBase':0,
'timeSpentHomeTerritory':0,
'timeSpentAwayTerritory':5,
'timeSpentAwayBase':0,
'damageTaken':-1,
'friendlyFire':-3,
'healEnemy':-3,
'fallDamageTaken':-1,
'statueDamageTaken':-3,
'manualBonus':0,
'victory':100,
'loss':-100,
'tie':0,
'teamSpirit':1,
'timeScaling':0.4,
}


# how many parallel arenas to play
n_arenas = int(pop_size / 3)

env = DerkEnv(turbo_mode=True,n_arenas=n_arenas,reward_function=reward_function,
   home_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#00ff00', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],},
      { 'primaryColor': '#ff0000', 'slots': ['Blaster', 'FrogLegs', 'HealingGland'],}
   ],
   away_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
      { 'primaryColor': '#00ff00', 'slots': ['Cleavers', 'Shell', 'ParalyzingDart'],},
      { 'primaryColor': '#ff0000','slots': ['Cleavers', 'Shell', 'ParalyzingDart'],}
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
                 q_table=None,
                 a_compression=None,random_action=False,
                 n_states=100,n_actions = 10,
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
    
        # init the action "comprehension"
        if a_compression is np.ndarray:
            self.a_compression = np.copy(a_compression)
        else: 
            # state is a matrix of genes that encode actions
            self.a_compression = np.zeros((1,len(ActionKeys)))
            if random_action:
                self.a_compression += np.random.choice([-1,0,1],size=(1,len(ActionKeys)))
        
        # init the q table
        if q_table is np.ndarray:
            self.q_table = np.copy(q_table)
        else:
            self.q_table = np.zeros((n_states,n_actions ))
            
        # init Q learning hyperparameters
        self.alpha = alpha
        self.discount = discount

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
        actions = np.zeros((len(ActionKeys),))
        best_action = 0
        p = np.random.rand(1,1)
        # random action
        if p < eps:
            actions[0] = (np.random.rand(1,1)-0.5)*2
            actions[1] = (np.random.rand(1,1)-0.5)*2
            actions[2] = (np.random.rand(1,1))
            actions[3] = int((np.random.rand(1,1))*4)%4
            actions[4] = int((np.random.rand(1,1)-0.5)*8)%8
        # follow greedy policy
        else:
            best_action = np.asarray(np.argmax(self.q_table[self.current_state])).reshape(1)
            actions = np.matmul(best_action,self.a_compression)
            actions[0] = actions[0]%2-1
            actions[1] = actions[1]%2-1
            actions[2] = actions[2]%1
            actions[3] = actions[3]%4
            actions[4] = actions[4]%8
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
        # mutate comprehension of the actions
        mutation_count = np.random.poisson(mutation_rate*len(ActionKeys))
        for i in range(mutation_count):
            gene_pos_x = (np.random.randint(0,self.a_compression.shape[0]))
            gene_pos_y = (np.random.randint(0,self.a_compression.shape[1]))
            self.a_compression[gene_pos_x,gene_pos_y] += np.random.normal(self.a_compression[gene_pos_x,gene_pos_y],size=1, scale=mutation_width)
        # mutate hyperparamters of Q learning, alpha and discount
        p = np.random.random((1,1))
        if p<mutation_rate:
            self.alpha += np.random.normal(self.alpha,scale=mutation_width)
        p = np.random.random((1,1))
        if p<mutation_rate:
            self.discount += np.random.normal(self.discount,scale=mutation_width)
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
        child_a_compression = np.concatenate(
            (self.a_compression[crossover_index:],
            lover.a_compression[:crossover_index]))
        child_alpha = (self.alpha+lover.alpha)/2
        child_discount = (self.discount+lover.discount)/2
        return Derk_Player(state=child_state,a_compression=child_a_compression,q_table=child_q_table,alpha=child_alpha,discount=child_discount)
    
# initial population
players_home = [Derk_Player() for i in range(pop_size)]
players_away = [Derk_Player() for i in range(pop_size)]

def evolutionary_strategy(players,rewards, tournament_size, mutation_rate , mutation_width, reproduction_parents, pop_size, offsprings_count):
    # get the fitness of the players
    offsprings = []
    # create the new population
    for i in range(offsprings_count):
        # chose the randomly player to make them partecipate in the selective tournament
        tournament = np.random.choice([i for i in range(pop_size)],size=tournament_size,replace=False)
        # chose the best n players to reproduce
        # need to sort the tournament by fitness and take the best n
        sort_indexes = sorted([derk for derk in tournament], key=lambda k: rewards[k],reverse=True)
        # slice only the best n
        best = sort_indexes[:reproduction_parents]
        # reproduce the best n
        offspring = players[best[0]].crossover(players[best[1]],cross_over_rate=0.5,mutation_rate=mutation_rate)
        offspring.mutate(mutation_rate,mutation_width)
        offsprings.append(offspring)
    # replace the worst n players with the new offsprings
    sort_indexes = sorted([i for i in range(pop_size)], key=lambda k: rewards[k],reverse=False)
    worst = sort_indexes[:offsprings_count]
    # import pdb; pdb.set_trace()
    for bad,offspring in zip(worst,offsprings):
        del(players[bad])
        players.append(offspring)
    # print(players_home)
    # import pdb; pdb.set_trace()

for e in range(500):
    observation_n = env.reset()
    count = 0
    
    while True:
        # do actions according to obeservations
        action_n_home = [players_home[i].actions(observation_n[i],0.1) for i in range(pop_size)]
        action_n_away = [players_away[i].actions(observation_n[i+pop_size],0.1) for i in range(pop_size)]
        # receive the rewards and update policies
        observation_n, reward_n, done_n, info = env.step([*action_n_home, *action_n_away])
        for i,player in enumerate(players_home):
            player.update_policy(observation_n[i],reward_n[i])
        for i,player in enumerate(players_away):
            player.update_policy(observation_n[i+pop_size],reward_n[i+pop_size])
        # print()
        if all(done_n):
            print("Episode {} finished".format(e))
            break
    # total_reward = env.total_reward()
    
    # genetic part part
    # get the rewards of the players
    # import pdb; pdb.set_trace()
    reward_home = env.total_reward[:pop_size]
    reward_away = env.total_reward[pop_size:]
    # evolve
    evolutionary_strategy(players_home,reward_home, tournament_size_home, mutation_rate_home,mutation_width_home, reproduction_parents_home, pop_size, offsprings_count_home)
    evolutionary_strategy(players_away,reward_away, tournament_size_away, mutation_rate_away,mutation_width_away, reproduction_parents_away, pop_size, offsprings_count_away)
    # import pdb;pdb.set_trace()
    # mix the players so they don't always play the with the same team and against the same enemies
    random.shuffle(players_home)
    random.shuffle(players_away)
    print(reward_home)
env.close()
print(reward_home)
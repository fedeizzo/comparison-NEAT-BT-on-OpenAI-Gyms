# try to do a simple evolutionary algorithm to solve  this problem
# the goal is to find the best solution to the problem
# Each player has a set of genes that are used to determine their actions
# Those genes are just a matrix of numbers
# Each action is deterministically mapped to a gene by matrix multiplication


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

from torch import rand
print = functools.partial(print, flush=True)


tournament_size_home = 5 # size of the tournament for reproductive selection. Within each tournament, the best reproduction_parents are selected for reproduction
mutation_rate_home = 0.05 # percentage of genes that on average will be mutated
mutation_width_home = 10 # the width of the mutation
reproduction_parents_home = 2 # number of parents to combine to generate a child
# the selection is done in a lamba + m
offsprings_count_home = 20 # number of offspring to generate at each generation. They replace the worst offsprint_count derklings in the population

tournament_size_away = 5 # size of the tournament for reproductive selection. Within each tournament, the best reproduction_parents are selected for reproduction
mutation_rate_away = 0.05 # percentage of genes that on average will be mutated
mutation_width_away = 10 # the width of the mutation
reproduction_parents_away = 2 # number of parents to combine to generate a child
# the selection is done in a lamba + m
offsprings_count_away = 20 # number of offspring to generate at each generation. They replace the worst offsprint_count derklings in the population

pop_size = 60 # number of derklings in the population, for each team, required to be multiple of 3.
#Every 3 derklings play a game in a single arena 

# TO DO:
# fix counts
# fix importing from toml config file
# increase the size of the arenas to pop_size / 6 done
# mix players in the arenas
# add alternate play

reward_function = {
'damageEnemyStatue':100,
'damageEnemyUnit':100,
'killEnemyStatue':50,
'killEnemyUnit':50,
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


# how many parallel arenas to play
n_arenas = int(pop_size / 3)

env = DerkEnv(turbo_mode=True,n_arenas=n_arenas,reward_function=reward_function,
   home_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Pistol', None, None],},
      { 'primaryColor': '#00ff00', 'slots': ['Pistol', None, None],},
      { 'primaryColor': '#ff0000', 'slots': ['Pistol', None, None],}
   ],
   away_team=[
      { 'primaryColor': '#ff00ff', 'slots': ['Pistol', None, None],},
      { 'primaryColor': '#00ff00', 'slots': ['Pistol', None, None],},
      { 'primaryColor': '#ff0000','slots': ['Pistol', None, None],}
   ],
   
)
# dumb semi-deterministic policy
class Derk_Player:
    # create new player given some weights/state. 
    # Random to set initial players to random
    def __init__(self,state=None,random = False):
        if state is np.ndarray:
            self.state = np.copy(state)
        else:
            # just keep a matrix. 
            # players deterministically do actions based on a simple linear combination of the observations
            self.state = np.zeros((1,(len(ObservationKeys))))
            if random:
                self.state += np.random.choice([-1,0,1],size=(1,(len(ObservationKeys))))
    # creates a perfect copy of this player
    def clone(self):
        return Derk_Player(np.copy(self.state))
    # compute actions as a linear combination of the observations
    def actions(self, observations,eps):
        best_action = np.matmul(self.state,observations)
        actions = actions_map[best_action]
        return actions

    # mutate this player
    def mutate(self, mutation_rate=0.1, mutation_width=0.1):
        # mutate comprehension of the observations
        mutation_count = np.random.poisson(mutation_rate*len(ObservationKeys)*len(ActionKeys))
        for i in range(mutation_count):
            gene_pos_x = (np.random.randint(0,self.state.shape[0]))
            gene_pos_y = (np.random.randint(0,self.state.shape[1]))
            # import pdb; pdb.set_trace()
            self.state[gene_pos_x,gene_pos_y] += np.random.normal(self.state[gene_pos_x,gene_pos_y],size=1, scale=mutation_width) 
            
    # creates a new player by crossing this player with another player
    def crossover(self, lover, cross_over_rate=0.5, mutation_rate=0.1):
        # print(lover)
        crossover_index = int(len(ObservationKeys)*len(ActionKeys)*cross_over_rate)
        # print(crossover_index)
        child = np.concatenate(
            (self.state.reshape((len(ObservationKeys)*len(ActionKeys),1))[crossover_index:],
            lover.state.reshape((len(ObservationKeys)*len(ActionKeys),1))[:crossover_index]))
        return Derk_Player(child.reshape(len(ObservationKeys),len(ActionKeys)))
    # creates a new player by crossing this player with n other players
    # def multiple_crossover(self, lover:list, cross_over_rate=0.5, mutation_rate=0.1):
    #     crossover_indexes = int(len(ObservationKeys)*len(ActionKeys)*cross_over_rate)
    #     child = np.concatenate(
    #         self.state.reshape(len(ObservationKeys)*len(ActionKeys))[crossover_index:],
    #         lover.state.reshape(len(ObservationKeys)*len(ActionKeys))[:crossover_index])
    #     child.mutate(mutation_rate)
    #     return Derk_Player(child.reshape(len(ObservationKeys),len(ActionKeys)))
path = './results.pkl/'
# initial population
if os.path.exists(path):
    players_home = pickle.load(open(path,'rb'))
else:
    players_home = [Derk_Player() for i in range(pop_size)]
players_away = [Derk_Player() for j in range(pop_size)]

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
        offspring = players[best[0]].crossover(players[best[1]])
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
        action_n_home = [players_home[i].actions(observation_n[i]) for i in range(pop_size)]
        action_n_away = [players_away[i].actions(observation_n[i+pop_size]) for i in range(pop_size)]
        observation_n, reward_n, done_n, info = env.step([*action_n_home, *action_n_away])
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
with open(path, "wb") as f:
            pickle.dump(players_home, f)
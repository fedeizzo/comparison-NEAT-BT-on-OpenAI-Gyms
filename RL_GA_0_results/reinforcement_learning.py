# try to do a simple evolutionary algorithm to solve  this problem
# the goal is to find the best solution to the problem
# Each player has a set of genes that are used to determine a mapping from observations to states
# Using that function, then a reinforcement learning procedure is used to train the q function

import imp
from platform import architecture
import re
from this import d
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
from gym_derk import ActionKeys
import numpy as np
from pygments import highlight
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
mutation_width_home = 1 # the width of the mutation
reproduction_parents_home = 2 # number of parents to combine to generate a child
# the selection is done in a lamba + m
offsprings_count_home = 30 # number of offspring to generate at each generation. They replace the worst offsprint_count derklings in the population

tournament_size_away = 20 # size of the tournament for reproductive selection. Within each tournament, the best reproduction_parents are selected for reproduction
mutation_rate_away = 0.05 # percentage of genes that on average will be mutated
mutation_width_away = 1 # the width of the mutation
reproduction_parents_away = 2 # number of parents to combine to generate a child
# the selection is done in a lamba + m
offsprings_count_away = 30 # number of offspring to generate at each generation. They replace the worst offsprint_count derklings in the population

pop_size = 102 # number of derklings in the population, for each team, required to be multiple of 3.
#Every 3 derklings play a game in a single arena 
#path
path = './rl.pkl'


reward_function = {
'damageEnemyStatue':5,
'damageEnemyUnit':5,
'killEnemyStatue':5,
'killEnemyUnit':5,
'healFriendlyStatue':5,
'healTeammate1':5,
'healTeammate2':5,
'timeSpentHomeBase':-5,
'timeSpentHomeTerritory':-5,
'timeSpentAwayTerritory':5,
'timeSpentAwayBase':5,
'damageTaken':0,
'friendlyFire':-5,
'healEnemy':-5,
'fallDamageTaken':0,
'statueDamageTaken':-5,
'manualBonus':0,
'victory':0,
'loss':0,
'tie':0,
'teamSpirit':0,
'timeScaling':0.4,
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

mask = np.asarray([False for i in range(len(ObservationKeys))])
mask[8:20] = False
mask_size = len(mask)-np.sum(mask)


class Derk_Player:
    # map observations to states using euclidean distance
    states = [np.asarray([0 for i in range(mask_size)])]
    q_table = [np.asarray([0 for i in range(len(actions_map))])]
    alpha = 0.5
    discount = 0.5
    def __init__(self):
        self.current_state = 0
        self.current_action = 0
        
    # maps observations into states by the inner matrix ( genes ).
    def observation_to_state(self, observations):
        highest = 0
        highest_index = 0
        for i,state in enumerate(self.states):
            similarity = np.linalg.norm(state-observations)
            if similarity > highest:
                highest = similarity
                highest_index = i
        if highest > 5:
            self.states.append(observations)
            self.q_table.append(np.zeros(len(actions_map)))
        return highest_index
        
    # compute best actions for a given state, using E-policy
    def actions(self, observations,eps):
        p = np.random.rand(1,1)
        # random action
        if p < eps:
            best_action = np.random.randint(0,len(actions_map))
            actions = actions_map[best_action]
        # follow greedy policy
        else:
            best_action = np.argmax(self.q_table[self.current_state])
            actions = actions_map[best_action]
        self.current_action = best_action
        return actions
    # q leaning  update policy rule
    def update_policy(self,observations,reward):
        # save previous state
        previous_state = self.current_state
        # compute the mapping to the current state
        self.current_state = self.observation_to_state(observations)
        # maximum q value for the new current state
        max_q = np.max(self.q_table[self.current_state])
        current_q_value = self.q_table[previous_state][self.current_action]
        self.q_table[previous_state][self.current_action] += self.alpha * (reward + self.discount * max_q - current_q_value)

    # mutate this player
    def mutate(self, mutation_rate=0.1, mutation_width=0.1):
        # mutate comprehension of the observations
        mutation_count = np.random.poisson(mutation_rate*self.state.shape[1]*self.state.shape[0])
        for i in range(mutation_count):
            gene_pos_x = (np.random.randint(0,self.state.shape[0]))
            gene_pos_y = (np.random.randint(0,self.state.shape[1]))
            # import pdb; pdb.set_trace()
            self.state[gene_pos_x,gene_pos_y] += np.random.normal(self.state[gene_pos_x,gene_pos_y],size=1, scale=mutation_width) 
        mutation_count = np.random.poisson(mutation_rate*self.q_table.shape[1]*self.q_table.shape[0])
        for i in range(mutation_count):
            gene_pos_x = (np.random.randint(0,self.q_table.shape[0]))
            gene_pos_y = (np.random.randint(0,self.q_table.shape[1]))
            # import pdb; pdb.set_trace()
            self.q_table[gene_pos_x,gene_pos_y] += np.random.normal(self.q_table[gene_pos_x,gene_pos_y],size=1, scale=mutation_width) 
         
         # mutate hyperparamters of Q learning, alpha and discount
        # p = np.random.random()
        # if p<mutation_rate:
        #     self.alpha += np.random.normal(self.alpha,scale=0.1)
        # p = np.random.random()
        # if p<mutation_rate:
        #     self.discount += np.random.normal(self.discount,scale=0.1)
        # if unlikely scenario, add or remove rows from the q table/a table
        # p = np.random.random((1,1))
        # if p<mutation_rate:
        #     np.append(self.s_table,np.zeros((self.q_table.shape[0],1)))
    # creates a new player by crossing this player with another player
    def crossover(self, lover, cross_over_rate=0.5, mutation_rate=0.1):
        crossover_index = int(self.state.shape[1]*cross_over_rate)
        # child_state = (self.state+lover.state)/2
        # child_q_table = (self.q_table+lover.q_table)/2
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
    def copy(self):
        return Derk_Player(state=self.state,q_table=self.q_table,alpha=self.alpha,discount=self.discount)
# initial population
if os.path.exists(path):
    players_home = pickle.load(open(path,'rb'))
else:
    players_home = [Derk_Player() for i in range(pop_size)]
players_away = [Derk_Player() for j in range(pop_size)]
while len(players_home) < pop_size:
    cp =players_home[0].copy()
    players_home.append(cp)
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

for e in range(50):
    observation_n = env.reset()
    observation_n_masked = np.asarray([np.delete(observation_n[i], np.where(mask)) for i in range(pop_size*2)])
    # print('')
    
    while True:
        # do actions according to obeservations
        action_n_home = [players_home[i].actions(observation_n_masked[i],0.5) for i in range(pop_size)]
        action_n_away = [players_away[j].actions(observation_n_masked[j+pop_size],1) for j in range(pop_size)]
        # import pdb; pdb.set_trace()
        # receive the rewards and update policies
        observation_n, reward_n, done_n, info = env.step([*action_n_home, *action_n_away])
        observation_n_masked = np.asarray([np.delete(observation_n[i], np.where(mask)) for i in range(pop_size*2)])
        # for i,player in enumerate(players_home):
        print(players_home[0].current_state)
        # print(np.max(players_home[0].q_))
        
        for i,player in enumerate(players_home):
            player.update_policy(observation_n_masked[i],reward_n[i])
        for i,player in enumerate(players_away):
            player.update_policy(observation_n_masked[i+pop_size],reward_n[i+pop_size])
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
    # evolutionary_strategy(players_home,reward_home, tournament_size_home, mutation_rate_home,mutation_width_home, reproduction_parents_home, pop_size, offsprings_count_home)
    # evolutionary_strategy(players_away,reward_away, tournament_size_away, mutation_rate_away,mutation_width_away, reproduction_parents_away, pop_size, offsprings_count_away)
    # import pdb;pdb.set_trace()
    # mix the players so they don't always play the with the same team and against the same enemies
    # random.shuffle(players_home)
    # random.shuffle(players_away)
    # reset state and actions
    for player in players_home:
        player.current_action = 0
        player.current_actions = np.zeros((1,len(ActionKeys)))
        player.current_state = 0
    for player in players_away:
        player.current_action = 0
        player.current_actions = np.zeros((1,len(ActionKeys)))
        player.current_state = 0
    print("average reward home :{}".format(np.average(np.asarray(reward_home))))
    print("average reward away :{}".format(np.average(np.asarray(reward_away))))
    print("max reward home :{}".format(np.max(np.asarray(reward_home))))
    print("max reward away :{}".format(np.max(np.asarray(reward_away))))
env.close()
with open(path, "wb") as f:
            pickle.dump(players_home, f)
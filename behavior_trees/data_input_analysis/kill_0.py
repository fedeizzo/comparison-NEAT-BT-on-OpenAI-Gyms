# even simpler script to load the observation from the environment of a manually controlled agent
from gym_derk.envs import DerkEnv
import numpy as np
import pickle

env = DerkEnv(n_arenas=1,turbo_mode=True,home_team=[
            {
                "primaryColor": "#ce03fc",
                "slots": ["Cleavers", "Shell", "HealingGland"],
            },
            {
                "primaryColor": "#8403fc",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
            {
                "primaryColor": "#0331fc",
                "slots": ["Blaster", "FrogLegs", "HealingGland"],
            },
        ],)
obs_all = np.zeros((1,150,6,64))
for i in range(1):
  observation_n = env.reset()
  j=0
  while True:
    action_n = [np.asarray([0,0,0,0,0]) for i in range(env.n_agents)]
    # import pdb; pdb.set_trace()
    if j==0:
      action_n[0] = [0,0,0,0,5]
    elif j<20:
      action_n[0] = [0,0,1,0,0]
    else:
      action_n[0] = [0,0,0,1,0]   
    observation_n, reward_n, done_n, info = env.step(action_n)
    obs_all[i,j] += observation_n
    j+=1
    if all(done_n):
      print(f"Episode {i} finished")
      break
env.close()
pickle.dump(obs_all, open("obs_all.b", "wb"))
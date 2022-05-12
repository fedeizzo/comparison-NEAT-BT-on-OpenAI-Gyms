# script to load the observation from the environment into a binary file.
# Later on, using the binary file, we can load the data into a numpy array and check properties of it
from gym_derk.envs import DerkEnv
import numpy as np
import pickle
env = DerkEnv(n_arenas=10,turbo_mode=True)
obs_all = np.zeros((10,200,60,64))
for i in range(10):
  observation_n = env.reset()
  j=0
  while True:
    action_n = [env.action_space.sample() for i in range(env.n_agents)]
    observation_n, reward_n, done_n, info = env.step(action_n)
    obs_all[i,j] += observation_n
    j+=1
    if all(done_n):
      print(f"Episode {i} finished")
      break
env.close()
pickle.dump(obs_all, open("obs_all.b", "wb"))
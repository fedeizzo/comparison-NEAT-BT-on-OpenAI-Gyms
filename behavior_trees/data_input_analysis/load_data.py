# script to load the observation from the environment
import pickle
import numpy as np
data = pickle.loads(open("obs_all.b", "rb").read())
import pdb; pdb.set_trace()
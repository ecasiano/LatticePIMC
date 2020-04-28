# Benchmark insertion/deletion of kinks
import pimc
import numpy as np
import matplotlib.pyplot as plt
import bisect
from random import shuffle
from scipy.stats import truncexpon
from scipy.integrate import quad, simps
import importlib
import argparse
importlib.reload(pimc)
import pickle
import random
import datetime
import time
from worldline import Kink,Worldline,Worm

start = time.time()

# Bose-Hubbard Parameters (1D)
L = int(4E+00)
N = L
D = 1

# Randomly generate Fock state at tau=0
alpha_0 = pimc.random_boson_config(L,D,N)

# Initialize list that will store worldlines (paths)
paths = []

# Initialize list that will store kink handles (helper_kinks)
max_num_kinks = int(1E+07)
num_kinks = 0 # Kinks IN USE
helper_kinks = [Kink(None,None,None,None,None)] * max_num_kinks

# Fill out paths and helper kinks with each site's initial kinks
for site,n in enumerate(alpha_0):
	paths.append(Worldline(n,site))
	helper_kinks[site] = paths[site].first
	num_kinks += 1

end = time.time()

print("Paths and helper list creation: %.2f seconds"%(end-start))

print(helper_kinks[:10])



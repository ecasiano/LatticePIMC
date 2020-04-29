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

# Bose-Hubbard Parameters (1D)
L = int(1)
N = L
D = 1
beta = 1

# Randomly generate Fock state at tau=0
alpha_0 = pimc.random_boson_config(L,D,N)

'------------------------------------------------------------------------------'
'''Linked list version'''

# Initialize list that will store worldlines of each site (paths)
paths = []

# Initialize list that will store kink handles (helper_kinks)
max_num_kinks = int(2E+07)
num_kinks = 0 # Kinks IN USE
helper_kinks = [Kink(None,None,None,None,None)] * max_num_kinks

# Fill out paths and helper kinks with each site's initial kinks
for site,n in enumerate(alpha_0):
	paths.append(Worldline(n,site))
	helper_kinks[site] = paths[site].first
	num_kinks += 1

# Grow the number of kinks in the worldline
test_size = int(1000000)
for i in range(test_size):

	# Randomly choose prev_kink from helper array
	r = int(np.random.random()*num_kinks)
	prev_kink = helper_kinks[r]

	# Determine length of flat interval
	if prev_kink.next is not None:
		tau_flat = prev_kink.next.tau - prev_kink.tau 
	else: 
		tau_flat = beta-prev_kink.tau

	tau = prev_kink.tau + np.random.random()*tau_flat
	n = int(np.random.random())*L
	src = int(np.random.random())*L
	dest = int(np.random.random())*L
	label = num_kinks+1
	new_kink = Kink(tau,n,src,dest,label)

	# Insert new_kink to worldline
	paths[0].insert(prev_kink,new_kink)

	# Insert kink to helper array
	helper_kinks[num_kinks] = new_kink
	num_kinks += 1

# Alternate between insert and delete many times
start = time.time()
insertions = int(1E+05)
for i in range(insertions):

	'''Insertion'''

	# Sample prev_kink (lower bound of flat interval) from helper array
	r = int(np.random.random()*num_kinks)
	prev_kink = helper_kinks[r] 

	# Determine length of flat interval
	if prev_kink.next is not None:
		tau_flat = prev_kink.next.tau - prev_kink.tau 
	else: 
		tau_flat = beta-prev_kink.tau

	# Generate kink data
	tau = prev_kink.tau + np.random.random()*tau_flat
	n = int(np.random.random())*L
	src = int(np.random.random())*L
	dest = int(np.random.random())*L
	label = i+1

	# Create the new kink
	new_kink = Kink(tau,n,src,dest,label)

	# Insert new_kink to worldline
	paths[0].insert(prev_kink,new_kink)

	# Insert kink to helper array
	helper_kinks[num_kinks] = new_kink
	num_kinks += 1

	'''Deletion'''

	# Sample kink to be deleted from helper array
	r = L + int(np.random.random()*(num_kinks-L)) # Cannot delete initial kinks
	kink_to_remove = helper_kinks[r]

	# Delete kink from worldline
	paths[0].delete(kink_to_remove)

	# In helper_array, swap deleted kink with last used kink
	helper_kinks[r],helper_kinks[num_kinks-1] = \
	helper_kinks[num_kinks-1],helper_kinks[r]


	# del helper_kinks[r]
	# helper_kinks.append(Kink(None,None,None,None,None))

	num_kinks -= 1

end = time.time()
print("Time elapsed to perform %d inserts/deletes on \
worldline with %d initial kinks: %.4f seconds"%(insertions,test_size,end-start))

# print('\n',paths[0],'\n')
# print(helper_kinks[:num_kinks+2])
'------------------------------------------------------------------------------'
'''Python list version'''

# Old version:
# 3) Create data_struct (list version) for one site
# 4) Grow data_struct[0] to desired size
# 5) Randomly sample a flat region and use list.insert()
# 6) Randomly sample a flat region and delete lower bound kink



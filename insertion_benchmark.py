import pimc # custom module
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
import worldline as wl
import time



# Set system size, dimension and particle number
L = 2
D = 1
N = 2

'''------------------------ Linked List ------------------------'''

print("Linked List")

# Create worldline configuration
worldlines = []
for i in range(L):
	worldlines.append(wl.Worldline(1,-1))

print("Worldlines: ", worldlines)

# Do many insertions
iterations = int(4.0E+04)
for iteration in range(iterations):

	# Randomly choose a lattice site
	i = int(np.random.random()*L)

	# Randomly choose an imaginary time
	tau = np.random.random()

	# These don't affect sorting
	n = -1
	src = -1
	dest = -1

	# Do insertion on chosen site
	worldlines[i].test_insert(tau,n,src,dest)

start = time.time()

# Do many insertions and removes
for iteration in range(iterations):

	# Randomly choose a lattice site
	i = int(np.random.random()*L)

	# Randomly choose an imaginary time
	tau = np.random.random()

	# These don't affect sorting
	n = -1
	src = -1
	dest = -1

	# Do insertion on chosen site
	test_kink = worldlines[i].test_insert(tau,n,src,dest)
	worldlines[i].remove(test_kink)

end = time.time()

print("%d insertions/removals done in %.2f seconds."%(iterations,(end-start)),'\n')


'''------------------------ Python List ------------------------'''

print("Python List")
# Create worldline configuration
worldlines = []
for i in range(L):
	worldlines.append([0,1,(-1,-1)])

print("Worldlines: ", worldlines)

# Do many insertions
for iteration in range(iterations):

	# Randomly choose a lattice site
	i = int(np.random.random()*L)

	# Randomly choose an imaginary time
	tau = np.random.random()

	N_flats = len(worldlines[i])
	# Random kink index

	r = int(np.random.random()*N_flats)
	k = 0
	for j in range(r):
		k = k + 1
		curr_kink = worldlines[i][j]

	# Randomly choose
	# Choose insertion index s.t list is sorted
	# N_flats = len(worldlines[i])
	# for k in range(N_flats):
	# 	tau_min = worldlines[i][k][0]
	# 	if tau_min > tau: break

	# These don't affect sorting
	n_i = -1
	src = -1
	dest = -1

	# Do insertion on chosen site
	worldlines[i].insert(k,[tau,n_i,(src,dest)])

start2 = time.time()
# Do many insertions
for iteration in range(iterations):

	# Randomly choose a lattice site
	i = int(np.random.random()*L)

	# Randomly choose an imaginary time
	tau = np.random.random()

	N_flats = len(worldlines[i])
	# Random kink index

	r = int(np.random.random()*N_flats)
	k = 0
	for j in range(r):
		k = k + 1

	# Randomly choose
	# Choose insertion index s.t list is sorted
	# N_flats = len(worldlines[i])
	# for k in range(N_flats):
	# 	tau_min = worldlines[i][k][0]
	# 	if tau_min > tau: break

	# These don't affect sorting
	n_i = -1
	src = -1
	dest = -1

	# Do insertion on chosen site
	worldlines[i].insert(k,[tau,n_i,(src,dest)])
	del worldlines[i][k]

end2 = time.time()
print("%d insertions done in %.2f seconds."%(iterations,(end2-start2)))



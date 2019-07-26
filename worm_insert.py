# Insert a worm (head and tail) on a random lattice site
# and imaginary time range

import numpy as np
import bisect

def worm_insert(data_struct, beta):
    'Accept/reject worm head AND tail insertion'

    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Randomly select a lattice site i on which to insert a worm
    i = np.random.randint(L)

    # Determine all flat intervals on i
    if len(data_struct[i]) = 1: # Completely flat worldline case
        tau_min = 0
        tau_max = beta

    flats = [] # Stores the flat intervals
    n_min = data_struct[i][0][1] # Initial number of particles in i
    nkinks = len(data_struct[i])
    for k in range(nkinks-1):
        if data_struct[i][k][1] == data_struct[i][k+1][1]:
            flats.append([tau_min,tau_max,n])

    # Randomly select imaginary time in [0,beta) at which the worm tail will be inserted
    tau_jump = beta*np.random.random()

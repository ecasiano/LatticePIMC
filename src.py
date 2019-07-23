# Functions to be used in main file (lattice_pimc.py)
import numpy as np
import bisect

def random_boson_config(L,N):
    '''Generates a random configuration of N bosons in a 1D lattice of size L'''

    psi = np.zeros(L,dtype=int) # Stores the random configuration of bosons
    for i in range(N):
        r = np.random.randint(L)
        psi[r] += 1

    return psi

'----------------------------------------------------------------------------------'

def particle_jump(data_struct, beta):
    'Proposes a particle jump betweem neighboring sites and accepts/rejects via Metropolis'
    'data_struct = [[(tau, N after tau, id),(tau, N after tau, id)],[(),(),()],...]'

    # Number of lattice sites
    L = len(data_struct)

    # Count the number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # ilist[taulist[list[(tau,N,jump_dir)]]]

    print("TOTAL N: ",N)

    # Randomly select a lattice site i from which particle will jump
    i = np.random.randint(L)

    # Randomly select imaginary time in [0,beta) at which particle will jump
    tau_jump = beta*np.random.random()

    # Reject update if there is no particle on i at time tau
    if data_struct[i][-1][1] == 0 : return None

    # Randomly select whether proposed insert kink update is to right or left
    r = np.random.random()
    if r < 0.5: jump_dir = -1 # 'left'
    else: jump_dir = 1 # 'right'

    # Set the index of the site where the particle on i will jump to
    j = i + jump_dir
    if j == L : j = 0      # Periodic Boundary Conditions
    if j == -1 : j = L-1

    # Metropolis sampling
    weight_jump = 0.1
    r = np.random.random()
    if r < weight_jump:
        # Accept particle jump from sites i -> j
        ni = data_struct[i][-1][1] - 1 # particles on i after inserting kink
        nj = data_struct[j][-1][1] + 1 # particles on j after inserting kink
        taui = bisect.bisect_left(data_struct[i],[tau_jump,ni,(i,j)])
        tauj = bisect.bisect_left(data_struct[j],[tau_jump,nj,(i,j)])
        data_struct[i].insert(taui,[tau_jump,ni,(i,j)])
        data_struct[j].insert(tauj,[tau_jump,nj,(i,j)])

    else: "Reject update"

    return None

'----------------------------------------------------------------------------------'

def particle_jump_backup(data_struct, beta):
    'Proposes a particle jump betweem neighboring sites and accepts/rejects via Metropolis'
    'data_struct = [[(tau, N after tau, id),(tau, N after tau, id)],[(),(),()],...]'

    # Number of lattice sites
    L = len(data_struct)

    # Count the number of total particles in the lattice
    N = 0
    for site in range(L):
        N += data_struct[site][0][1] # ilist[taulist[list[(tau,N,jump_dir)]]]

    # Randomly select a lattice site i from which particle will jump
    i = np.random.randint(L)

    # Randomly select imaginary time in [0,beta) at which particle will jump
    tau_jump = beta*np.random.random()

    # Reject update if there is no particle on i at time tau
    if data_struct[i][-1][1] == 0 : return None

    # Randomly select whether proposed insert kink update is to right or left
    r = np.random.random()
    if r < 0.5: jump_dir = -1 # 'left'
    else: jump_dir = 1 # 'right'

    # Set the index of the site where the particle on i will jump to
    j = i + jump_dir
    if j == L : j = 0      # Periodic Boundary Conditions
    if j == -1 : j = L-1

    # Metropolis sampling
    weight = 0.6
    r = np.random.random()
    if r < 0.5:
        # Accept particle jump from sites i -> j
        ni = data_struct[i][-1][1] - 1 # particles on i after inserting kink
        nj = data_struct[j][-1][1] + 1 # particles on j after inserting kink
        data_struct[i].append([tau_jump,ni,(i,j)]) # (i,j) = ()source site, dest site)
        data_struct[j].append([tau_jump,nj,(i,j)])


    else: "Reject update"

    return None

'----------------------------------------------------------------------------------'


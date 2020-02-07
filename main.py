# Calculate BoseHubbard ground state energy at fixed U/t, but varying beta

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

# -------- Set command line arguments -------- #

# Positional arguments
parser = argparse.ArgumentParser()

parser.add_argument("L",help="Total number of lattice sites",type=int)
parser.add_argument("N",help="Total number of bosons",type=int)
parser.add_argument("U",help="Interaction potential",type=float)   
parser.add_argument("mu",help="Chemical potential",type=float)

# Optional arguments
parser.add_argument("--t",help="Hopping parameter (default: 1.0)",
                    type=float,metavar='\b')
parser.add_argument("--eta",help="Worm end fugacity (default: 1/sqrt(L*beta)",
                    type=float,metavar='\b')
parser.add_argument("--beta",help="Thermodynamic beta 1/(K_B*T) (default: 1.0)",
                    type=float,metavar='\b')
parser.add_argument("--M",help="Number of Monte Carlo steps (default: 1E+05)",
                    type=int,metavar='\b') 
parser.add_argument("--canonical",help="Statistical ensemble (Default: Grand Canonical)",
                    action='store_true') 

# Parse arguments
args = parser.parse_args()

#Positional arguments
L = args.L
N = args.N
U = args.U
mu = args.mu

#Optional arguments (done this way b.c of some argparse bug) 
t = 1.0 if not(args.t) else args.t
beta = 1.0 if not(args.beta) else args.beta
eta = 1/np.sqrt(L*beta) if not(args.eta) else args.eta
M = int(1E+05) if not(args.M) else args.M
canonical = False if not(args.canonical) else True

# ---------------- Lattice PIMC ---------------- #

# Initialize Fock state
alpha = pimc.random_boson_config(L,N)
alpha = [1]*L

# Create worldline data structure
data_struct = pimc.create_data_struct(alpha,L)

# List that will contain site and kink indices of worm head and tail
head_loc = [] # 
tail_loc = []

# Total particle number tracker
N_tracker = [N] # Tracks the total number of particle to enforce Canonical simulations

# Set the window at which kinetic energy will count kinks
dtau = 0.1*beta # i.e count kinks at beta/2 +- dtau

### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####
### CLEAN FROM HERE WHEN YOU COME BACK!!!!!!#####

# Initialize values to be measured
Z_ctr = 0 # Count Z or diagonal configurations (i.e, no worm ends)
diagonal_list = []
kinetic_list = []
N_list = [] # average total particles 
occ_list = [] # average particle occupation
E_N_list = [] # Fixed total particle energies
E_canonical_list = [] # To save energies only for N space configurations

# Counters for acceptance of each move
insert_worm_data = [0,0] # [accepted,proposed]
delete_worm_data = [0,0]

insert_anti_data = [0,0]
delete_anti_data = [0,0]

advance_head_data = [0,0]
recede_head_data = [0,0]

advance_tail_data = [0,0]
recede_tail_data = [0,0]

insertZero_worm_data = [0,0]
deleteZero_worm_data = [0,0]

insertZero_anti_data = [0,0]
deleteZero_anti_data = [0,0]

insertBeta_worm_data = [0,0]
deleteBeta_worm_data = [0,0]

insertBeta_anti_data = [0,0]
deleteBeta_anti_data = [0,0]

ikbh_data = [0,0]
dkbh_data = [0,0]

ikah_data = [0,0]
dkah_data = [0,0]

ikbt_data = [0,0]
dkbt_data = [0,0]

ikat_data = [0,0]
dkat_data = [0,0]


# Set the number of times the set of updates will be attempted
for m in range(M):
    
    # assign a label to each update
    labels = list(range(15)) # There 15 functions
    shuffle(labels)
    
    # At every mc step, try EVERY update in random order
    for label in labels:   
        
        # Non-Spaceshift moves
        if label == 0:
            pimc.worm_insert(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical, N_tracker,insert_worm_data,insert_anti_data)

        elif label == 1:
            pimc.worm_delete(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical, N_tracker,delete_worm_data,delete_anti_data)

        elif label == 2:
            pimc.worm_timeshift(data_struct,beta,head_loc,tail_loc,U,mu,L,N,canonical, N_tracker,advance_head_data,recede_head_data,advance_tail_data,recede_tail_data)

        elif label == 3:
            pimc.insertZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical, N_tracker,insertZero_worm_data,insertZero_anti_data)

        elif label == 4:
            pimc.deleteZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical, N_tracker,deleteZero_worm_data,deleteZero_anti_data)

        elif label == 5:
            pimc.insertBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical, N_tracker,insertBeta_worm_data,insertBeta_anti_data)

        elif label == 6:
            pimc.deleteBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical, N_tracker,deleteBeta_worm_data,deleteBeta_anti_data)

        # Spaceshift moves   
        elif label == 7:
            pimc.insert_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,ikbh_data)  

        elif label == 8:
            pimc.delete_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,dkbh_data) 

        elif label == 9:
            pimc.insert_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,ikah_data)   

        elif label == 10:
            pimc.delete_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,dkah_data)

        elif label == 11:
            pimc.insert_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,ikbt_data)  

        elif label == 12:
            pimc.delete_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,dkbt_data) 

        elif label == 13:
            pimc.insert_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,ikat_data)   

        else:
            pimc.delete_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical, N_tracker,dkat_data)
            
pimc.view_worldlines(data_struct,beta,figure_name='main_worldline.pdf')


# Set what values to ignore due to equilibration
#mc_fraction = 0
start = int(len(diagonal_list)*0.10)
start = int(len(diagonal_list)*0.10)
#start = 100

diagonal = np.mean(diagonal_list[start:])
kinetic = np.mean(kinetic_list[start:])
N_mean = np.mean(N_list[start:])
occ = np.mean(occ_list,axis=0)
print(Z_ctr)
print(len(diagonal_list))
print(len(diagonal_list)-start)


if canonical:
    print("\nEnsemble: Canonical\n")
else:
    print("\nEnsemble: Grand Canonical")
print("-------- Ground State Energy (E/t) --------")
print("E/t: %.8f "%((diagonal+kinetic)/t))
print("-------- Average N --------")
print("<N>: %.8f"%(N_mean))
print("-------- Average occupation --------")
print("<n_i>:",occ)
print("-------- Z-configuration fraction --------")
print("Z-fraction: %.2f%% (%d/%d) "%(Z_ctr/ M*100,Z_ctr, M))

kinetic_list = np.array(kinetic_list)
with open("kinetic_%i_%i_%.4f_%.4f_%.4f_%.4f_%.4f_%i.dat"%(L,N,U,mu,t,eta,beta, M),"w+") as data:
    np.savetxt(data,kinetic_list,delimiter=",",fmt="%.16f",header="MC_step <E> // BH Parameters: L=%d,N=%d,U=%.8f,mu=%.8f,t=%.4f,eta=%.8f,beta=%.4f, M=%i"%(L,N,U,mu,t,eta,beta,M))
    
diagonal_list = np.array(diagonal_list)
with open("diagonal_%i_%i_%.4f_%.4f_%.4f_%.4f_%.4f_%i.dat"%(L,N,U,mu,t,eta,beta, M),"w+") as data:
    np.savetxt(data,diagonal_list,delimiter=",",fmt="%.16f",header="MC_step <E> // BH Parameters: L=%d,N=%d,U=%.8f,mu=%.8f,t=%.4f,eta=%.8f,beta=%.4f, M=%i"%(L,N,U,mu,t,eta,beta,M))    
# Calculate BoseHubbard ground state energy at fixed U/t, but varying beta

import pimc # custom module
import numpy as np
import fastrand
import argparse
import time
import os
import math

# -------- Set command line arguments -------- #

# Positional arguments
parser = argparse.ArgumentParser()

parser.add_argument("L",help="Number of sites per dimension",type=int)
parser.add_argument("N",help="Total number of bosons",type=int)
parser.add_argument("U",help="Interaction potential",type=float)   
parser.add_argument("mu",help="Chemical potential",type=float)

# Optional arguments
parser.add_argument("--t",help="Hopping parameter (default: 1.0)",
                    type=float,metavar='\b')
parser.add_argument("--eta",help="Worm end fugacity (default: 1/sqrt(<N_flats>)",
                    type=float,metavar='\b')
parser.add_argument("--beta",help="Thermodynamic beta 1/(K_B*T) (default: 1.0)",
                    type=float,metavar='\b')
parser.add_argument("--n-slices",help="Measurement window (default: 43)",
                    type=int,metavar='\b')
parser.add_argument("--M",help="Number of Monte Carlo steps (default: 1E+05)",
                    type=int,metavar='\b') 
parser.add_argument("--M-pre",help="Number of Calibration steps (default: 5E+04)",
                    type=int,metavar='\b') 
parser.add_argument("--canonical",help="Statistical ensemble (Default: Grand Canonical)",
                    action='store_true') 
parser.add_argument("--bin-size",help="Number of measurements at each bin (defaul: 10)",
                    type=int,metavar='\b')  
parser.add_argument("--get-fock-state",help="Measure Fock state at beta (Default: False)",
                    action='store_true') 
parser.add_argument("--rseed",help="Set the random number generator's seed (default: 0)",
                    type=int,metavar='\b') 
parser.add_argument("--mfreq",help="Measurements made every other mfreq*L*beta steps (default: 2000)",
                    type=int,metavar='\b')
parser.add_argument("--D",help="Lattice dimension (default: 1)",
                    type=int,metavar='\b')
parser.add_argument("--M-equil",help="Number of Monte Carlo steps given to system to equilibrate",
                    type=int,metavar='\b')

# Parse arguments
args = parser.parse_args()

#Positional arguments
L = args.L
N = args.N
U = args.U
mu = args.mu

# Optional arguments (done this way b.c of some argparse bug) 
t = 1.0 if not(args.t) else args.t
beta = 1.0 if not(args.beta) else args.beta
M = int(1E+05) if not(args.M) else args.M
canonical = False if not(args.canonical) else True
n_slices = 43 if not(args.n_slices) else args.n_slices
M_pre = int(5E+05) if not(args.M_pre) else args.M_pre
bin_size = 10 if not(args.bin_size) else args.bin_size
get_fock_state = False if not(args.get_fock_state) else True
rseed = int(0) if not(args.rseed) else args.rseed
mfreq = int(2000) if not(args.mfreq) else args.mfreq
D = int(1) if not(args.D) else args.D
eta = 1/(L**D*beta)**1.75 if not(args.eta) else args.eta
M_equil = 100000 if not(args.M_equil) else args.M_equil

# Set the random seed
np.random.seed(rseed)
fastrand.pcg32_seed(rseed)

# Build the adjacency matrix
A = pimc.build_adjacency_matrix(L,D,'pbc')

start = time.time()

# Initialize Fock state
alpha = pimc.random_boson_config(L,D,N)

# Create worldline data structure
data_struct = pimc.create_data_struct(alpha,L,D)

# Worm head and tail SITE & FLAT indices
head_loc = []
tail_loc = []

# Initialize trackers
N_tracker = [N]
N_flats_tracker = [L]
N_zero = [N]
N_beta = [N]

# ACCEPT/REJECT counter for each move
dummy_data = [0,0]

# Randomly propose every update M_pre*0.2 times (equilibration)
for m in range(M):

    #Non-Spaceshift moves                

    pimc.worm_insert(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,
                    N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data) 
    pimc.worm_delete(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,
                    N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data)

#     pimc.worm_timeshift(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data,
#                        dummy_data,dummy_data) 
#     pimc.insertZero(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data)
#     pimc.deleteZero(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data)
#     pimc.insertBeta(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data) 
#     pimc.deleteBeta(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data,dummy_data) 

#     # Spaceshi moves   
#     pimc.insert_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)   
#     pimc.delete_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)  
#     pimc.insert_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)     
#     pimc.delete_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)  
#     pimc.insert_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)    
#     pimc.delete_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)   
#     pimc.insert_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)     

#     pimc.delete_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,
#                     N_tracker,N_flats_tracker,A,N_zero,N_beta,dummy_data)  

end = time.time()

print(f"Elapsed time: {end-start} seconds.")
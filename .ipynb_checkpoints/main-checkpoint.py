# Calculate BoseHubbard ground state energy at fixed U/t, but varying beta

import pimc # custom module
import numpy as np
import matplotlib.pyplot as plt
import importlib
import argparse
importlib.reload(pimc)
import random
import datetime
import time

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
parser.add_argument("--n-slices",help="Measurement window",
                    type=int,metavar='\b')
parser.add_argument("--M",help="Number of Monte Carlo steps (default: 1E+05)",
                    type=int,metavar='\b') 
parser.add_argument("--M-pre",help="Number of Calibration steps (default: 5E+04)",
                    type=int,metavar='\b') 
parser.add_argument("--canonical",help="Statistical ensemble (Default: Grand Canonical)",
                    action='store_true') 
parser.add_argument("--bin-size",help="Number of measurements at each bin (defaul: 10)",
                    type=int,metavar='\b') 
parser.add_argument("--no-energies",help="Measure diagonal and kinetic energies (Default: True)",
                    action='store_true') 
parser.add_argument("--get-fock-state",help="Measure Fock state at beta (Default: False)",
                    action='store_true') 
parser.add_argument("--rseed",help="Set the random number generator's seed (default: 0)",
                    type=int,metavar='\b') 
parser.add_argument("--mfreq",help="Measurements made every other mfreq*L*beta steps (default: 2000)",
                    type=int,metavar='\b')
parser.add_argument("--D",help="Lattice dimension (default: 1)",
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
no_energies = False if not(args.no_energies) else True
get_fock_state = False if not(args.get_fock_state) else True
rseed = int(0) if not(args.rseed) else args.rseed
mfreq = int(2000) if not(args.mfreq) else args.mfreq
D = int(1) if not(args.D) else args.D
eta = 1/(L**D*beta) if not(args.eta) else args.eta

# Set the random seed
np.random.seed(rseed)

# Initialize Fock state
alpha = pimc.random_boson_config(L,D,N)

# Create worldline data structure
data_struct = pimc.create_data_struct(alpha,L,D)

# Build the adjacency matrix
A = pimc.build_adjacency_matrix(L,D,'pbc')

# List that will contain site and kink indices of worm head and tail
head_loc = []
tail_loc = []

# Trackers
N_tracker = [N]         # Total particles
N_flats_tracker = [L]   # Total flat regions
N_zero = [N]
N_beta = [N]

# ---------------- Lattice PIMC ---------------- #

start = time.time()

# Open files that will save data
if canonical: # kinetic
        
    if not(no_energies):
        kinetic_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_canK.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
        diagonal_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_canV.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")

print("LatticePIMC started...\n")

# Initialize values to be measured
bin_ctr = 0
tau_slices = np.linspace(0,beta,n_slices)[1:-1][::2] # slices were observables are measured
tr_kinetic_list = np.zeros_like(tau_slices)
tr_diagonal_list = np.zeros_like(tau_slices)
N_list = [] 

# Count measurements and measurement attempts
measurements = [0,0] # [made,attempted]

# Redefine M to be an actual Monte Carlo step (a sweep)
M = M*L**D*beta

# Pre-allocate the random move labels
labels = (np.random.random(int(M))*15).astype(np.ushort)

# Pre-allocate random site indices for insert type moves
insertion_sites = (np.random.random(int(M))*L**D).astype(np.uint32)
print(insertion_sites[:10])

# Pool of worm algorithm updates
pool = {0: pimc.worm_insert,
    1: pimc.worm_delete,
    2: pimc.worm_timeshift,
    4: pimc.insertZero,
    3: pimc.deleteZero,
    5: pimc.insertBeta,
    6: pimc.deleteBeta,
    7: pimc.insert_kink_before_head,
    8: pimc.delete_kink_before_head,
    9: pimc.insert_kink_after_head,
    10: pimc.delete_kink_after_head,
    11: pimc.insert_kink_before_tail,
    12: pimc.delete_kink_before_tail,
    13: pimc.insert_kink_after_tail,
    14: pimc.delete_kink_after_tail,
   }

# Randomly an update M times
for m in range(int(M)):
            
    # Set the move label
    label = labels[m]
    
    # Set the random insertion site
    insertion_site = insertion_sites[m]  
    
    # Pool of worm algorithm updates
    pool[label](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta,insertion_site)
                  
#     # After 25% equilibration, measure every L*beta steps
#     if m%(int(mfreq*L*beta))==0:
#         try_measure=True
#     else:
#         try_measure=False
            
#     if try_measure and m > int(M*0.25):  
        
#         # Add to MEASUREMENTS ATTEMPTS counter
#         measurements[1] += 1
        
#         # Make measurement if no worm ends present
#         if not(pimc.check_worm(head_loc,tail_loc)):
    
#             if canonical: # Canonical simulation
     
#                 #print(round(N_tracker[0],10),N_tracker[0])
#                 if round(N_tracker[0],10)==N:
                    
#                     bin_ctr += 1

#                     # Add to MEASUREMENTS MADE counter
#                     measurements[0] += 1
                    
#                     if not(no_energies):
                        
#                         # Energies, but measured at different tau slices (time resolved)
#                         tr_kinetic,tr_diagonal = pimc.tau_resolved_energy(data_struct,beta,n_slices,U,mu,t,L,D)
#                         tr_kinetic_list += tr_kinetic # cumulative sum
#                         tr_diagonal_list += tr_diagonal # cumulative sum
                        
#                         # Take the binned average of the time resolved energies
#                         if bin_ctr == bin_size:

#                             tr_kinetic_list /= bin_size
#                             tr_diagonal_list /= bin_size

#                             # Write to file(s)
#                             data_len = len(tr_kinetic_list)
#                             np.savetxt(kinetic_file,np.reshape(tr_kinetic_list,(1,data_len)),fmt="%.16f",delimiter=" ")
#                             np.savetxt(diagonal_file,np.reshape(tr_diagonal_list,(1,data_len)),fmt="%.16f",delimiter=" ")

#                             # Reset bin_ctr and time resolved energies arrays
#                             bin_ctr = 0
#                             tr_kinetic_list *= 0
#                             tr_diagonal_list *= 0
                            

#                 else: # Worldline doesn't have target particle number
#                     pass
                
#             else: # Grand canonical
#             	pass
#                 # # Energies
#                 # kinetic,diagonal = pimc.bh_egs(data_struct,beta,dtau,U,mu,t,L,tau_slice)
#                 # kinetic_list.append(kinetic)
#                 # diagonal_list.append(diagonal)

#                 # # Total number of particles in worldline configuration
#                 # N_list.append(N_tracker[0])
                
#         else: # There's a worm in the worldline configuration
#             pass
        
#     else: # We  only measure every L*beta steps
#         pass
                
# # Close the data files
# if not(no_energies):
#     kinetic_file.close()
#     diagonal_file.close()
# if get_fock_state:
#     fock_state_file.close()

# # Save diagonal fraction obtained from main loop
# Z_frac = 100*measurements[0]/measurements[1]
# print("100%%")
# print("\nLattice PIMC done.\n")

end = time.time()

print("Time elapsed: %.2f seconds"%(end-start))

# # ---------------- Print acceptance ratios ---------------- #

# if canonical:
#     print("\nEnsemble: Canonical\n")
# else:
#     print("\nEnsemble: Grand Canonical\n")
# print("-------- Z-configuration fraction --------")
# print("Z-fraction: %.2f%% (%d/%d) "%(100*measurements[0]/measurements[1],measurements[0],measurements[1]))
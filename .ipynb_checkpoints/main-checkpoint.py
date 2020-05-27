# Calculate BoseHubbard ground state energy at fixed U/t, but varying beta

import pimc # custom module
import numpy as np
import matplotlib.pyplot as plt
import importlib
import argparse
importlib.reload(pimc)
import time
import fastrand

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

# Pool of worm algorithm updates
pool = [ pimc.worm_insert, # 0
         pimc.worm_delete,
         pimc.worm_timeshift,
         pimc.insertZero, # 3
         pimc.deleteZero,
         pimc.insertBeta, # 5
         pimc.deleteBeta,
         pimc.insert_kink_before_head,
         pimc.delete_kink_before_head,
         pimc.insert_kink_after_head,
         pimc.delete_kink_after_head,
         pimc.insert_kink_before_tail,
         pimc.delete_kink_before_tail,
         pimc.insert_kink_after_tail,
         pimc.delete_kink_after_tail ]

# Initialize Fock state
alpha = pimc.random_boson_config(L,D,N)

# Create worldline data structure
data_struct = pimc.create_data_struct(alpha,L,D)

# Build the adjacency matrix
A = pimc.build_adjacency_matrix(L,D,'pbc')

# ---------------- Pre-M_equil (determine mu) ---------------- #

# ---------------- Lattice PIMC ---------------- #
    
start = time.time()

# Open files that will save data        
if canonical:
    kinetic_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_can_K.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
    diagonal_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_can_V.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
    N_file  = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_can_N.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
else:
    kinetic_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_K.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
    diagonal_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_V.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
    N_file  = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_N.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")

# Data structure containing the worldlines
data_struct = pimc.create_data_struct(alpha,L,D)

# List that will contain site and kink indices of worm head and tail
head_loc = []
tail_loc = []

# Trackers
N_tracker = [N]         # Total particles
N_flats_tracker = [L]   # Total flat regions
N_zero = [N]
N_beta = [N]

print("Equilibration started...")

# Redefine M to be an actual Monte Carlo step (a sweep)
M *= int(L**D*beta)
M_equil *= int(L**D*beta)

# M_equil loop
for m in range(M_equil): 
    
    # Propose move from pool of worm algorithm updates
    pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta)
    
print("Equilibration done...\n")
    
# ---------------- Lattice PIMC ---------------- #

print("LatticePIMC started...")

# Initialize values to be measured
bin_ctr = 0
tau_slices = np.linspace(0,beta,n_slices)[1:-1][::2] # slices were observables are measured
tr_kinetic_list = np.zeros_like(tau_slices)
tr_diagonal_list = np.zeros_like(tau_slices)
N_list = [] 

# Length of energies arrays (needed for reshape later down)
data_len = len(tr_kinetic_list)

# Count measurements and measurement attempts
measurements = [0,0] # [made,attempted]

# Observables
N_mean = 0

# Other quantities to measure
Z_sector_ctr = 0 # Configurations in Z-sector
N_sector_ctr = 0 # Configurations in N-sector

# Measure every other mfreq sweeps
skip_ctr = int(mfreq*L**D*beta)
try_measurement = True

# Total particle number bins
N_data = []

# Randomly an update M times
for m in range(M): 
    
    # Pool of worm algorithm updates
    pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta) 
  
    # When counter reaches zero, the next Z-configuration that occurs will be measured 
    if skip_ctr <= 0:
        try_measurement = True
    else:
        #try_measurement = False
        skip_ctr -= 1 
            
    if try_measurement:  
        
        # Add to measurement ATTEMPTS counter
        measurements[1] += 1
        
        # Make measurement if no worm ends present
        if not(pimc.check_worm(head_loc,tail_loc)):  
            
            N_data.append(round(N_tracker[0]))

            # Update diagonal fraction counter
            Z_sector_ctr += 1
            
            # Update N accumulator
            N_mean += N_tracker[0]
            
            if canonical:
                
                #print(N_tracker[0])
                # Check if in correct N-sector for canonical simulations
                if N-N_tracker[0] > -1.0E-12 and N-N_tracker[0] < +1.0E-12:
                #if round(N_tracker[0])==4.0:
                #if True:
        
                    #print(round(N_tracker[0]))
                   
                    # Update the N-sector counter
                    N_sector_ctr += 1

                    # Measurement just performed, will not measure again in at least mfreq sweeps
                    skip_ctr = int(mfreq*L**D*beta)
                    try_measurement = False

                    # Binning average counter
                    bin_ctr += 1

                    # Add to MEASUREMENTS MADE counter
                    measurements[0] += 1

                    # Energies, but measured at different tau slices (time resolved)
                    tr_kinetic,tr_diagonal = pimc.tau_resolved_energy(data_struct,beta,n_slices,U,mu,t,L,D)
                    tr_kinetic_list += tr_kinetic # cumulative sum
                    tr_diagonal_list += tr_diagonal # cumulative sum

                    # Take the binned average of the time resolved energies
                    if bin_ctr == bin_size:

                        tr_kinetic_list /= bin_size
                        tr_diagonal_list /= bin_size

                        # Write to file(s)
                        np.savetxt(kinetic_file,np.reshape(tr_kinetic_list,(1,data_len)),fmt="%.16f",delimiter=" ")
                        np.savetxt(diagonal_file,np.reshape(tr_diagonal_list,(1,data_len)),fmt="%.16f",delimiter=" ")

                        # Reset bin_ctr and time resolved energies arrays
                        bin_ctr = 0
                        tr_kinetic_list *= 0
                        tr_diagonal_list *= 0
                    
                else: # Worldline doesn't have target particle number
                    #print(N_tracker[0])
                    pass
                
            else: # Grand canonical simulation
                
                # Measurement just performed, will not measure again in at least mfreq sweeps
                skip_ctr = int(mfreq*L**D*beta)
                try_measurement = False

                # Binning average counter
                bin_ctr += 1

                # Add to MEASUREMENTS MADE counter
                measurements[0] += 1

                # Energies, but measured at different tau slices (time resolved)
                tr_kinetic,tr_diagonal = pimc.tau_resolved_energy(data_struct,beta,n_slices,U,mu,t,L,D)
                tr_kinetic_list += tr_kinetic # cumulative sum
                tr_diagonal_list += tr_diagonal # cumulative sum

                # Take the binned average of the time resolved energies
                if bin_ctr == bin_size:

                    tr_kinetic_list /= bin_size
                    tr_diagonal_list /= bin_size

                    # Write to file(s)
                    np.savetxt(kinetic_file,np.reshape(tr_kinetic_list,(1,data_len)),fmt="%.16f",delimiter=" ")
                    np.savetxt(diagonal_file,np.reshape(tr_diagonal_list,(1,data_len)),fmt="%.16f",delimiter=" ")

                    # Reset bin_ctr and time resolved energies arrays
                    bin_ctr = 0
                    tr_kinetic_list *= 0
                    tr_diagonal_list *= 0
                    
                    # Array of measured total particle number. Needed for mu determination.
                    N_file.write(str(N_tracker[0])+'\n')

        else: # Not a diagonal configuration. There's a worm end.
            pass

    else: # We  only measure every L**D*beta iterations
        pass

# Close the data files
kinetic_file.close()
diagonal_file.close()
N_file.close()

# Save diagonal fraction obtained from main loop
print("Lattice PIMC done.\n")

print("<N> = %.12f"%(N_mean/Z_sector_ctr))

end = time.time()

print("Time elapsed: %.2f seconds"%(end-start))

# ---------------- Print acceptance ratios ---------------- #

if canonical:
    print("\nEnsemble: Canonical\n")
else:
    print("\nEnsemble: Grand Canonical\n")
print("-------- Z-configuration fraction --------")
print("Z-fraction: %.2f%% (%d/%d) "%(100*Z_sector_ctr/measurements[1],Z_sector_ctr,measurements[1]))

if canonical:
    print("-------- N-configuration fraction --------")
    print("N-fraction: %.2f%% (%d/%d) "%(100*N_sector_ctr/Z_sector_ctr,N_sector_ctr,Z_sector_ctr))
   
print(f'\nP(3)={N_data.count(3)/Z_sector_ctr} P(4)={N_data.count(4)/Z_sector_ctr} P(5)={N_data.count(5)/Z_sector_ctr} sum[P(N)]={(N_data.count(3)+N_data.count(4)+N_data.count(5))/Z_sector_ctr} {len(N_data)}')
# print("Pre-equilibration started. Determining mu...")

# pre_equilibration = int(M_pre*L**D*beta)
# N_frac = 0
# need_mu = True
# while need_mu and False :
    
#     insertion_site = int(np.random.random()*L**D)
    
#     data_struct = pimc.create_data_struct(alpha,L,D)
    
#     # List that will contain site and kink indices of worm head and tail
#     head_loc = []
#     tail_loc = []

#     # Trackers
#     N_tracker = [N]         # Total particles
#     N_flats_tracker = [L]   # Total flat regions
#     N_zero = [N]
#     N_beta = [N]
    
#     measurements = [0,0]

#     # Initialize counter of target N sector and the number Z sector configs
#     N_sector_ctr = 0
#     Z_sector_ctr = 0
#     N_frac = 0
    
#     skip_ctr = int(mfreq*L**D*beta)
#     try_measurement = True
    
#     for m in range(pre_equilibration):

#         # When counter reaches zero, the next Z-configuration that occurs will be measured 
#         if skip_ctr <= 0:
#             try_measurement = True
#         else:
#             #try_measurement = False
#             skip_ctr -= 1 
            
#         if try_measurement:
            
#             # Randomly choose move label and insertion site
#             label = int(np.random.random()*15)
#             if label == 0 or label == 3 or label == 5: 
#                 insertion_site = int(np.random.random()*L**D)

#             # Propose move from pool of worm algorithm updates
#             pool[label](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta)

#             measurements[1] += 1 # measurement attempts

#             # Make measurement if no worm ends present
#             if not(pimc.check_worm(head_loc,tail_loc)): 

#                 Z_sector_ctr += 1

#                 if N-N_tracker[0] > -1.0E-12 and N-N_tracker[0] < 1.0E-12:

#                     N_sector_ctr += 1

#                     # Measurement just performed, will not measure again in at least mfreq sweeps
#                     skip_ctr = int(mfreq*L**D*beta)
#                     try_measurement = False
            
#     # Calculate N-sector and Z-sector fractions
#     N_frac = N_sector_ctr/measurements[1]
#     Z_frac = Z_sector_ctr/measurements[1]
        
#     # Print mu and N_frac to screen
#     print("mu = %.4f | N_frac = %.2f (%d/%d) | Z_frac = %.2f (%d/%d)"%(mu,N_frac,N_sector_ctr,measurements[1],Z_frac,Z_sector_ctr,measurements[1]))
    
#     # If below desired N-sector fraction, increase mu
#     if N_frac < 0.3:
#         need_mu = True
#         mu -= (0.001)
#     else:
#         need_mu = False

# print("Pre-equilibration done...\n")
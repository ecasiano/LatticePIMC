# Calculate BoseHubbard ground state energy at fixed U/t, but varying beta

import pimc # custom module
import numpy as np
import fastrand
import argparse
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

# Worm head and tail SITE & FLAT indices
head_loc = []
tail_loc = []

# Initialize trackers
N_tracker = [N]
N_flats_tracker = [L]
N_zero = [N]
N_beta = [N]

# ---------------- Pre-equilibration 1: mu calibration ---------------- #

print("Pre-equilibration (1/2): Determining mu...")

print("mu | P(N-1) | P(N) | P(N+1)")

M_pre = int(M_pre*L**D*beta)
M_equil = int(M_equil*L**D*beta)

need_eta = True
# Iterate until particle distribution P(N) is peaked at target N
while need_eta:
    
    # Restart the data structure
    data_struct = pimc.create_data_struct(alpha,L,D)
    head_loc = []
    tail_loc = []
    N_tracker = [N]
    N_flats_tracker = [L]
    N_zero = [N]
    N_beta = [N]

    # Equilibrate the system
    for m in range(M_equil):
        
        # Pool of worm algorithm updates
        pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,True,N_tracker,N_flats_tracker,A,N_zero,N_beta)
        
    # Collect data for N-histogram
    N_data = []
    for m in range(M_pre):

        pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,True,N_tracker,N_flats_tracker,A,N_zero,N_beta)

        # Make measurement if no worm ends present
        if not(pimc.check_worm(head_loc,tail_loc)):

            N_data.append(round(N_tracker[0]))
    
    # Promote list of N to array
    N_data = np.array(N_data)
    
    # Build histogram of total particle number
    bins = np.linspace(N_data.min(),N_data.max()+1,len(np.unique(N_data))+1)
    N_histogram = np.histogram(N_data,bins=bins)
    P_N = (N_histogram[0]/sum(N_histogram[0])) # Normalize the distribution
    N_bins = np.array(bins).astype(int)
    
#     print(f"{mu:.4f}:{list(zip(N_bins,P_N))},Max: {max(list(zip(N_bins,P_N)))}")
#     print(mu,N_bins,P_N)
    print(f"{mu}|{P_N[0]}|{P_N[1]}|{P_N[2]}")

#     if 
    # Determine mu via Eq. 15 in: https://arxiv.org/pdf/1312.6177.pdf
    N_target = N # target number of particles
    mu_gc = mu   
    
    N_idx = np.where(N_bins==N_target)[0][0] # Index of bin corresponding to N_target
    
    mu_right = mu_gc - (1/beta)*np.log(P_N[N_idx+1]/P_N[N_idx]) # Target chemical potential
    mu_left = mu_gc - (1/beta)*np.log(P_N[N_idx]/P_N[N_idx-1]) # Target chemical potential
    mu = (mu_left + mu_right)/2

    
        



# ---------------- Pre-equilibration 2: eta calibration  ---------------- #

# ---------------- Equilibration ---------------- #

# ---------------- Main Loop ---------------- #


# print("Pre-equilibration started. Determining mu...")

# if canonical:
#     print("eta |  mu  |  Z_frac  | P(N)")
# else:
#     print("eta |  mu  |  Z_frac  | <N>")

# # Do M_pre sweeps for various eta and mu, until desired Z_frac,N_frac are obtained
# need_eta = True
# need_mu = True
# #while need_eta or need_mu:
# while False:
    
#     # Reset worldlines 
#     data_struct = pimc.create_data_struct(alpha,L,D)
    
#     # Reset worm head and tail location containers
#     head_loc = []
#     tail_loc = []

#     # Reset the trackers
#     N_tracker = [N]         # Total particles
#     N_flats_tracker = [L]   # Total flat regions
#     N_zero = [N]
#     N_beta = [N]
    
#     # Reset counters of measurements made/attempted
#     measurements = [0,0]

#     # Reset N-sector,Z-sector counters
#     N_sector_ctr = 0
#     Z_sector_ctr = 0
#     N_frac = 0
#     Z_frac = 0
    
#     # Count sweeps since last measurement occured
#     skip_ctr = 0        
#     try_measurement = True
    
#     # Total particle number sectors; for P(N)
#     N_data = []
    
#     # Average N (for grand canonical)
#     N_mean = 0
    
#     # Pre equilibration also needs to be "thermalized"
#     for m in range(M_equil):
        
#         # Pool of worm algorithm updates
#         pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta) 
           
#     for m in range(M_pre):

#         # Pool of worm algorithm updates
#         pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta) 
        
#         # In this iteration, measurement still correlated. Won't try to measure.
#         if skip_ctr%(mfreq*L**D*beta)!=0:
#            skip_ctr += 1

#         # Only attempt measurements mfreq sweeps after the last Z-sector visit
#         else:          

#             # Add to measurement ATTEMPTS counter
#             measurements[1] += 1

#             # Make measurement if no worm ends present
#             if not(pimc.check_worm(head_loc,tail_loc)):  

#                 # Update diagonal fraction counter
#                 Z_sector_ctr += 1

#                 # Update N accumulator
#                 N_mean += N_tracker[0]

#                 # Store N values to build P(N)
#                 N_data.append(round(N_tracker[0]))

#                 if canonical:

#                     # Check if in correct N-sector for canonical simulations
#                     if N-N_tracker[0] > -1.0E-12 and N-N_tracker[0] < +1.0E-12: 

#                         # Reset skipped measurement counter
#                         skip_ctr = 1  

#                         # Update the N-sector counter
#                         N_sector_ctr += 1

#                         # Add to MEASUREMENTS MADE counter
#                         measurements[0] += 1

#                     else: # Worldline doesn't have target particle number
#                         pass

#                 else: # Grand canonical simulation

#                     # Reset skipped measurement counter
#                     skip_ctr = 1  

#                     # Binning average counter
#                     bin_ctr += 1

#                     # Add to MEASUREMENTS MADE counter
#                     measurements[0] += 1

#             else: # Not a diagonal configuration. There's a worm end.
#                 pass
            
#     # Calculate Z-sector and N-sector fractions
#     Z_frac = Z_sector_ctr/measurements[1]
#     N_frac = N_sector_ctr/Z_sector_ctr
    
#     # Calculate normalized particle probability distribution P(N)
#     P_N = [N_data.count(n)/Z_sector_ctr for n in [N-1,N,N+1]]
    
#     # Calculate average N
#     N_mean /= Z_sector_ctr
    
#     if canonical:
#         print(f"{eta} | {mu:.4f} | {Z_frac} | {P_N}")
#     else:
#         print(f"{eta} | {mu:.4f} | {Z_frac} | {N_mean}")
        
#     # Decrease eta if Z_frac is too low. Increase otherwise.
#     if Z_frac < 0.25:
#         need_eta = True
#         eta *= 0.9
#     elif Z_frac > 0.55:
#         need_eta = True
#         eta *= 1.1
#     else:
#         need_eta = False

#     # Modify mu until desired N is at least 33% likely
#     if canonical:
#         if P_N[1] < 0.33: # Want target N at least one third of the times
#             need_mu = True
#             if P_N[0] > P_N[2]: # P(N-1) is the largest
#                 mu += 0.0001
#             else: # P(N+1) is the largest
#                 mu -= 0.0001
#         else:
#             need_mu = False
#     else: # In general, not interested in target N in grand canonical simulations
#         need_mu = False
            
# print("Pre-equilibration done...\n")

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

print("Equilibration started...")

# Redefine M to be an actual Monte Carlo step (a sweep)
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
try_measurement = True

# Total particle number sectors; for P(N)
N_data = []

# Count sweeps since last measurement occured
skip_ctr = 0

# Redefine M as number of sweeps
M *= int(L**D*beta)

# Randomly an update M times
for m in range(M): 
#while measurements[0] < int(M/(L**D*beta)):
    
    # Pool of worm algorithm updates
    pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta) 
    
    # Accumulate Z-fracion data
    if not(pimc.check_worm(head_loc,tail_loc)):  
        # Update diagonal fraction counter
        Z_sector_ctr += 1
        
    # In this iteration, measurement still correlated. Won't try to measure.
    if skip_ctr%(mfreq*L**D*beta)!=0:
       skip_ctr += 1
        
    # Only attempt measurements mfreq sweeps after the last Z-sector visit
    else:          
        
        # Add to measurement ATTEMPTS counter
        measurements[1] += 1
        
        # Make measurement if no worm ends present
        if not(pimc.check_worm(head_loc,tail_loc)):  
            
            # Update N accumulator
            N_mean += N_tracker[0]
            
            # Store N values to build P(N)
            N_data.append(round(N_tracker[0]))
            
            if canonical:
                
                # Check if in correct N-sector for canonical simulations
                if N-N_tracker[0] > -1.0E-12 and N-N_tracker[0] < +1.0E-12: 
                    
                    # Reset skipped measurement counter
                    skip_ctr = 1  
                                   
                    # Update the N-sector counter
                    N_sector_ctr += 1

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

                # Reset skipped measurement counter
                skip_ctr = 1  

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
print("Z-fraction: %.2f%% (%d/%d) "%(100*Z_sector_ctr/M,Z_sector_ctr,M))

if canonical:
    print("-------- N-configuration fraction --------")
    print("N-fraction: %.2f%% (%d/%d) "%(100*N_sector_ctr/Z_sector_ctr,N_sector_ctr,Z_sector_ctr))
   
print(f'\nP(3)={N_data.count(3)/Z_sector_ctr} P(4)={N_data.count(4)/Z_sector_ctr} P(5)={N_data.count(5)/Z_sector_ctr} sum[P(N)]={(N_data.count(3)+N_data.count(4)+N_data.count(5))/Z_sector_ctr} {len(N_data)}')
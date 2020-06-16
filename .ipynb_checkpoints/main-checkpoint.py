# Calculate BoseHubbard ground state energy at fixed U/t, but varying beta

import pimc # custom module
import numpy as np
import fastrand
import argparse
import time
import os

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
N_file_pre = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_N_pre.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")
mu_file_pre = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_mu_pre.dat"%(L,N,U,mu,t,beta,M,rseed,D),"w+")

mu_initial = mu

print("Pre-equilibration (1/2): Determining mu...")

print("mu | P(N-1) | P(N) | P(N+1)")

M_pre = int(M_pre*L**D*beta)
M_equil = int(M_equil*L**D*beta)

# Iterate until particle distribution P(N) is peaked at target N
for i in range(2):
    can_flag = [False,False][i]
    while True:

        # Restart the data structure
        data_struct = pimc.create_data_struct(alpha,L,D)
        head_loc = []
        tail_loc = []
        N_tracker = [N]
        N_flats_tracker = [L]
        N_zero = [N]
        N_beta = [N]
        skip_ctr = 0

        # Equilibrate the system
        for m in range(int(M_pre*0.2)):

            # Pool of worm algorithm updates
            pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,N_tracker,N_flats_tracker,A,N_zero,N_beta)

        # Collect data for N-histogram
        N_data = []
        for m in range(M_pre):

            pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,can_flag,N_tracker,N_flats_tracker,A,N_zero,N_beta)

            # In this iteration, measurement still correlated. Won't try to measure.
            if skip_ctr%(mfreq*L**D*beta)!=0:
               skip_ctr += 1

            else:
                
                # Make measurement if no worm ends present
                if not(pimc.check_worm(head_loc,tail_loc)):

                    N_data.append(round(N_tracker[0]))
                    
                    skip_ctr = 1                    

                    # Array of measured total particle number. Needed for mu determination.
                    N_file_pre.write(str(N_tracker[0])+'\n')
                    mu_file_pre.write(str(mu)+'\n')


        # Promote list to numpy array
        N_data = np.array(N_data)

        # Build histogram of total particle number
        N_bins = np.linspace(N_data.min(),N_data.max()+1,len(np.unique(N_data))+1,dtype=np.int32)
        N_histogram = np.histogram(N_data,bins=N_bins)
        P_N = (N_histogram[0]/sum(N_histogram[0])) # Normalize the distribution

        print(f"{mu}|{list(zip(N_bins,P_N))}")

        # Save the index at which the N target bin is
        N_target = N # target number of particles
            
        if N_target in N_bins:
            N_idx = np.where(N_bins==N_target)[0][0] # Index of bin corresponding to N_target
            
            # Stop the loop if the peak is P(N)
            if P_N.max() == P_N[N_idx]: break


            else:
                # Determine mu via Eq. 15 in: https://arxiv.org/pdf/1312.6177.pdf
                mu_gc = mu   

                if ((N_target-1) in N_bins[:-1]) and ((N_target+1) in N_bins[:-1]):
                    mu_right = mu_gc - (1/beta)*np.log(P_N[N_idx+1]/P_N[N_idx])
                    mu_left = mu_gc - (1/beta)*np.log(P_N[N_idx]/P_N[N_idx-1])
                    mu = (mu_left + mu_right)/2 # target mu
                elif N_target+1 in N_bins[:-1]:
                    mu_right = mu_gc - (1/beta)*np.log(P_N[N_idx+1]/P_N[N_idx])
                    mu = mu_right     
                else:                   
                    mu_left = mu_gc - (1/beta)*np.log(P_N[N_idx]/P_N[N_idx-1])
                    mu = mu_left
            
        else:
            peak_idx = np.argmax(P_N)
            N_max = N_bins[peak_idx]
            
            if N_max > N_target:                
                mu -= 1
            else:          
                mu += 1
                
# Rename files using new chemical potential
os.rename("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_N_pre.dat"%(L,N,U,mu_initial,t,beta,M,rseed,D),
         "%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_N_pre.dat"%(L,N,U,mu,t,beta,M,rseed,D))
os.rename("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_mu_pre.dat"%(L,N,U,mu_initial,t,beta,M,rseed,D),
         "%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%i_%iD_gc_mu_pre.dat"%(L,N,U,mu,t,beta,M,rseed,D))

N_file_pre.close()
mu_file_pre.close()
# ---------------- Pre-equilibration 2: eta calibration  ---------------- #
print("\nPre-equilibration (2/2): Determining eta...")

print("mu | P(N-1) | P(N) | P(N+1) | eta | Z-frac ")


while True:
    
    # Restart the data structure
    data_struct = pimc.create_data_struct(alpha,L,D)
    head_loc = []
    tail_loc = []
    N_tracker = [N]
    N_flats_tracker = [L]
    N_zero = [N]
    N_beta = [N]
    skip_ctr = 0
    
    for m in range(int(M_pre*0.2)):
        
        # Pool of worm algorithm updates
        pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta)
    
    Z_frac = 0
    N_data = []
    for m in range(M_pre):
        
        # Pool of worm algorithm updates
        pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta)
        
        # Make measurement if no worm ends present
        if not(pimc.check_worm(head_loc,tail_loc)):

            Z_frac += 1
            
        # In this iteration, measurement still correlated. Won't try to measure.
        if skip_ctr%(mfreq*L**D*beta)!=0:
           skip_ctr += 1

        else:

            # Make measurement if no worm ends present
            if not(pimc.check_worm(head_loc,tail_loc)):

                N_data.append(round(N_tracker[0]))

                skip_ctr = 1
            
    # Calculate diagonal fraction
    Z_frac /= M_pre
    
    # Promote list to numpy array
    N_data = np.array(N_data)

    # Build histogram of total particle number
    N_bins = np.linspace(N_data.min(),N_data.max()+1,len(np.unique(N_data))+1,dtype=np.int32)
    N_histogram = np.histogram(N_data,bins=N_bins)
    P_N = (N_histogram[0]/sum(N_histogram[0])) # Normalize the distribution
    
    print(f"{mu}|{list(zip(N_bins,P_N))}|{eta}|{Z_frac}")
    
    # Modify it if necessary
    if Z_frac > 0.13 and Z_frac < 0.16:
        break
    else:
        if Z_frac < 0.13: 
            eta *= 0.5
        else:
            eta *= 1.5
        
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

print("\nEquilibration started...")

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
N_data = []

# Length of energies arrays (needed for reshape later down)
data_len = len(tr_kinetic_list)

# Count measurements and measurement attempts
measurements = [0,0] # [made,attempted]

# Observables
N_mean = 0

# Other quantities to measure
Z_sector_ctr = 0 # Configurations in Z-sector
N_sector_ctr = 0 # Configurations in N-sector
Z_frac = 0

# Measure every other mfreq sweeps
try_measurement = True

# Total particle number sectors; for P(N)
N_data = []
N_data_len = 0

# Count sweeps since last measurement occured
skip_ctr = 0

M *= int(L**D*beta)
# Randomly an update M times
for m in range(M-M_equil): 
#while measurements[0] < int(M/(L**D*beta)):
    
    # Pool of worm algorithm updates
    pool[fastrand.pcg32bounded(15)](data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,D,N,canonical,N_tracker,N_flats_tracker,A,N_zero,N_beta) 
    
    # Accumulate Z-fracion data
    if not(pimc.check_worm(head_loc,tail_loc)):  
        # Update diagonal fraction accumulator
        Z_frac += 1
        
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
            N_data_len += 1
            
            # Reset skipped measurement counter
            skip_ctr = 1  
            
            if canonical:
                
                # Check if in correct N-sector for canonical simulations
                if N-N_tracker[0] > -1.0E-12 and N-N_tracker[0] < +1.0E-12: 
                                   
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
                        
                        # Array of measured total particle number. Needed for mu determination.
                        N_file.write(str(round(N_tracker[0]))+'\n')
                    
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
                    N_file.write(str(round(N_tracker[0]))+'\n')

        else: # Not a diagonal configuration. There's a worm end.
            pass

# Close the data files
kinetic_file.close()
diagonal_file.close()
N_file.close()

# Promote list to numpy array
N_data = np.array(N_data)

# Build histogram of total particle number
N_bins = np.linspace(N_data.min(),N_data.max()+1,len(np.unique(N_data))+1,dtype=np.int32)
N_histogram = np.histogram(N_data,bins=N_bins)
P_N = (N_histogram[0]/sum(N_histogram[0])) # Normalize the distribution

# Save diagonal fraction obtained from main loop
print("Lattice PIMC done.\n")

print("<N> = %.12f"%(N_mean/N_data_len))

end = time.time()

print("Time elapsed: %.2f seconds"%(end-start))

# ---------------- Print acceptance ratios ---------------- #

if canonical:
    print("\nEnsemble: Canonical\n")
else:
    print("\nEnsemble: Grand Canonical\n")
print("-------- Z-configuration fraction --------")
print("Z-fraction: %.2f%% (%d/%d) "%(100*Z_frac/M,Z_frac,M))

if canonical:
    print("-------- N-configuration fraction --------")
    print("N-fraction: %.2f%% (%d/%d) "%(100*N_sector_ctr/N_data_len,N_sector_ctr,N_data_len))
    
print(f"{mu}|{list(zip(N_bins,P_N))}|{eta}|{Z_frac/M}")

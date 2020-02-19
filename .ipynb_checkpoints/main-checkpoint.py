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
import pickle

is_pickled = False

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
parser.add_argument("--eta",help="Worm end fugacity (default: 1/sqrt(<N_flats>)",
                    type=float,metavar='\b')
parser.add_argument("--beta",help="Thermodynamic beta 1/(K_B*T) (default: 1.0)",
                    type=float,metavar='\b')
parser.add_argument("--tau-slice",help="Im. time slice at which to measure (default: beta/2)",
                    type=float,metavar='\b')
parser.add_argument("--dtau",help="Measurement window)",
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

# Optional arguments (done this way b.c of some argparse bug) 
t = 1.0 if not(args.t) else args.t
beta = 1.0 if not(args.beta) else args.beta
M = int(1E+05) if not(args.M) else args.M
canonical = False if not(args.canonical) else True
tau_slice = beta/2 if not(args.tau_slice) else args.tau_slice
dtau = 0.5 if not(args.dtau) else args.dtau

# Initial eta value (actual value will be obtained in pre-equilibration stage)
eta = 1/np.sqrt(L*beta)

# Initialize Fock state
alpha = pimc.random_boson_config(L,N)
alpha = [1]*L

# Create worldline data structure
data_struct = pimc.create_data_struct(alpha,L)

# List that will contain site and kink indices of worm head and tail
head_loc = []
tail_loc = []

# Trackers
N_tracker = [N]         # Total particles
N_flats_tracker = [L]   # Total flat regions

# Initialize values to be measured
diagonal_list = []
kinetic_list = []
tr_diagonal_list = []
tr_kinetic_list = []
N_list = []              # average total particles 
occ_list = []            # average particle occupation
E_N_list = []            # Fixed total particle energies
    
# ---------------- Pre-Equilibration and Lattice PIMC ---------------- #

# First iteration: eta optimization
# Second iteration: mu optimization (canonical simulations only)
# Second iteration: Main run
    
is_pre_equilibration = True
need_main = True
if is-:

    # Print out the stage of the code.
    if is_pre_equilibration:
        print("\nStarting pre-equilibration stage. Determining eta...")
    else:
        print("LatticePIMC started...")

    # Counters for acceptance and proposal of each move
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

    N_flats_calibration = 0     # Average number of flats
    N_calibration = 0           # Store total N to check mu is good
    calibration_samples = 0     # Pre-equilibration samples
    

    # Count measurements and measurement attempts
    measurements = [0,0] # [made,attempted]

    # Set how many times the set of updates will be attempted based on stage of the code
    M_pre = 1.0E+05
    if is_pre_equilibration:
        iterations = M_pre
    else: # main run
        iterations = M

    # Randomly propose every update M times
    for m in range(int(iterations)):
        
        # assign a label to each update
        labels = list(range(15)) # There are 15 update functions
        shuffle(labels)
        
        # At every mc step, try EVERY update in random order
        for label in labels:   
            
            # Non-Spaceshift moves
            if label == 0:
                pimc.worm_insert(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                    N_tracker,
                    insert_worm_data,insert_anti_data,N_flats_tracker)

            elif label == 1:
                pimc.worm_delete(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                    N_tracker,delete_worm_data,delete_anti_data,N_flats_tracker)

            elif label == 2:
                #pass
                pimc.worm_timeshift(data_struct,beta,head_loc,tail_loc,U,mu,L,N,canonical,
                    N_tracker,advance_head_data,recede_head_data,
                    advance_tail_data,recede_tail_data,N_flats_tracker)

            elif label == 3:
#                 pass
                pimc.insertZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                    N_tracker,insertZero_worm_data,insertZero_anti_data,N_flats_tracker)

            elif label == 4:
#                pass
                pimc.deleteZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                    N_tracker,deleteZero_worm_data,deleteZero_anti_data,N_flats_tracker)

            elif label == 5:
#                pass
                pimc.insertBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                    N_tracker,insertBeta_worm_data,insertBeta_anti_data,N_flats_tracker)

            elif label == 6:
#                pass
                pimc.deleteBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                    N_tracker,deleteBeta_worm_data,deleteBeta_anti_data,N_flats_tracker)

            # Spaceshift moves   
            elif label == 7:
                pimc.insert_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,ikbh_data,N_flats_tracker)  

            elif label == 8:
                pimc.delete_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,dkbh_data,N_flats_tracker) 

            elif label == 9:
                pimc.insert_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,ikah_data,N_flats_tracker)   

            elif label == 10:
                pimc.delete_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,dkah_data,N_flats_tracker)

            elif label == 11:
                pimc.insert_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,ikbt_data,N_flats_tracker)  

            elif label == 12:
                pimc.delete_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,dkbt_data,N_flats_tracker) 

            elif label == 13:
                pimc.insert_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,ikat_data,N_flats_tracker)   

            else:
                pimc.delete_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                    N_tracker,dkat_data,N_flats_tracker)
        
        
        # Count <N_flats> and <N>
        if is_pre_equilibration:
                if m >= M_pre/2: # Only count after half of pre-equilibration steps
                    N_flats_calibration += N_flats_tracker[0]
                    N_calibration += N_tracker[0]
                    calibration_samples += 1
           
        else: # Main loop

            # Attempt to measure every L*beta steps
            if m%int(L*beta)==0:
                
                # Add to MEASUREMENTS ATTEMPTS counter
                measurements[1] += 1

                # Make measurement if no worm ends present
                if not(pimc.check_worm(head_loc,tail_loc)):

                    # Add to MEASUREMENTS MADE counter
                    measurements[0] += 1
                    
                    if canonical:
                        
                     #print(N_tracker[0])

                        #if round(N_tracker[0])==N:
                        if round(N_tracker[0],12)==N:

                            kinetic,diagonal = pimc.tau_resolved_energy(data_struct,beta,dtau,U,mu,t,L)
                            tr_kinetic_list.append(np.array(kinetic))
                            tr_diagonal_list.append(np.array(diagonal))

                            # Total number of particles in worldline configuration
                            N_list.append(N_tracker[0])     

                            if not(is_pickled) and m > int(m/2):
                                kinetic,diagonal = pimc.tau_resolved_energy(data_struct,beta,dtau,U,mu,t,L)
                                print(kinetic,diagonal)
                                with open('pickled_config.pickle', 'wb') as pfile:
                                    pickle.dump(data_struct,pfile,pickle.HIGHEST_PROTOCOL)

                                is_pickled = True

                        else: # Worldline doesn't have target particle number
                            measurements[0] -= 1 # Disregard measurement

                    else: # Grand canonical

                        # Energies
                        kinetic,diagonal = pimc.bh_egs(data_struct,beta,dtau,U,mu,t,L,tau_slice)
                        kinetic_list.append(kinetic)
                        diagonal_list.append(diagonal)

                        # Total number of particles in worldline configuration
                        N_list.append(N_tracker[0])

   

    # Calculate average pre-equilibration <N_flats> and N
    if is_pre_equilibration:  
        N_flats_calibration /= calibration_samples
        N_calibration /= calibration_samples
        
        # Set the worm end fugacity to 1/sqrt(2*<N_flats>) (unless it was user defined)
        eta = 1/(np.sqrt(N_flats_calibration)) if not(args.eta) else args.eta  # beta=1,mu=1

        if round(N_calibration,4) != N:
            
            # Stay in pre_equilibration
            is_pre_equilibration = True
            
            # Tweak mu based on total particles measured
            if round(N_calibration,4) >= N:
                mu -= 0.01
            else:
                mu += 0.01
        
        else: # N is good
            
            # Turn pre-equilibration off
            is_pre_equilibration = False

    # Print out the value of eta or indicate when the main loop ends.
    if is_pre_equilibration:
        print("Pre-Equilibration stage complete.")
        print("eta = %.4f and mu = %.4f "%(eta,mu))
    else:
        print("Lattice PIMC done. Saving data to disk...")

# ---------------- Format data and save to disk ---------------- #

# Promote lists to arrays so we can use np.savetxt
diagonal_list = np.array(diagonal_list)            # <H_0>
kinetic_list = np.array(kinetic_list)              # <H_1>
total_list = np.array(kinetic_list+diagonal_list)  # <H_0> + <H_1> 
N_list = np.array(N_list)                          # <N>

# Combine all arrays to a single array
data_list = [diagonal_list/t,kinetic_list/t,total_list,N_list]

# Create file header
header_K = '{0:^67s}\n{1:^6s} {2:^29s} {3:^10s} {4:^27s}'.format(
    "L=%i,N=%i,U=%.4f,mu=%.4f,t=%.4f,eta=%.4f,beta=%.4f,tau_slice=%.4f,M=%i"%(L,N,U,mu,t,eta,beta,tau_slice,M),
    'H_0/t','H_1/t','E/t', 'N')
header_V = '{0:^67s}\n{1:^6s} {2:^29s} {3:^10s} {4:^27s}'.format(
    "L=%i,N=%i,U=%.4f,mu=%.4f,t=%.4f,eta=%.4f,beta=%.4f,tau_slice=%.4f,M=%i"%(L,N,U,mu,t,eta,beta,tau_slice,M),
    'H_0/t','H_1/t','E/t', 'N')

# Save to disk
# if canonical:
#     with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f_%i_can.dat"%(L,N,U,mu,t,eta,beta,tau_slice,M),"w+") as data:
#         np.savetxt(data,np.transpose(data_list),delimiter=" ",fmt="%-20s",header=header) 
# else:
#     with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f_%i_gcan.dat"%(L,N,U,mu,t,eta,beta,tau_slice,M),"w+") as data:
#         np.savetxt(data,np.transpose(data_list),delimiter=" ",fmt="%-20s",header=header) 

# Save to disk
if canonical: # kinetic
    with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canK.dat"%(L,N,U,mu,t,beta,M),"w+") as data:
        np.savetxt(data,tr_kinetic_list,delimiter=" ",fmt="%-20s",header=header_K) 
if canonical: # diagonal
    with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canV.dat"%(L,N,U,mu,t,beta,M),"w+") as data:
        np.savetxt(data,tr_diagonal_list,delimiter=" ",fmt="%-20s",header=header_V)
if canonical:
    with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canN.dat"%(L,N,U,mu,t,beta,M),"w+") as data:
        np.savetxt(data,np.transpose(N_list),delimiter=" ",fmt="%-20s",header=header_K) 

# ---------------- Print acceptance ratios ---------------- #

if canonical:
    print("\nEnsemble: Canonical\n")
else:
    print("\nEnsemble: Grand Canonical\n")
print("-------- Z-configuration fraction --------")
print("Z-fraction: %.2f%% (%d/%d) "%(100*measurements[0]/measurements[1],measurements[0],measurements[1]))

# Acceptance ratios
print("\n-------- Acceptance Ratios --------\n")

print("       Insert worm: (%d/%d)"%(insert_worm_data[0],insert_worm_data[1]))
print("       Delete worm: (%d/%d)\n"%(delete_worm_data[0],delete_worm_data[1]))

print("       Insert anti: (%d/%d)"%(insert_anti_data[0],insert_anti_data[1]))
print("       Delete anti: (%d/%d)\n"%(delete_anti_data[0],delete_anti_data[1]))

print("       Advance head: (%d/%d)"%(advance_head_data[0],advance_head_data[1]))
print("        Recede head: (%d/%d)\n"%(recede_head_data[0],recede_head_data[1]))

print("       Advance tail: (%d/%d)"%(advance_tail_data[0],advance_tail_data[1]))
print("        Recede tail: (%d/%d)\n"%(recede_tail_data[0],recede_tail_data[1]))

print("   InsertZero worm: (%d/%d)"%(insertZero_worm_data[0],insertZero_worm_data[1]))
print("   DeleteZero worm: (%d/%d)\n"%(deleteZero_worm_data[0],deleteZero_worm_data[1]))

print("   InsertZero anti: (%d/%d)"%(insertZero_anti_data[0],insertZero_anti_data[1]))
print("   DeleteZero anti: (%d/%d)\n"%(deleteZero_anti_data[0],deleteZero_anti_data[1]))

print("   InsertBeta worm: (%d/%d)"%(insertBeta_worm_data[0],insertBeta_worm_data[1]))
print("   DeleteBeta worm: (%d/%d)\n"%(deleteBeta_worm_data[0],deleteBeta_worm_data[1]))

print("   InsertBeta anti: (%d/%d)"%(insertBeta_anti_data[0],insertBeta_anti_data[1]))
print("   DeleteBeta anti: (%d/%d)\n"%(deleteBeta_anti_data[0],deleteBeta_anti_data[1]))

print("              IKBH: (%d/%d)"%(ikbh_data[0],ikbh_data[1])) 
print("              DKBH: (%d/%d)\n"%(dkbh_data[0],dkbh_data[1]))

print("              IKAH: (%d/%d)"%(ikah_data[0],ikah_data[1])) 
print("              DKAH: (%d/%d)\n"%(dkah_data[0],dkah_data[1])) 

print("              IKBT: (%d/%d)"%(ikbt_data[0],ikbt_data[1])) 
print("              DKBT: (%d/%d)\n"%(dkbt_data[0],dkbt_data[1]))

print("              IKAT: (%d/%d)"%(ikat_data[0],ikat_data[1])) 
print("              DKAT: (%d/%d)\n"%(dkat_data[0],dkat_data[1])) 

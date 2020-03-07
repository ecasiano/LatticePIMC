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
import random
import datetime

time = datetime.datetime.now()
timeID = int((str(time).split(":")[1]+str(time).split(":")[2]).replace('.',''))

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
parser.add_argument("--n-slices",help="Measurement window)",
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

# Initial eta value (actual value will be obtained in pre-equilibration stage)
eta = 1/np.sqrt(L*beta)

# Initialize Fock state
alpha = pimc.random_boson_config(L,N)

# Create worldline data structure
data_struct = pimc.create_data_struct(alpha,L)

# List that will contain site and kink indices of worm head and tail
head_loc = []
tail_loc = []

# Trackers
N_tracker = [N]         # Total particles
N_flats_tracker = [L]   # Total flat regions

  
# ---------------- Pre-Equilibration ---------------- #

print("\nStarting pre-equilibration stage. Determining eta and mu...\n")

print("  eta  |   mu   | N_calibration | N_target | Z_calibration")

is_pre_equilibration = True
need_eta = True
while(is_pre_equilibration):

    # Counters for acceptance and proposal of each move
    dummy_data = [0,0]

    N_flats_calibration = 0     # Average number of flats
    N_calibration = 0           # Store total N to check mu is good
    Z_calibration = 0           # Count diagonal fraction in pre-equil stage
    Z = 0.80                     # Minimum diagonal fraction desired
    calibration_samples = 0     # Pre-equilibration samples    

    # Randomly propose every update M_pre times
    for m in range(int(M_pre)):
        
        # assign a label to each update
        label = np.random.randint(15) # There are 15 update functions

        # Non-Spaceshift moves
        if label == 0:
            pimc.worm_insert(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,dummy_data,N_flats_tracker)

        elif label == 1:
            pimc.worm_delete(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,dummy_data,N_flats_tracker)

        elif label == 2:
            #pass
            pimc.worm_timeshift(data_struct,beta,head_loc,tail_loc,U,mu,L,N,canonical,
                N_tracker,dummy_data,dummy_data,dummy_data,dummy_data,N_flats_tracker)

        elif label == 3:
            pimc.insertZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,dummy_data,N_flats_tracker)

        elif label == 4:
            pimc.deleteZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,dummy_data,N_flats_tracker)

        elif label == 5:
            pimc.insertBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,dummy_data,N_flats_tracker)

        elif label == 6:
            pimc.deleteBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,dummy_data,N_flats_tracker)

        # Spaceshift moves   
        elif label == 7:
            pimc.insert_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker)  

        elif label == 8:
            pimc.delete_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker) 

        elif label == 9:
            pimc.insert_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker)   

        elif label == 10:
            pimc.delete_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker)

        elif label == 11:
            pimc.insert_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker)  

        elif label == 12:
            pimc.delete_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker) 

        elif label == 13:
            pimc.insert_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker)   

        else:
            pimc.delete_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,
                N_tracker,dummy_data,N_flats_tracker)
              
        # Count <N_flats> and <N>
        trash_percent = 0.25
        if m >= int(M_pre*trash_percent): # Only count after half of pre-equilibration steps
            N_flats_calibration += N_flats_tracker[0]
            N_calibration += N_tracker[0]
            calibration_samples += 1

        if not(pimc.check_worm(head_loc,tail_loc)) and m >= int(M_pre*trash_percent):
            Z_calibration += 1
   
    # Calculate average pre-equilibration <N_flats>, N, and Z fraction
    N_flats_calibration /= calibration_samples
    N_calibration /= calibration_samples
    N_calibration = round(N_calibration,1)
    Z_samples = M_pre - int(M_pre*trash_percent)
    Z_calibration /= Z_samples

    # Set the worm end fugacity to 1/sqrt(2*<N_flats>) (unless it was user defined)
    if need_eta:
        eta = 1/(np.sqrt(N_flats_calibration)) if not(args.eta) else args.eta  # beta=1,mu=1
        need_eta = False

    # Print the current set of parameters
    print("%.4f | %.4f | %.1f | %i | %.2f"%(eta,mu,N_calibration,N,Z_calibration))
    
    if N_calibration != N or (Z_calibration < Z or Z_calibration >= 1.2*Z):
        
        # Stay in pre_equilibration stage
        is_pre_equilibration = True

        # Tweak mu based on total particles measured
        if N_calibration >= N:
            mu -= 0.1
        else:
            mu += 0.1
            
        # Tweak eta based on diagonal fraction obtained
        if Z_calibration <= Z:
            eta -= (eta*0.1)
        else:
            eta += (eta*0.1)                        

    else: # N is good

        # Turn pre-equilibration off
        is_pre_equilibration = False
        
print("Pre-Equilibration done.\n")

# ---------------- Lattice PIMC ---------------- #

# Open files that will save data
if canonical: # kinetic
    
    
    if not(no_energies):
        kinetic_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canK.dat"%(L,N,U,mu,t,beta,M),"w+")
        diagonal_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canV.dat"%(L,N,U,mu,t,beta,M),"w+")
    
    if get_fock_state:
        fock_state_file = open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_%d_fock.dat"%(L,N,U,mu,t,beta,M,timeID),"w+")
        
print("LatticePIMC started...\n")

# Initialize values to be measured
bin_ctr = 0
tau_slices = np.linspace(0,beta,n_slices)[1:-1][::2] # slices were observables are measured
tr_kinetic_list = np.zeros_like(tau_slices)
tr_diagonal_list = np.zeros_like(tau_slices)
N_list = []              # average total particles 
occ_list = []            # average particle occupation
E_N_list = []            # Fixed total particle energies

# Energies using original method
kinetic_og_list = []
diagonal_og_list = []

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

# Count measurements and measurement attempts
measurements = [0,0] # [made,attempted]

# Randomly an update M times
for m in range(int(M)):
        
    label = np.random.randint(15)

    # Non-Spaceshift moves
    if label == 0:
        pimc.worm_insert(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
            N_tracker,
            insert_worm_data,insert_anti_data,N_flats_tracker)

    elif label == 1:
        pimc.worm_delete(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
            N_tracker,delete_worm_data,delete_anti_data,N_flats_tracker)

    elif label == 2:
        pimc.worm_timeshift(data_struct,beta,head_loc,tail_loc,U,mu,L,N,canonical,
            N_tracker,advance_head_data,recede_head_data,
            advance_tail_data,recede_tail_data,N_flats_tracker)

    elif label == 3:
        pimc.insertZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
            N_tracker,insertZero_worm_data,insertZero_anti_data,N_flats_tracker)

    elif label == 4:
        pimc.deleteZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
            N_tracker,deleteZero_worm_data,deleteZero_anti_data,N_flats_tracker)

    elif label == 5:
        pimc.insertBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,
            N_tracker,insertBeta_worm_data,insertBeta_anti_data,N_flats_tracker)

    elif label == 6:
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
    
    # Print completion percent
    if m%(int(M/10))==0:        
        print("%.2f%%"%(100*m/M))
    else: pass
                 
                
    # After 25% equilibration, measure every L*beta steps
    if beta < 1: 
        if m%(int(1000*L)):
            try_measure=True
        else:
            try_measure=False
    else:
        if m%(int(1000*L*beta))==0:
            try_measure=True
        else:
            try_measure=False
            
    if try_measure and m > int(M*0.25):  
        
        # Add to MEASUREMENTS ATTEMPTS counter
        measurements[1] += 1
        
        # Make measurement if no worm ends present
        if not(pimc.check_worm(head_loc,tail_loc)):
    
            if canonical: # Canonical simulation
     
                #print(round(N_tracker[0],10),N_tracker[0])
                if round(N_tracker[0],10)==N:
                    
                    bin_ctr += 1

                    # Add to MEASUREMENTS MADE counter
                    measurements[0] += 1

                    if not(no_energies):
                        
                        # Energies, but measured at different tau slices (time resolved)
                        tr_kinetic,tr_diagonal = pimc.tau_resolved_energy(data_struct,beta,n_slices,U,mu,t,L)
                        tr_kinetic_list += tr_kinetic # cumulative sum
                        tr_diagonal_list += tr_diagonal # cumulative sum

                        # Take the binned average of the time resolved energies
                        if bin_ctr == bin_size:

                            tr_kinetic_list /= bin_size
                            tr_diagonal_list /= bin_size

                            # Write to file(s)
                            data_len = len(tr_kinetic_list)
                            np.savetxt(kinetic_file,np.reshape(tr_kinetic_list,(1,data_len)),fmt="%.16f",delimiter=" ")
                            np.savetxt(diagonal_file,np.reshape(tr_diagonal_list,(1,data_len)),fmt="%.16f",delimiter=" ")

                            # Reset bin_ctr and time resolved energies arrays
                            bin_ctr = 0
                            tr_kinetic_list *= 0
                            tr_diagonal_list *= 0
                            
                    if get_fock_state:

                        # Measue the Fock state at tau=beta
                        alpha = np.array(pimc.get_fock_state_beta(data_struct,L),dtype=int)
                        
                        # Write to file
                        np.savetxt(fock_state_file,np.reshape(alpha,(1,L)),fmt="%i",delimiter=" ")                   

#                     if not(is_pickled) and m > int(m/2):
#                         kinetic,diagonal = pimc.tau_resolved_energy(data_struct,beta,n_slices,U,mu,t,L)
#                         #print(kinetic,diagonal)
#                         with open('pickled_config.pickle', 'wb') as pfile:
#                             pickle.dump(data_struct,pfile,pickle.HIGHEST_PROTOCOL)

#                         is_pickled = True

                else: # Worldline doesn't have target particle number
                    pass
                
            else: # Grand canonical
            	pass
                # # Energies
                # kinetic,diagonal = pimc.bh_egs(data_struct,beta,dtau,U,mu,t,L,tau_slice)
                # kinetic_list.append(kinetic)
                # diagonal_list.append(diagonal)

                # # Total number of particles in worldline configuration
                # N_list.append(N_tracker[0])
                
        else: # There's a worm in the worldline configuration
            pass
        
    else: # We  only measure every L*beta steps
        pass
                
# Close the data files
if not(no_energies):
    kinetic_file.close()
    diagonal_file.close()
if get_fock_state:
    fock_state_file.close()

# Save diagonal fraction obtained from main loop
Z_frac = 100*measurements[0]/measurements[1]
print("100%%")
print("\nLattice PIMC done.")


# ---------------- Format data and save to disk ---------------- #

# # Promote lists to arrays so we can use np.savetxt
# diagonal_list = np.array(diagonal_list)            # <H_0>
# kinetic_list = np.array(kinetic_list)              # <H_1>
# total_list = np.array(kinetic_list+diagonal_list)  # <H_0> + <H_1> 
# N_list = np.array(N_list)                          # <N>

# # Combine all arrays to a single array
# data_list = [diagonal_list/t,kinetic_list/t,total_list,N_list]

# # Create file headers
# header_K = '{0:^67s}\n{1:^6s} {2:^29s} {3:^10s} {4:^27s} {5:^10s}'.format(
#     "L=%i,N=%i,U=%.4f,mu=%.4f,t=%.4f,eta=%.4f,n_slices=%i,beta=%.2f,M=%i,Z_frac=%.2f%%"%(L,N,U,mu,t,eta,n_slices,beta,M,Z_frac),
#     '0.1*tau/b','0.3*tau/b','0.5*tau/b','0.7*tau/b','1.0*tau/b')
# header_V = '{0:^67s}\n{1:^6s} {2:^29s} {3:^10s} {4:^27s} {5:^10s}'.format(
#     "L=%i,N=%i,U=%.4f,mu=%.4f,t=%.4f,eta=%.4f,n_slices=%i,beta=%.2f,M=%i,Z_frac=%.2f%%"%(L,N,U,mu,t,eta,n_slices,beta,M,Z_frac),
#     '0.1*tau/b','0.3*tau/b','0.5*tau/b','0.7*tau/b','1.0*tau/b')
# header_N = '{0:^67s}\n{1:^6s} {2:^29s} {3:^10s} {4:^27s} {5:^10s}'.format(
#     "L=%i,N=%i,U=%.4f,mu=%.4f,t=%.4f,eta=%.4f,n_slices=%i,beta=%.2f,M=%i,Z_frac=%.2f%%"%(L,N,U,mu,t,eta,n_slices,beta,M,Z_frac),
#     '0.1*tau/b','0.3*tau/b','0.5*tau/b','0.7*tau/b','1.0*tau/b')


# # Save time resolved energies to disk
# if canonical: # kinetic
#     with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canK.dat"%(L,N,U,mu,t,beta,M),"w+") as data:
#         np.savetxt(data,tr_kinetic_list,delimiter=" ",fmt="%-20s",header=header_K) 
# if canonical: # diagonal
#     with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%i_canV.dat"%(L,N,U,mu,t,beta,M),"w+") as data:
#         np.savetxt(data,tr_diagonal_list,delimiter=" ",fmt="%-20s",header=header_V)
# if canonical:
#     with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%.4f_%i_canN.dat"%(L,N,U,mu,t,n_slices,beta,M),"w+") as data:
#         np.savetxt(data,np.transpose(N_list),delimiter=" ",fmt="%-20s",header=header_K) 
        
# # Save energies using original method to disk
# header_og = '{0:^67s}\n{1:^6s} {2:^29s}'.format(
#     "L=%i,N=%i,U=%.4f,mu=%.4f,t=%.4f,eta=%.4f,n_slices=%.2f,beta=%.2f,M=%i,Z_frac=%.2f%%"%(L,N,U,mu,t,eta,n_slices,beta,M,Z_frac),
# 'H_0/t','H_1/t')
# if canonical: # kinetic
#     with open("%i_%i_%.4f_%.4f_%.4f_%.4f_%.4f_%i_canK_og.dat"%(L,N,U,mu,t,n_slices,beta,M),"w+") as data:
#         np.savetxt(data,np.transpose([diagonal_og_list,kinetic_og_list]),delimiter=" ",fmt="%-20s",header=header_og) 


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
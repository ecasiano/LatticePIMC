# Functions to be used in main file (lattice_pimc.py)
import numpy as np
import bisect
import matplotlib.pyplot as plt

def random_boson_config(L,N):
    '''Generates a random configuration of N bosons in a 1D lattice of size L'''

    alpha = np.zeros(L,dtype=int) # Stores the random configuration of bosons
    for i in range(N):
        r = np.random.randint(L)
        alpha[r] += 1

    return alpha

'----------------------------------------------------------------------------------'

def create_data_struct(alpha):
    '''Generate the [tau,N,(src,dest)] data_struct from the configuration'''
    L = len(alpha)

    data_struct = []
    for i in range(L):
        data_struct.append([[0,alpha[i],(i,i)]])

    return data_struct

'----------------------------------------------------------------------------------'


def worm_insert(data_struct, beta, ira_loc, masha_loc, U, mu, eta):
    '''Inserts a worm or antiworm'''

    # Can only insert worm if there are no wormends present
    if ira_loc != [] or masha_loc != [] : return None

    # Number of lattice sites
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    p_L = 1/L # probability of selecting the L site

    # Randomly select a flat tau interval at which to possibly insert worm
    n_flats = len(data_struct[i])
    flat_min_idx = np.random.randint(n_flats)           # Index of lower bound of flat region
    tau_min = data_struct[i][flat_min_idx][0]
    if flat_min_idx == n_flats - 1 : tau_max = beta     # In case that last flat is chosen
    else : tau_max = data_struct[i][flat_min_idx+1][0]
    p_flat = 1/n_flats                                  # prob. of selecting the flat interval
    dtau_flat = tau_max - tau_min                       # length of the flat interval

    # Randomly choose either to insert worm or, if possible, an antiworm
    n_i = data_struct[i][flat_min_idx][1]  # initial number of particles in the flat interval
    if n_i == 0 : # only worm can be inserted
        insert_worm = True
        p_wormtype = 1
    else:
        if np.random.random() < 0.5:
            insert_worm = True
        else:
            insert_worm = False
        p_wormtype = 0.5 # prob. of the worm being either a worm or antiworm

    # Randomly choose the length of the worm or antiworm
    dtau_worm  = np.random.random()*(dtau_flat)
    p_wormlen = 1/(dtau_flat) # prob. of the worm being of the chosen length

    # Randomly choose the time where the first worm end will be inserted
    if insert_worm: # worm
        tau_2 = tau_min + np.random.random()*(dtau_flat - dtau_worm) # worm tail (creates a particle)
        tau_1 = tau_2 + dtau_worm                                    # worm head (destroys a particle)
    else: # antiworm
        tau_1 = tau_min + np.random.random()*(dtau_flat - dtau_worm)
        tau_2 = tau_1 + dtau_worm
    p_tau = 1/(dtau_flat-dtau_worm)     # prob. of inserting the worm end at the chosen time

    # Reject update if worm end is inserted at the bottom kink of the flat
    # (this will probably never happen in the 2 years I have left to complete my PhD :p )
    if tau_1 == tau_min or tau_2 == tau_min : return None

    # Reject update if both worm ends are at the same tau
    if tau_1 == tau_2 :
        return None

    # Build the worm end kinks to be inserted on i
    if insert_worm: # worm
        N_after_masha = data_struct[i][flat_min_idx][1] + 1
        N_after_ira = N_after_masha - 1
        masha_kink = [tau_2,N_after_masha,(i,i)]
        ira_kink = [tau_1,N_after_ira,(i,i)]
    else: # antiworm
        N_after_ira = data_struct[i][flat_min_idx][1] - 1
        #if N_after_ira == -1 : return None # Reject update if there were no particles for Ira to destroy
        N_after_masha = N_after_ira + 1
        ira_kink = [tau_1,N_after_ira,(i,i)]
        masha_kink = [tau_2,N_after_masha,(i,i)]

    # Calculate the change in potential energy (will be a factor of the Metropolis condition later on)
    if insert_worm == True:            # case: inserted worm
        dV = U*n_i + mu
        weight_ratio = (n_i+1)*eta**2 * np.exp(-dtau_worm*dV)   # w_+ / w_- = worm_config / wormless_config
    else:
        dV = U*(1-n_i) - mu            # case: inserted antiworm
        weight_ratio = (n_i)*eta**2 * np.exp(-dtau_worm*dV)   # w_+ / w_- = worm_config / wormless_config

    # Build the Metropolis ratio (R)
    p_tunable = 1                                     # p_delete / p_insert (tunable)
    R = weight_ratio / (p_tunable * p_L * p_flat * p_wormtype * p_wormlen * p_tau)
    # Metropolis Sampling
    if np.random.random() < R:
        # Insert worm
        if insert_worm:
            if flat_min_idx == n_flats - 1: # if selected flat is the last
                data_struct[i].append(masha_kink)
                data_struct[i].append(ira_kink)
            else:
                data_struct[i].insert(flat_min_idx+1,masha_kink)
                data_struct[i].insert(flat_min_idx+2,ira_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            masha_loc.extend([i,flat_min_idx+1])
            ira_loc.extend([i,flat_min_idx+2])

        # Insert antiworm
        else:
            if flat_min_idx == n_flats - 1: # last flat
                data_struct[i].append(ira_kink)
                data_struct[i].append(masha_kink)
            else:
                data_struct[i].insert(flat_min_idx+1,ira_kink)
                data_struct[i].insert(flat_min_idx+2,masha_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            ira_loc.extend([i,flat_min_idx+1])
            masha_loc.extend([i,flat_min_idx+2])

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def worm_delete(data_struct, beta, ira_loc, masha_loc, U, mu, eta):

    # Can only propose worm deletion if both worm ends are present
    if ira_loc == [] or masha_loc == [] : return None

    # Only delete if worm ends are on the same site and on the same flat interval
    if ira_loc[0] != masha_loc[0] or abs(ira_loc[1]-masha_loc[1]) != 1: return None

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]
    
    # Identify the type of worm
    if ik > mk : is_worm = True   # worm
    else: is_worm = False         # antiworm
    
    # Calculate worm length
    tau_1 = data_struct[ix][ik][0]
    tau_2 = data_struct[mx][mk][0]
    dtau = np.abs(tau_1-tau_2)

    # Identify the lower and upper limits of the flat interval where the worm lives
    if is_worm:
        tau_min = data_struct[mx][mk-1][0] 
        tau_max = data_struct[ix][ix+1][0]
        n_before_worm = data_struct[mk][mk-1][1]
    else: # antiworm
        tau_min = data_struct[ix][ix-1][0] 
        tau_max = data_struct[mk][mk+1][0] 
        n_before_worm = data_struct[ix][ik-1][1]

    # Worm insert proposal probability
    p_L = 1/len(data_struct)           # prob of choosing site
    p_flat = 1/len(data_struct[ix])    # prob of choosing flat
    if n_before_worm == 0:             # prob of choosing worm/antiworm
        p_wormtype = 1 # Only a worm could've been inserted
    else:
        p_wormtype = 1/2
    p_wormlen = 1/(tau_max-tau_min)    # prob of choosing wormlength
    p_tau = 1/((tau_max-tau_min)-dtau) # prob of choosing the tau of the first wormend
    
    # Choose the appropriate weigh ratio based on the worm type
    if is_worm:
        n_i = data_struct[mx][mk][1]   # particles before delete
        dV = U*(1-n_i) - mu            # deleted energy minus energy of worm still there
        weight_ratio = np.exp(dV*dtau)/(n_i*eta**2)   # W_deleted/W_stillthere
    else: # delete antiworm
        n_i = data_struct[ix][ik][1]
        dV =  U*n_i + mu
        weight_ratio = np.exp(dV*dtau)/(n_i*eta**2)
                   
    # Metropolis sampling
    # Accept
    p_tunable = 1 # p_iw/p_dw
    R = (p_tunable * p_L * p_flat * p_wormtype * p_wormlen * p_tau) * weight_ratio 
    if np.random.random() < R:
        # Delete the worm ends
        if ik > mk : # worm
            del data_struct[ix][ik] # Deletes ira
            del data_struct[mx][mk] # Deletes masha
        else: # antiworm
            del data_struct[mx][mk] # Deletes masha
            del data_struct[ix][ik] # Deletes ira

        # Update the locations ira and masha
        del ira_loc[:]
        del masha_loc[:]

        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def gsworm_insert(data_struct, beta, ira_loc, masha_loc, U, mu, eta):

    '''Insert a ground state (T=0) worm or antiworm (looks like inserting only one end)'''

    # Update only possible if there's either zero or one worm ends
    if ira_loc != [] and masha_loc != []:
        return None

    # Number of lattice sites
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    p_L = 1/L # probability of selecting the L site

    # Randomly choose to insert at the first or last flat interval
    if len(data_struct[i]) != 1: # Worldline with kinks
        if np.random.random() < 0.5:
            is_first_flat = True
        else:
            is_first_flat = False
        p_flat = 0.5

    else: # Case of i_th worldline having no kinks
        is_first_flat = False # Due to data_struct, convenient to treat worldline as last flat
        p_flat = 1

    # Determine the lower and upper bounds of the worm insertion
    if is_first_flat:
        flat_min_idx = 0
        tau_min = 0
        tau_max = data_struct[i][flat_min_idx+1][0]
    else:
        flat_min_idx = -1
        tau_min = data_struct[i][flat_min_idx][0]
        tau_max = beta

    # Randomly choose the type: ground state i) worm OR ii) antiworm
    n_i = data_struct[i][flat_min_idx][1] # initial particles on the selected flat
    if n_i == 0:
        insert_worm = True
        p_wormtype = 1
    else:
        if np.random.random() < 0.5:
            insert_worm = True
        else:
            insert_worm = False
        p_wormtype = 0.5

    # Randomly choose the time where the worm will be inserted
    dtau_flat = tau_max - tau_min                 # length of the flat interval
    tau = tau_min + np.random.random()*dtau_flat  # insertion time
    p_tau = 1/dtau_flat

    # Reject update if the insertion time is the same as the kink time
    if tau == tau_min :
        return None

    # Build the worm end kink to be inserted on i & the previous one if it were to be modified
    if insert_worm:
        if is_first_flat:
            worm_kink = [tau,n_i,(i,i)]
            before_worm_kink = [tau_min,n_i+1,(i,i)]
        else: # last flat
            worm_kink = [tau,n_i+1,(i,i)]
            before_worm_kink = [tau_min,n_i,(i,i)]
    else: # insert antiworm
        if is_first_flat:
            worm_kink = [tau,n_i,(i,i)]
            before_worm_kink = [tau_min,n_i-1,(i,i)]
        else: # last flat
            worm_kink = [tau,n_i-1,(i,i)]
            before_worm_kink = [tau_min,n_i,(i,i)]

    # Calculate the change in potential energy (will be a factor of the Metropolis condition later on)
    if insert_worm == True:            # case: inserted worm
        dV = U*n_i + mu_i
    else:
        dV = U*(1-n_i) - mu_i    # case: inserted antiworm

    # Build the Metropolis ratio (R)
    p_ratio = 1                                     # p_delete / p_insert
    #weight_ratio = eta**2 * np.exp(-dtau_worm*dV)   # w_+ / w_- = worm_config / wormless_config
    #R = p_ratio * weight_ratio / (p_L * p_flat * p_wormtype * p_wormlen * p_tau)

    # Metropolis Sampling

    # To Do:  BUILD THE METROPOLIS CONDITION

    # Accept
    worm_weight = 1
    if np.random.random() < 0.5:
        if insert_worm:
            if is_first_flat:
                data_struct[i][flat_min_idx] = before_worm_kink
                data_struct[i].insert(flat_min_idx+1,worm_kink)

                # Update worm end location
                ira_loc.extend([i,flat_min_idx+1])

            else:
                data_struct[i][flat_min_idx] = before_worm_kink
                data_struct[i].insert(flat_min_idx+1,worm_kink)

                # Update worm end location
                masha_loc.extend([i,flat_min_idx+1])

        else: # insert antiworm
            if is_first_flat:
                data_struct[i][flat_min_idx] = before_worm_kink
                data_struct[i].insert(flat_min_idx+1,worm_kink)

                # Update worm end location
                masha_loc.extend([i,flat_min_idx+1])

            else:
                data_struct[i][flat_min_idx] = before_worm_kink
                data_struct[i].insert(flat_min_idx+1,worm_kink)

                # Update worm end location
                ira_loc.extend([i,flat_min_idx+1])

        # Check if there is now a worm
        if ira_loc != [] and masha_loc != []: is_worm_present = True

        return None

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def gsworm_delete(data_struct, beta, ira_loc, masha_loc, U, mu, eta):

    #### THIS IS CURRENTLY THE SAME AS worm_delete() ###
    ### ### ### a;sldjfhas;dufhad ### ### ## ###

    # Can only propose worm deletion if there is one worm end present
    if ira_loc == [] and masha_loc == [] : return None

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]

    # Count how many worm ends are deletable (i.e, part of either the first or last flat)
    deletable = 0
    ira_deletable = False
    masha_deletable = False
    if ik == 0 or ik == len(data_struct[ix]) - 1 :
        ira_deletable = True
        deletable+=1
    if mk == 0 or mk == len(data_struct[mx]) - 1 :
        masha_deletable = True
        deletable+=1

    # Reject update if neither end lies in the first or last flat region
    if deletable == 0 : return None

    # Decide which worm end to delete


    # Metropolis sampling
    # Accept
    delete_weight = 1
    if np.random.random() < delete_weight:
        # Delete the worm ends
        if ik > mk : # worm
            del data_struct[ix][ik] # Deletes ira
            del data_struct[mx][mk] # Deletes masha
        else: # antiworm
            del data_struct[mx][mk] # Deletes masha
            del data_struct[ix][ik] # Deletes ira

        # Update the worm flag and ira_loc,masha_loc
        del ira_loc[:]
        del masha_loc[:]

        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def worm_timeshift(data_struct,beta,ira_loc,masha_loc, U, mu):

    # Reject update if there is no worm present
    if ira_loc == [] and masha_loc == [] : return None

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]

    # Retrieve the actual times of Ira and Masha
    tau_1 = data_struct[ix][ik][0]
    tau_2 = data_struct[mx][mk][0]

    # Randomly choose to shift Ira or Masha
    if np.random.random() < 0.5:
        shift_ira = True
    else:
        shift_ira = False

    # Save the site and kink indices of the end that will be moved
    if shift_ira == True :
        x = ix
        k = ik
    else:
        x = mx
        k = mk

    # Determine the lower and upper bounds of the worm end for the timeshift
    tau_max_idx = k + 1
    tau_min_idx = k - 1

    # Get tau_max
    if tau_max_idx == len(data_struct[x]):
        tau_max = beta  # This covers the case when there's no kinks after the worm end
    else:
        tau_max = data_struct[x][tau_max_idx][0] #actual times
    # Get tau_min
    if tau_min_idx == 0:
        tau_min = 0
    else:
        tau_min = data_struct[x][tau_min_idx][0]

    # Randomly propose a time for the worm end between tau_min and tau_max
    tau_new = tau_min + np.random.random()*(tau_max - tau_min)

    # Get the diaonal energy differences between new and old configurations
    if shift_ira and tau_new > tau_1:               # ira forward
        n_i = 
    elif shift_ira and tau_new < tau_1:             # ira backward
    elif shift_ira==False and tau_new > tau_2:      # masha forward
    else:                                           # masha backward        
    
    # Metropolis sampling
    # Accept
    weight_timeshift = 1
    if np.random.random() < weight_timeshift:
        data_struct[x][k][0] = tau_new # Modify Ira time
        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def worm_spaceshift_before(data_struct,beta,ira_loc,masha_loc):

    # Update not possible if there's no worm
    if ira_loc == [] and masha_loc == [] : return None

    # Number of lattice sites
    L = len(data_struct)

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]
    tau_1 = data_struct[ix][ik][0]
    tau_2 = data_struct[mx][mk][0]

    # Select the worm end to which the kink to the left will be inserted/deleted
    if ira_loc == [] and masha_loc != []: # Only masha is present
        before_ira = False
        x = mx
        k = mk
        pprob = 1 # Proposal Probability
    elif ira_loc != [] and masha_loc == []: # Only ira is present
        before_ira = True
        x = ix
        k = ik
        pprob = 1
    elif data_struct[mx][mk-1][1] == 0: # Can't insert/delete kink before masha if there's no particles
        before_ira = True
        x = ix
        k = ik
        pprob = 1
    elif np.random.random() < 0.5: # Both are present, select randomly
        before_ira = True
        x = ix
        k = ik
        pprob = 0.5
    else: 
        before_ira = False
        x = mx
        k = mk
        pprob = 0.5
    
    # Check if a kink deletion is possible before the selected end
    if data_struct[x][k-1][2][0] == data_struct[x][k-1][2][1]: # only kinks between n.n. sites can be deleted
        can_delete = False
    else:
        can_delete = True
        
    # Randomly choose to send Ira to j via: a) insert kink b) delete kink
    if can_delete: 
        if np.random.random() < 0.5:
            insert_kink = True  # Will move end by INSERTING kink before
        else:
            insert_kink = False # Will move end by DELETING kink before
        pprob *= 0.5
    else:
        insert_kink = True
        pprob *= 1
        
    # Randomly select the neighboring site at which to send the worm end
    if insert_kink == False: # kink deletion
        j = data_struct[x][k-1][2][0] # end sent to src site of previous kink if deleted
        pprob *= 1
    else:                   # kink insertion
        if np.random.random() < 0.5:
            j = x + 1
            if j == L : j = 0      # PBC
        else:
            j = x - 1
            if j == -1 : j = L - 1 # PBC
        pprob *= 0.5

    # --- Insert kink branch --- #
    if insert_kink == True:

        # Determine lower and upper bound in im. time at which to insert the kink
        if before_ira:
            tau_max = tau_1 # tau_max is at Ira's time
        else:
            tau_max = tau_2 # tau_max is at Masha's time

        # tau_min candidate on site i
        if len(data_struct[x]) == 1: tau_min_i = 0 # case where the site of the wormend is completely flat
        else: tau_min_i = data_struct[x][ik-1][0]
        # tau_min candidate on site j
        for k in range(len(data_struct[j])):
            if len(data_struct[j]) == 1:
                tau_min_j = 0
                flat_min_idx_j = 0
            else:
                if data_struct[j][k][0] > tau_max: break # want the kink with tau preceding tau_max
                tau_min_j = data_struct[j][k][0]
                flat_min_idx_j = k

        # Select the tau_min value between the i and j candidates
        if tau_min_i > tau_min_j : tau_min = tau_min_i
        else : tau_min = tau_min_j

        # Randomly choose the kink time between tau_min and tau_max
        tau_kink = tau_min + np.random.random()*(tau_max - tau_min)

        # Insertion not possible if the proposed kink time happens at the time of another kink
        if tau_kink == tau_min :  return None

        # Build the kinks to be inserted if update is accepted
        N_i = data_struct[x][ik][1]                            # particles on i after kink
        N_j = data_struct[j][flat_min_idx_j][1] + 1             # particles on j after kink
        N_after_ira = N_j - 1                                   # particles on j after Ira
        new_kink_i = [tau_kink,N_i,(x,j)]
        new_kink_j = [tau_kink,N_j,(x,j)]
        ira_kink = [tau_max,N_after_ira,(j,j)]

        # Metropolis Sampling
        spaceshift_before_weight = 1
        # Accept
        if np.random.random() < spaceshift_before_weight:
            data_struct[x].insert(ik,new_kink_i)
            del data_struct[x][ik+1]
            if flat_min_idx_j == len(data_struct[j]) - 1:
                data_struct[j].append(new_kink_j)
                data_struct[j].append(ira_kink)
            else:
                data_struct[j].insert(flat_min_idx_j+1,new_kink_j)
                data_struct[j].insert(flat_min_idx_j+2,ira_kink)

            # Reindex ira
            ira_loc[0] = j
            ira_loc[1] = flat_min_idx_j+2
            # Reindex masha if necessary
            if j == mx and tau_2 > tau_1:
                masha_loc[1] += 2

            return None

        # Reject
        else : return None


    # --- Delete kink branch --- #
    else:
        # Deletion only possible if there's a kink preceding Ira
        is_delete_possible = False
        kink_before_ira = data_struct[x][ik-1]
        kink_source = kink_before_ira[2][0]
        kink_dest = kink_before_ira[2][1]
        if kink_source != kink_dest: # kinks with equal src and dest are not actual kinks. they are worm ends or initial values.
            is_delete_possible = True

        # Reject update if there was no kink preceding Ira
        if is_delete_possible == False: return None

        # To send Ira to j via delete_kink, the kink's source must also be j
        if kink_source != j: return None

        # Determine the tau of the kink preceding Ira. Will be used later for deletion.
        tau_kink = data_struct[x][ik-1][0]
        tau_kink_idx_i = ik - 1

        # Determine the kink idx in j of the kink
        for k in range(len(data_struct[j])):
            if data_struct[j][k][0] == tau_kink:
                tau_kink_idx_j = k
                break

        # The kink cannot be deleted if Ira goes another kink or wormend on site j
        if tau_kink_idx_j < len(data_struct[j]) - 1:
            if tau_1 > data_struct[j][tau_kink_idx_j+1][0]:
                return None

        # Build the Ira kink to be moved to j
        N_after_ira = data_struct[j][tau_kink_idx_j][1]
        ira_kink = [tau_1,N_after_ira,(j,j)]

        # Metropolis Sampling
        spaceshift_before_weight = 1
        # Accept
        if np.random.random() < spaceshift_before_weight:
            del data_struct[x][ik]
            del data_struct[x][ik-1]
            if tau_kink_idx_j == len(data_struct[j]) - 1: # kink precedes beta
                data_struct[j].append(ira_kink)
            else:
                data_struct[j].insert(tau_kink_idx_j+1,ira_kink)
            del data_struct[j][tau_kink_idx_j]

            # Reindex Ira
            ira_loc[0] = j
            ira_loc[1] = tau_kink_idx_j
            # Reindex Masha if necessary:
            if ix == mx and tau_2 > tau_1:
                masha_loc[1] -= 2

            return None

        # Reject
        else:
            return None

'----------------------------------------------------------------------------------'

def worm_spaceshift_before_old(data_struct,beta,ira_loc,masha_loc):

    # Update not possible if there's no worm
    if ira_loc == [] and masha_loc == [] : return None

    # Number of lattice sites
    L = len(data_struct)

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]
    tau_1 = data_struct[ix][ik][0]
    tau_2 = data_struct[mx][mk][0]

    # Randomly select the neighboring site at which to send Ira
    if np.random.random() < 0.5:
        j = ix + 1
        if j == L : j = 0      # PBC
    else:
        j = ix - 1
        if j == -1 : j = L - 1 # PBC

    # Randomly choose to send Ira to j via: a) insert kink b) delete kink
    if np.random.random() < 0.5:
        insert_kink = True
    else:
        insert_kink = False

    # --- Insert kink branch --- #
    if insert_kink == True:

        # Determine lower and upper bound in im. time at which to insert the kink
        tau_max = tau_1 # tau_max is at Ira's time

        # tau_min candidate on site i
        if len(data_struct[ix]) == 1: tau_min_i = 0
        else: tau_min_i = data_struct[ix][ik-1][0]

        # tau_min candidate on site j
        for k in range(len(data_struct[j])):
            if len(data_struct[j]) == 1:
                tau_min_j = 0
                flat_min_idx_j = 0
            else:
                if data_struct[j][k][0] > tau_max: break # want the kink with tau preceding tau_max
                tau_min_j = data_struct[j][k][0]
                flat_min_idx_j = k

        # Select the tau_min value between the i and j candidates
        if tau_min_i > tau_min_j : tau_min = tau_min_i
        else : tau_min = tau_min_j

        # Randomly choose the kink time between tau_min and tau_max
        tau_kink = tau_min + np.random.random()*(tau_max - tau_min)

        # Insertion not possible if the proposed kink time happens at the time of another kink
        if tau_kink == tau_min :  return None

        # Build the kinks to be inserted if update is accepted
        N_i = data_struct[ix][ik][1]                            # particles on i after kink
        N_j = data_struct[j][flat_min_idx_j][1] + 1             # particles on j after kink
        N_after_ira = N_j - 1                                   # particles on j after Ira
        new_kink_i = [tau_kink,N_i,(ix,j)]
        new_kink_j = [tau_kink,N_j,(ix,j)]
        ira_kink = [tau_max,N_after_ira,(j,j)]

        # Metropolis Sampling
        spaceshift_before_weight = 1
        # Accept
        if np.random.random() < spaceshift_before_weight:
            data_struct[ix].insert(ik,new_kink_i)
            del data_struct[ix][ik+1]
            if flat_min_idx_j == len(data_struct[j]) - 1:
                data_struct[j].append(new_kink_j)
                data_struct[j].append(ira_kink)
            else:
                data_struct[j].insert(flat_min_idx_j+1,new_kink_j)
                data_struct[j].insert(flat_min_idx_j+2,ira_kink)

            # Reindex ira
            ira_loc[0] = j
            ira_loc[1] = flat_min_idx_j+2
            # Reindex masha if necessary
            if j == mx and tau_2 > tau_1:
                masha_loc[1] += 2

            return None

        # Reject
        else : return None


    # --- Delete kink branch --- #
    else:
        # Deletion only possible if there's a kink preceding Ira
        is_delete_possible = False
        kink_before_ira = data_struct[ix][ik-1]
        kink_source = kink_before_ira[2][0]
        kink_dest = kink_before_ira[2][1]
        if kink_source != kink_dest: # kinks with equal src and dest are not actual kinks. they are worm ends or initial values.
            is_delete_possible = True

        # Reject update if there was no kink preceding Ira
        if is_delete_possible == False: return None

        # To send Ira to j via delete_kink, the kink's source must also be j
        if kink_source != j: return None

        # Determine the tau of the kink preceding Ira. Will be used later for deletion.
        tau_kink = data_struct[ix][ik-1][0]
        tau_kink_idx_i = ik - 1

        # Determine the kink idx in j of the kink
        for k in range(len(data_struct[j])):
            if data_struct[j][k][0] == tau_kink:
                tau_kink_idx_j = k
                break

        # The kink cannot be deleted if Ira goes another kink or wormend on site j
        if tau_kink_idx_j < len(data_struct[j]) - 1:
            if tau_1 > data_struct[j][tau_kink_idx_j+1][0]:
                return None

        # Build the Ira kink to be moved to j
        N_after_ira = data_struct[j][tau_kink_idx_j][1]
        ira_kink = [tau_1,N_after_ira,(j,j)]

        # Metropolis Sampling
        spaceshift_before_weight = 1
        # Accept
        if np.random.random() < spaceshift_before_weight:
            del data_struct[ix][ik]
            del data_struct[ix][ik-1]
            if tau_kink_idx_j == len(data_struct[j]) - 1: # kink precedes beta
                data_struct[j].append(ira_kink)
            else:
                data_struct[j].insert(tau_kink_idx_j+1,ira_kink)
            del data_struct[j][tau_kink_idx_j]

            # Reindex Ira
            ira_loc[0] = j
            ira_loc[1] = tau_kink_idx_j
            # Reindex Masha if necessary:
            if ix == mx and tau_2 > tau_1:
                masha_loc[1] -= 2

            return None

        # Reject
        else:
            return None

'----------------------------------------------------------------------------------'

def worm_spaceshift_after(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Check if there's a worm
    if is_worm_present[0] == False : return None

    # Number of lattice sites
    L = len(data_struct)

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]
    tau_1 = data_struct[ix][ik][0]
    tau_2 = data_struct[mx][mk][0]

    # Randomly select the neighboring site at which to send Ira
    if np.random.random() < 0.5:
        j = ix + 1
        if j == L : j = 0      # PBC
    else:
        j = ix - 1
        if j == -1 : j = L - 1 # PBC

    # Determine the lower and upper bounds for the kink to be inserted
    tau_min = tau_1

    # Randomly choose to send Ira to j via: a) insert kink b) delete kink
    if np.random.random() < 0.5:
        insert_kink = True
    else:
        insert_kink = False

    # --- Insert kink branch --- #
    if insert_kink == True:

        # tau_max candidate on site i
        if ik == len(data_struct[ix]) - 1: tau_max_i = beta
        else: tau_max_i = data_struct[ix][ik+1][0]

        # tau_max candidate on site j
        for k in range(len(data_struct[j])):
            if len(data_struct[j]) == 1 :
                tau_max_j = beta
                flat_min_idx_j = 0
            else:
                tau_max_j = data_struct[j][k][0]
                flat_min_idx_j = k-1  # Index of the maximum value of the flat interval in j
                if data_struct[j][k][0] > tau_min: break
                if k == len(data_struct[j]) - 1 :
                    tau_max_j = beta
                    flat_min_idx_j = len(data_struct[j]) - 1 # flat worldline or flat_min preceding beta

        # Select the tau_max value between the i and j candidates
        if tau_max_i < tau_max_j : tau_max = tau_max_i
        else: tau_max = tau_max_j

        # Suggest the kink time
        tau_kink = tau_min + np.random.random()*(tau_max - tau_min)

        # Insertion not possible if the proposed kink time happens at the time of another kink
        if tau_kink == tau_min : return None

        # Check if Ira can even be sent to site j (need particles there first)
        if data_struct[j][flat_min_idx_j][1] == 0 : return None

        # Build the kinks to be inserted if update is accepted
        N_i = data_struct[ix][ik][1] # after the i kink, N is the same as what was originally post ira
        N_after_ira = data_struct[j][flat_min_idx_j][1] - 1
        N_j = N_after_ira + 1 # particles on j after the kink
        new_kink_i = [tau_kink,N_i,(ix,j)]
        ira_kink = [tau_min,N_after_ira,(j,j)]
        new_kink_j = [tau_kink,N_j,(ix,j)]

        # Metropolis Sampling
        spaceshift_after_weight = 1
        # Accept
        if np.random.random() < spaceshift_after_weight:
            data_struct[ix].insert(ik,new_kink_i)
            del data_struct[ix][ik+1]
            if flat_min_idx_j == len(data_struct[j]) - 1: # takes care of data_struct w/ 1 element or flat_min at last interval
                data_struct[j].append(ira_kink)
                data_struct[j].append(new_kink_j)
            else:
                data_struct[j].insert(flat_min_idx_j+1,ira_kink)
                data_struct[j].insert(flat_min_idx_j+2,new_kink_j)

            # Reindex ira
            ira_loc[0] = j
            ira_loc[1] = flat_min_idx_j+1
            # Reindex masha if necessary
            if j == mx and tau_2 > tau_1:
                masha_loc[1] += 2

            return None

        # Reject
        else : return None

    # --- Delete kink branch --- #
    else:
        # Deletion only possible if there's a kink succeeding Ira
        if ik == len(data_struct[ix]) - 1 : return None # this means that ira is the last "kink" before beta
        is_delete_possible = False
        kink_after_ira = data_struct[ix][ik+1]
        kink_source = kink_after_ira[2][0]
        kink_dest = kink_after_ira[2][1]
        if kink_source != kink_dest: # kinks with equal src and dest are not actual kinks. they are worm ends or initial values.
            is_delete_possible = True

        # Reject update if there is no kink succeeding Ira
        if is_delete_possible == False: return None

        # To send Ira to j via delete_kink, the kink's source must also be j
        if kink_source != j: return None

        # Determine the tau of the kink following Ira. Will be used later for deletion.
        tau_kink = data_struct[ix][ik+1][0]
        tau_kink_idx_i = ik + 1

        # Determine the kink idx in j of the kink
        for k in range(len(data_struct[j])):
            if data_struct[j][k][0] == tau_kink:
                tau_kink_idx_j = k
                break

        # The kink cannot be deleted if Ira there will be another kink between Ira and the one to be deleted in j
        if tau_kink_idx_j > 1:
            if tau_1 < data_struct[j][tau_kink_idx_j-1][0]:
                return None

        # Build the Ira kink to be moved to j
        N_after_ira = data_struct[j][tau_kink_idx_j][1]
        ira_kink = [tau_1,N_after_ira,(j,j)]

        # Metropolis Sampling
        spaceshift_after_weight = 1
        # Accept
        if np.random.random() < spaceshift_after_weight:
            del data_struct[ix][ik+1]
            del data_struct[ix][ik]
            data_struct[j].insert(tau_kink_idx_j,ira_kink)
            del data_struct[j][tau_kink_idx_j+1]

            # Reindex Ira
            ira_loc[0] = j
            ira_loc[1] = tau_kink_idx_j
            # Reindex Masha if necessary:
            if ix == mx and tau_2 > tau_1:
                masha_loc[1] -= 2

            return None

        # Reject
        else:
            return None

'----------------------------------------------------------------------------------'

# Visualize worldline configurations for Lattice Path Integral Monte Carlo (PIMC)

def view_worldlines(data_struct,beta,figure_name=None):
    import matplotlib.pyplot as plt

    # Set initial configuration
    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    # N = 0
    # for site_idx in range(L):
    #     N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Stores the initial particle configuration
    x = []
    for i in range(L):
        x.append(data_struct[i][0][1])

    # Spread tau,N,dir data to different arrays (hopefully will help with visualization???)
    # Store kink times, particles after kinks, and kink directions for each site i
    tau_list = []
    N_list = []
    dirs_list = []
    for i in range(L):
        tau_i = []
        N_i = []
        dirs_i = []
        events = len(data_struct[i]) # Number of kinks at site i (includes initial config.)
        for e in range(events):
            tau_i.append(data_struct[i][e][0])
            N_i.append(data_struct[i][e][1])
            dirs_i.append(data_struct[i][e][2])
        tau_list.append(tau_i)
        N_list.append(N_i)
        dirs_list.append(dirs_i)

    # Initialize figure
    plt.figure()

    # Plot (loop over sites, then loop over kinks)
    for i in range(L):
        L_tau = len(tau_list[i]) # Number of "kinks" on site i including initial config
        for j in range(L_tau):
            tau_initial = tau_list[i][j]
            if j + 1 < L_tau: # To avoid running out of range
                tau_final = tau_list[i][j+1]
            else:
                tau_final = beta

            n = N_list[i][j] # Occupation of the i_th site at the j_th "kink"
            if n == 0: ls,lw = ':',1
            elif n == 1: ls,lw = '-',1
            elif n == 2: ls,lw = '-',3
            elif n == 3: ls,lw = '-',5.5
            src_site = dirs_list[i][j][0]    # Index of source site
            dest_site = dirs_list[i][j][1]  # Index of destination site

            # Draw flat regions
            plt.vlines(i,tau_initial,tau_final,linestyle=ls,linewidth=lw)
            # Draw worm ends
            if src_site == dest_site and tau_initial != 0:
                #plt.plot(i,tau_initial,marker='_',ms=5,lw=5)
                plt.hlines(tau_initial,i-0.06,i+0.06,lw=1)

            # Wrap around the spatial direction axis if kink connects the first and last sites
            if (src_site == 0 and dest_site == L-1):
                plt.hlines(tau_list[i][j],-0.5,0,linewidth=1)
                plt.hlines(tau_list[i][j],L-1,L-1+0.5,linewidth=1)

            elif (src_site == L-1 and dest_site == 0):
                plt.hlines(tau_list[i][j],-0.5,0,linewidth=1)
                plt.hlines(tau_list[i][j],L-1,L-1+0.5,linewidth=1)

            else:
                plt.hlines(tau_list[i][j],src_site,dest_site,linewidth=1)

    plt.xticks(range(0,L))
    plt.xlim(-0.5,L-1+0.5)
    plt.ylim(0,1)
    plt.tick_params(axis='y',which='both',left=False,right=False)
    plt.tick_params(axis='x',which='both',top=False,bottom=False)
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\tau/\beta$")
    if figure_name != None:
        plt.savefig(figure_name)
    plt.show()
    #plt.close()

    return None

# -------------------------------------------------------------------------------------- #


# ----- Main ----- #
# # Test updates
# data_struct = [ [[0,1,(0,0)],[0.25,2,(1,0)],[0.5,1,(0,2)],[0.75,0,(0,1)]],
#                 [[0,1,(1,1)],[0.25,0,(1,0)],[0.75,1,(0,1)]],
#                 [[0,1,(2,2)],[0.5,2,(0,2)]] ]
# data_struct = [ [[0,1,(0,0)]],
#                 [[0,1,(1,1)]],
#                 [[0,1,(2,2)]] ]

# #L = int(1E+05)
# #N = L # unit filling
# #x = random_boson_config(L,N)
# #data_struct = create_data_struct(x)
# #print(data_struct)

# beta = 1
# is_worm_present = [False] # made flag a list so it can be passed "by reference"
# ira_loc = []    # If there's a worm present, these will store
# masha_loc = []  # the site_idx and tau_idx "by reference"

# M = int(1E+03)
# ctr00, ctr01, ctr02, ctr03, ctr04 = 0, 0, 0, 0, 0
# # Plot original configuration
# file_name = "worldlines_0%d_00.pdf"%ctr00
# #view_worldlines(data_struct,beta,file_name)
# print(" --- Progress --- ")
# for m in range(M):
#     # Test insert/delete worm and plot it
#     worm(data_struct,beta,ira_loc,masha_loc)
#     file_name = "worldlines_0%d_01.pdf"%ctr01
#     #view_worldlines(data_struct,beta,file_name)
#     ctr01 += 1

#     # Test timeshift and plot it
#     worm_timeshift(data_struct,beta,is_worm_present,ira_loc,masha_loc)
#     file_name = "worldlines_0%d_02.pdf"%ctr02
#     #view_worldlines(data_struct,beta,file_name)
#     ctr02 += 1

#     # Test spaceshift_before_insert and plot it
#     worm_spaceshift_before(data_struct,beta,is_worm_present,ira_loc,masha_loc)
#     file_name = "worldlines_0%d_03.pdf"%ctr03
#     #view_worldlines(data_struct,beta,file_name)
#     ctr03 += 1

#     # Test spaceshift_after and plot it
#     worm_spaceshift_after(data_struct,beta,is_worm_present,ira_loc,masha_loc)
#     file_name = "worldlines_0%d_04.pdf"%ctr04
#     #view_worldlines(data_struct,beta,file_name)
#     ctr04 += 1

#     # Test gsworm_insert
#     gsworm_insert(data_struct,beta,is_worm_present,ira_loc,masha_loc)


#     # Progress
#     print("%.2f%%"%((m+1)/M*100))


    ##############################################
    # Forces worm instead of antiworm
    #if tau_2 > tau_1:
    #    tmp = tau_1
    #   tau_1 = tau_2
    #    tau_2 = tmp

    # Forces insert antiworm instead of worm
    #if tau_1 > tau_2:
    #    tmp = tau_2
    #    tau_2 = tau_1
    #    tau_1 = tmp
    ##############################################				                      
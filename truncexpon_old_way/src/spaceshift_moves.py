import numpy as np
import bisect
import matplotlib.pyplot as plt
from scipy.stats import truncexpon

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
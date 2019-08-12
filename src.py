# Functions to be used in main file (lattice_pimc.py)
import numpy as np
import bisect
import matplotlib.pyplot as plt

def random_boson_config(L,N):
    '''Generates a random configuration of N bosons in a 1D lattice of size L'''

    psi = np.zeros(L,dtype=int) # Stores the random configuration of bosons
    for i in range(N):
        r = np.random.randint(L)
        psi[r] += 1

    return psi

'----------------------------------------------------------------------------------'

def worm_insert(data_struct, beta, is_worm_present,ira_loc = [], masha_loc = []):
    'Accept/reject worm head AND tail insertion'

    # Reject update if there is a worm present
    if is_worm_present[0] == True :
        return None

    # Number of lattice sites
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm
    i = np.random.randint(L)

    # Randomly select a flat tau interval at which to possibly insert worm
    n_flats = len(data_struct[i])
    r = np.random.randint(n_flats) # Index of lower bound time of worm
    tau_min = data_struct[i][r][0]
    if r == n_flats - 1 : tau_max = beta  # Avoids running out of range
    else : tau_max = data_struct[i][r+1][0]

    # Randomly select imaginary times at which worm ends will be inserted
    tau_1 = tau_min + np.random.random()*(tau_max - tau_min) # Ira (anihilation)
    tau_2 = tau_min + np.random.random()*(tau_max - tau_min) # Masha (creation)
    if tau_1 == tau_2 :
        return None

    ##############################################
    # Forces worm instead of antiworm
    #if tau_2 > tau_1:
    #    tmp = tau_1
    #    tau_1 = tau_2
    #    tau_2 = tmp

    # Forces insert antiworm instead of worm
    if tau_1 > tau_2:
        tmp = tau_2
        tau_2 = tau_1
        tau_1 = tmp
    ##############################################

    # Propose to insert worm (Metropolis Sampling)
    weight_insert = 1

    # Accept
    if np.random.random() < weight_insert:
        # Case 1: Ira first, then Masha
        if tau_1 < tau_2 :
            #print("Ira first , Masha second")
            n = data_struct[i][r][1] - 1
            if n == -1 :
                return None # Reject if there were no particles to destroy
            # Insert worm 'kink' here
            if r == n_flats - 1:
                data_struct[i].append([tau_1,n,(i,i)])
                data_struct[i].append([tau_2,n+1,(i,i)])
            else:
                data_struct[i].insert(r+1,[tau_1,n,(i,i)])
                data_struct[i].insert(r+2,[tau_2,n+1,(i,i)])

            # Save ira and masha locations (site_idx, tau_idx)
            ira_loc.extend([i,r+1])
            masha_loc.extend([i,r+2])

        # Case 2: Masha first, then Ira
        else:
            #print("Masha first , Ira second")
            n = data_struct[i][r][1] + 1
            # Insert worm kink here
            if r == n_flats - 1:
                data_struct[i].append([tau_2,n,(i,i)])
                data_struct[i].append([tau_1,n-1,(i,i)])
            else:
                data_struct[i].insert(r+1,[tau_2,n,(i,i)])
                data_struct[i].insert(r+2,[tau_1,n-1,(i,i)])

            # Save ira and masha locations (site_idx, tau_idx)
            ira_loc.extend([i,r+2])
            masha_loc.extend([i,r+1])

        # Flag indicationg a worm is now present
        is_worm_present[0] = True

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def worm_timeshift(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Reject update if there is no worm present
    if is_worm_present[0] == False : return None

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]

    #print("ira site before/after timeshift: ")
    #print(data_struct[ix])
    # Retrieve the actual times of Ira and Masha
    tau_1 = data_struct[ix][ik][0]
    tau_2 = data_struct[mx][mk][0]

    # Determine the lower and upper bounds of Ira for the timeshift
    tau_max_idx = ik + 1
    tau_min_idx = ik - 1

    # Get tau_max
    if tau_max_idx == len(data_struct[ix]):
        tau_max = beta  # This covers the case when there's no kinks after ira
    else:
        tau_max = data_struct[ix][tau_max_idx][0] #actual times
    # Get tau_min
    if tau_min_idx == 0:
        tau_min = 0
    else:
        tau_min = data_struct[ix][tau_min_idx][0]

    # Randomly propose a time for Ira between tau_min and tau_max
    tau_1 = tau_min + np.random.random()*(tau_max - tau_min)

    # NOTE: Can Ira shift back all the way to Masha? If so, the worm will be destroyed
    # and technically, only the worm_delete update can do that

    # Delete the worm if the end is shifted to the location of the other
    if tau_1 == tau_2 and ix == mx:
        worm_delete(data_struct,beta,is_worm_present,ira_loc,masha_loc)

    # Metropolis sampling
    # Accept
    weight_timeshift = 1
    if np.random.random() < weight_timeshift:
        data_struct[ix][ik][0] = tau_1 # Modify Ira time
        #print(data_struct[ix])
        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def worm_spaceshift_before(data_struct,beta,is_worm_present,ira_loc,masha_loc):

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

    # Randomly choose to do the insert or delete kink part of this update
    if np.random.random() < 0.5:
        insert = True
    else:
        insert = False

    # --- Insert kink branch --- #
    if insert == True:
        # Randomly select the neighboring site at which to send Ira
        if np.random.random() < 0.5:
            j = ix + 1
            if j == L : j = 0      # PBC
        else:
            j = ix - 1
            if j == -1 : j = L - 1 # PBC

        print("Welcome to the insert branch of ss_before!")
        print("--- Initial ---")
        print("i=0 : ", data_struct[0])
        print("i=1 : ", data_struct[1])
        print("i=2 : ", data_struct[2])
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

            print("--- Final ---")
            print("i=0 : ", data_struct[0])
            print("i=1 : ", data_struct[1])
            print("i=2 : ", data_struct[2])
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

        # If kink is deleted, Ira will be sent to the src_site of its preceding kink
        j = data_struct[ix][ik-1][2][0]

        print("Welcome to the delete branch of ss_before!")
        print("--- Initial ---")
        print("i=0 : ", data_struct[0])
        print("i=1 : ", data_struct[1])
        print("i=2 : ", data_struct[2])
        # Determine the tau of the kink preceding Ira. Will be used later for deletion.
        tau_kink = data_struct[ix][ik-1][0]
        tau_kink_idx_i = ik - 1

        # Determine the kink idx in j of the kink
        #print(data_struct[j])
        for k in range(len(data_struct[j])):
            print(data_struct[j][k][0],tau_kink,data_struct[j][k][0]==tau_kink)
            if data_struct[j][k][0] == tau_kink:
                tau_kink_idx_j = k
                break

        # The kink cannot be deleted if it interferes with another kink or wormend on site j
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

            print("--- Final ---")
            print("i=0 : ", data_struct[0])
            print("i=1 : ", data_struct[1])
            print("i=2 : ", data_struct[2])
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

    # Check if there's a kink before Ira to see if the delete part of this update is possible
    #is_delete_possible = False
    #kink_before_ira = data_struct[ix][ik-1]
    #kink_source = kink_before_ira[2][0]
    #kink_dest = kink_before_ira[2][1]
    #if kink_source != kink_dest: # delete only possible if kink before is an actual kink and not worm end or initial data_struct
    #    is_delete_possible = True

    # Flip a coin to decide if to attempt a delete kink or insert kink (before Ira)
    #insert = True
    #if is_delete_possible:
    #    if np.random.random() < 0.5:
    #        insert = False

    # Randomly select the neighboring site at which to send Ira
    if np.random.random() < 0.5:
        j = ix + 1
        if j == L : j = 0      # PBC
    else:
        j = ix - 1
        if j == -1 : j = L - 1 # PBC

    # Determine the lower and upper bounds for the kink to be inserted
    tau_min = tau_1

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

'----------------------------------------------------------------------------------'

def worm_delete(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Check if there's a worm
    if is_worm_present[0] == False : return None

    # Only delete if worm ends are on the same site and on the same flat interval
    if ira_loc[0] != masha_loc[0] or abs(ira_loc[0]-masha_loc[0]) != 1: return None

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    ix = ira_loc[0]
    ik = ira_loc[1]
    mx = masha_loc[0]
    mk = masha_loc[1]

    # Metropolis sampling
    # Accept
    weight_delete = 1
    if np.random.random() < weight_delete:
        # Delete the worm ends
        if ik > mk : # Ira ahead
            del data_struct[ix][ik] # Deletes ira
            del data_struct[mx][mk] # Deletes masha
        else: # Ira behind
            del data_struct[mx][mk] # Deletes masha
            del data_struct[ix][ik] # Deletes ira

        # Update the worm flag and ira_loc,masha_loc
        is_worm_present[0] = False
        del ira_loc[:]
        del masha_loc[:]

        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def view_worldlines(data_struct,beta,figure_name):

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
    plt.savefig(file_name)
    #plt.show()
    plt.close()

    return None

# Test updates
data_struct = [ [[0,1,(0,0)],[0.25,2,(1,0)],[0.5,1,(0,2)],[0.75,0,(0,1)]],
                [[0,1,(1,1)],[0.25,0,(1,0)],[0.75,1,(0,1)]],
                [[0,1,(2,2)],[0.5,2,(0,2)]] ]
data_struct = [ [[0,1,(0,0)]],
                [[0,1,(1,1)]],
                [[0,1,(2,2)]] ]
beta = 1
is_worm_present = [False] # made flag a list so it can be passed "by reference"
ira_loc = []    # If there's a worm present, these will store
masha_loc = []  # the site_idx and tau_idx "by reference"

M = 50
ctr00, ctr01, ctr02, ctr03, ctr04, ctr05 = 0, 0, 0, 0, 0, 0
# Plot original configuration
file_name = "worldlines_0%d_00.pdf"%ctr00
view_worldlines(data_struct,beta,file_name)
for n in range(M):

    # Test insert and plot it
    worm_insert(data_struct,beta,is_worm_present,ira_loc,masha_loc)
    file_name = "worldlines_0%d_01.pdf"%ctr01
    view_worldlines(data_struct,beta,file_name)
    ctr01 += 1

    # Test timeshift and plot it
    worm_timeshift(data_struct,beta,is_worm_present,ira_loc,masha_loc)
    file_name = "worldlines_0%d_02.pdf"%ctr02
    view_worldlines(data_struct,beta,file_name)
    ctr02 += 1

    # Test spaceshift_before_insert and plot it
    worm_spaceshift_before(data_struct,beta,is_worm_present,ira_loc,masha_loc)
    file_name = "worldlines_0%d_03.pdf"%ctr03
    view_worldlines(data_struct,beta,file_name)
    ctr03 += 1

    if data_struct[ira_loc[0]] != sorted(data_struct[ira_loc[0]]):
        print("ss_before ruined the sort!!!!")
        break

    # Test spaceshift_after and plot it
    worm_spaceshift_after(data_struct,beta,is_worm_present,ira_loc,masha_loc)
    file_name = "worldlines_0%d_04.pdf"%ctr04
    view_worldlines(data_struct,beta,file_name)
    ctr04 += 1

    if data_struct[ira_loc[0]] != sorted(data_struct[ira_loc[0]]):
        print("ss_after ruined the sort!!!!")
        break

# Test delete and plot
ctr05 += ctr04
worm_delete(data_struct,beta,is_worm_present,ira_loc,masha_loc)
file_name = "worldlines_0%d_05.pdf"%ctr05
view_worldlines(data_struct,beta,file_name)
#ctr04 += 1

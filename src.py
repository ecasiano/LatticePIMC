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

    # Number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Randomly select a lattice site i on which to insert a worm
    i = np.random.randint(L)

    print("\nOriginal : ")
    print(data_struct[i])
#
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
    # FOR DEBUGGING ONLY!!!!!!!!!!!!!! # FORCES INSERT WORM INSTEAD OF ANTIWORM
    if tau_1 < tau_2:
        tmp = tau_1
        tau_1 = tau_2
        tau_2 = tmp
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
                print("Ira could not destroy at the proposed time & location :( ")
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

        print("Insert: " )
        print(data_struct[i])

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def worm_timeshift(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Reject update if there is no worm present
    if is_worm_present[0] == False : return None

    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    tau_1_siteidx = ira_loc[0]
    tau_1_tauidx = ira_loc[1]
    tau_2_siteidx = masha_loc[0]
    tau_2_tauidx = masha_loc[1]

    # Retrieve the actual times of ira, masha, and the closest kinks (in im. time)
    tau_1 = data_struct[tau_1_siteidx][tau_1_tauidx][0]
    tau_2 = data_struct[tau_2_siteidx][tau_2_tauidx][0]

    # Similar to above, but find the locs of closest kinks to ira and masha in im time

    # Case 1: Ira is ahead in imaginary time
    if tau_1 > tau_2:
        past_kink_tauidx = tau_2_tauidx - 1                # idx of kink before masha
        future_kink_tauidx = tau_1_tauidx + 1              # idx of kink after ira
        if future_kink_tauidx == len(data_struct[tau_1_siteidx]):
            tau_future = beta  # This covers the case when there's no kinks after ira
        else:
            tau_future = data_struct[tau_1_siteidx][future_kink_tauidx][0] #actual times
        tau_past = data_struct[tau_2_siteidx][past_kink_tauidx][0]

        tau_1 = tau_2 + np.random.random()*(tau_future - tau_2)  # Ira's proposed time

    # Case 2: Ira is behind in imaginary time
    else:
        past_kink_tauidx = tau_1_tauidx - 1                 # idx of kink before ira
        future_kink_tauidx = tau_2_tauidx + 1               # idx of kink after masha
        if future_kink_tauidx == len(data_struct[tau_2_siteidx]):
            tau_future = beta  # This covers the case when there's no kinks after ira
        else:
            tau_future = data_struct[tau_2_siteidx][future_kink_tauidx][0] # actual times
        tau_past = data_struct[tau_1_siteidx][past_kink_tauidx][0]

        tau_1 = tau_past + np.random.random()*(tau_2 - tau_past)

    # Metropolis sampling
    # Accept
    weight_timeshift = 1
    if np.random.random() < weight_timeshift:
        data_struct[tau_1_siteidx][tau_1_tauidx][0] = tau_1
        print("Timeshift: ")
        print(data_struct[tau_1_siteidx])
        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def worm_spaceshift_before(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Check if there's a worm
    if is_worm_present[0] == False : return None

    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    tau_1_siteidx = ira_loc[0]
    tau_1_tauidx = ira_loc[1]
    tau_2_siteidx = masha_loc[0]
    tau_2_tauidx = masha_loc[1]

    # Randomly select the neighboring site at which to send Ira
    if np.random.random() < 0.5:
        j = tau_1_siteidx + 1
        if j == L : j = 0      # PBC
    else:
        j = tau_1_siteidx - 1
        if j == -1 : j = L - 1 # PBC

    # If Ira is not returning to its previous site, a kink is created
    # ira_previous_site = data_struct[tau_1_siteidx][i]
    # if data_struct[tau_1_siteidx][tau_1_tauidx-1][0]

    # Determine lower and upper bound in im time at which to insert the kink
    tau_max = data_struct[tau_1_siteidx][tau_1_tauidx][0] # tau_max is at Ira's time
    L_tau = len(data_struct[j]) # Number of kinks on the jth site, including initial kink
    tau_min = 0 # Initialize value for the lower bound of tau
    for k in range(L_tau):
        if data_struct[j][k][0] < tau_max:
            tau_min = data_struct[j][k][0]
            tau_min_idx = k # This tau_min is on the j_th site
    # tau_min will be the largest of the taus for the previous kinks in i and j
    print("k = ", k)
    if data_struct[tau_1_siteidx][tau_1_tauidx-1][0] > tau_min:
        tau_min = data_struct[tau_1_siteidx][tau_1_tauidx-1][0]

    tau_kink_idx = tau_min_idx+1 # Index in im time at the jth site where to insert the kink
    # Randomly choose the kink time between tau_min and tau_max
    tau_kink = tau_min + np.random.random()*(tau_max - tau_min)

    # Build the "tau-tuple" ([tau_kink, N_after kink, (src_site,dest_site)]) to be added to j
    N_after_kink = data_struct[j][tau_min_idx][1] + 1
    tau_tuple_j = [tau_kink,N_after_kink,(tau_1_siteidx,j)]

    # Build the "tau-tuple" ([tau_kink, N_after kink, (src_site,dest_site)]) to be added to i
    N_after_kink_i = data_struct[tau_1_siteidx][tau_1_tauidx][1]
    tau_tuple_i = [tau_kink,N_after_kink_i,(tau_1_siteidx,j)]

    # Metropolis Sampling
    spaceshift_before_weight = 1
    # Accept
    if np.random.random() < spaceshift_before_weight:
        # Insert kink in j
        if tau_min_idx+1 == L_tau:
            data_struct[j].append(tau_tuple_j)
        else:
            data_struct[j].insert(tau_min_idx+1,tau_tuple_j)

        # Insert ira in j
        if tau_min_idx+2 == L_tau+1:
            data_struct[j].append([tau_max,N_after_kink-1,(j,j)])
        else:
            data_struct[j].insert(tau_min_idx+2,[tau_max,N_after_kink-1,(j,j)])

        # Modify the ira kink in i
        data_struct[tau_1_siteidx][tau_1_tauidx] = tau_tuple_i

        # Update the location of Ira and Masha's tau index (deleting kink shifts original)
        ira_loc[0] = j
        ira_loc[1] = tau_min_idx+2
        if ira_loc[1] < masha_loc[1]: # Case where Ira goes before Masha in imaginary time
            masha_loc[1] = masha_loc[1] - 1

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def worm_spaceshift_after(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Check if there's a worm
    if is_worm_present[0] == False : return None

    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    tau_1_siteidx = ira_loc[0]
    tau_1_tauidx = ira_loc[1]
    tau_2_siteidx = masha_loc[0]
    tau_2_tauidx = masha_loc[1]

    # Randomly select the neighboring site at which to send Ira
    if np.random.random() < 0.5:
        j = tau_1_siteidx + 1
        if j == L : j = 0      # PBC
    else:
        j = tau_1_siteidx - 1
        if j == -1 : j = L - 1 # PBC

    #

    return None

'----------------------------------------------------------------------------------'

def worm_delete(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Check if there's a worm
    if is_worm_present[0] == False : return None

    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    tau_1_siteidx = ira_loc[0]
    tau_1_tauidx = ira_loc[1]
    tau_2_siteidx = masha_loc[0]
    tau_2_tauidx = masha_loc[1]

    # Metropolis sampling
    # Accept
    weight_delete = 1
    if np.random.random() < weight_delete:
        # Delete the worm ends
        if tau_1_tauidx > tau_2_tauidx : # Ira ahead
            del data_struct[tau_1_siteidx][tau_1_tauidx] # Deletes ira
            del data_struct[tau_2_siteidx][tau_2_tauidx] # Deletes masha
        else: # Ira behind
            del data_struct[tau_2_siteidx][tau_2_tauidx] # Deletes masha
            del data_struct[tau_1_siteidx][tau_1_tauidx] # Deletes ira

        # Delete the kinks created by spaceshifts
        for i in range(len(data_struct)):
            kinks_to_delete = [] # stores the indices of the kinks that shall be deleted
            for k in range(len(data_struct[i])):
                if data_struct[i][k][2][0] !=  data_struct[i][k][2][1]:
                    kinks_to_delete.append(k)
            kinks_to_delete = kinks_to_delete[::-1] # reverse to avoid index shift after delete
            for k in kinks_to_delete:
                del data_struct[i][k]

        print("Delete: ")
        print(data_struct[tau_1_siteidx])

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
                [[0,0,(1,1)]],
                [[0,1,(2,2)]] ]
beta = 1
is_worm_present = [False] # made flag a list so it can be passed "by reference"
ira_loc = []    # If there's a worm present, these will store
masha_loc = []  # the site_idx and tau_idx "by reference"

N = 5
ctr00, ctr01, ctr02, ctr03, ctr04 = 0, 0, 0, 0, 0
# Plot original configuration
file_name = "worldlines_0%d_00.pdf"%ctr00
view_worldlines(data_struct,beta,file_name)
for n in range(N):

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

    # Test spaceshift_before and plot it
    worm_spaceshift_before(data_struct,beta,is_worm_present,ira_loc,masha_loc)
    file_name = "worldlines_0%d_03.pdf"%ctr03
    view_worldlines(data_struct,beta,file_name)
    ctr03 += 1

    # Test delete and plot
    worm_delete(data_struct,beta,is_worm_present,ira_loc,masha_loc)
    file_name = "worldlines_0%d_04.pdf"%ctr04
    view_worldlines(data_struct,beta,file_name)
    ctr04 += 1

'----------------------------------------------------------------------------------'

def particle_jump(data_struct, beta):
    'Proposes a particle jump betweem neighboring sites and accepts/rejects via Metropolis'
    'data_struct = [[(tau, N after tau, id),(tau, N after tau, id)],[(),(),()],...]'

    # Number of lattice sites
    L = len(data_struct)

    # Count the number of total particles in the lattice
    N = 0
    for site_idx in range(L):
        N += data_struct[site_idx][-1][1] # ilist[taulist[list[(tau,N,jump_dir)]]]

    #print("TOTAL N: ",N)

    # Randomly select a lattice site i from which particle will jump
    i = np.random.randint(L)

    # Randomly select imaginary time in [0,beta) at which particle will jump
    tau_jump = beta*np.random.random()
    if tau_jump < data_struct[i][-1][0] : return None # TEMPORARY to check if this was the problem (it is)

    # Randomly select whether proposed insert kink update is to right or left
    r = np.random.random()
    if r < 0.5: jump_dir = -1 # 'left'
    else: jump_dir = 1 # 'right'

    # Set the index of the site where the particle on i will jump to
    j = i + jump_dir
    if j == L : j = 0      # Periodic Boundary Conditions
    if j == -1 : j = L-1

    # Number of particles in i,j before the jump

    # Metropolis sampling
    weight_jump = 0.5
    r = np.random.random()
    if r < weight_jump: # Accept particle jump from sites i -> j
        # Get indices of where proposed tau will fit in the data structure
        taui = bisect.bisect_left(data_struct[i],[tau_jump,-326,(i,j)]) # will insert wrt tau_jump
        tauj = bisect.bisect_left(data_struct[j],[tau_jump,-326,(i,j)])
        # Reject update if there is no particle on i at time tau
        if data_struct[i][taui-1][1] == 0 : return None
        # Number of particles before the jump
        ni = data_struct[i][taui-1][1]
        nj = data_struct[j][tauj-1][1]
        data_struct[i].insert(taui,[tau_jump,ni-1,(i,j)])
        data_struct[j].insert(tauj,[tau_jump,nj+1,(i,j)])

    else: "Reject update"

    return None

'----------------------------------------------------------------------------------'

def particle_jump_backup(data_struct, beta):
    'Proposes a particle jump betweem neighboring sites and accepts/rejects via Metropolis'
    'data_struct = [[(tau, N after tau, id),(tau, N after tau, id)],[(),(),()],...]'

    # Number of lattice sites
    L = len(data_struct)

    # Count the number of total particles in the lattice
    N = 0
    for site in range(L):
        N += data_struct[site][0][1] # ilist[taulist[list[(tau,N,jump_dir)]]]

    # Randomly select a lattice site i from which particle will jump
    i = np.random.randint(L)

    # Randomly select imaginary time in [0,beta) at which particle will jump
    tau_jump = beta*np.random.random()

    # Reject update if there is no particle on i at time tau
    if data_struct[i][-1][1] == 0 : return None

    # Randomly select whether proposed insert kink update is to right or left
    r = np.random.random()
    if r < 0.5: jump_dir = -1 # 'left'
    else: jump_dir = 1 # 'right'

    # Set the index of the site where the particle on i will jump to
    j = i + jump_dir
    if j == L : j = 0      # Periodic Boundary Conditions
    if j == -1 : j = L-1

    # Metropolis sampling
    weight = 0.6
    r = np.random.random()
    if r < 0.5:
        # Accept particle jump from sites i -> j
        ni = data_struct[i][-1][1] - 1 # particles on i after inserting kink
        nj = data_struct[j][-1][1] + 1 # particles on j after inserting kink
        data_struct[i].append([tau_jump,ni,(i,j)]) # (i,j) = ()source site, dest site)
        data_struct[j].append([tau_jump,nj,(i,j)])


    else: "Reject update"

    return None

'----------------------------------------------------------------------------------'


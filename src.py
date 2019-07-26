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
    if is_worm_present[0] == True : return None

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

    # Randomly select a flat tau interval at which to possibly insert worm
    n_flats = len(data_struct[i])
    r = np.random.randint(n_flats) # Index of lower bound time of worm
    tau_min = data_struct[i][r][0]
    if r == n_flats - 1 : tau_max = beta  # Avoids running out of range
    else : tau_max = data_struct[i][r+1][0]

    # Randomly select imaginary times at which worm ends will be inserted
    tau_1 = tau_min + np.random.random()*(tau_max - tau_min) # Ira (anihilation)
    tau_2 = tau_min + np.random.random()*(tau_max - tau_min) # Masha (creation)
    if tau_1 == tau_2 : return None

    # Propose to insert worm (Metropolis Sampling)
    weight_insert = 1

    # Accept
    if np.random.random() < weight_insert:
        # Case 1: Ira first, then Masha
        if tau_1 < tau_2 :
            #print("Ira first , Masha second")
            n = data_struct[i][r][1] - 1
            if n == -1 : return None # Reject if there were no particles to destroy
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

        print("Insert: " )
        print(data_struct[i])
    # Reject
    else: return None

    # Flag indicationg a worm is now present
    is_worm_present[0] = True

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
    #print("Site indices for tau1, tau2: ", tau_1_siteidx,tau_2_siteidx)
    #print("tau indices for tau1, tau2: ", tau_1_tauidx,tau_2_tauidx)
    #print("\n")
    #print("Pre timeshift: ")
    #print(data_struct[tau_1_siteidx])
    tau_1 = data_struct[tau_1_siteidx][tau_1_tauidx][0]
    tau_2 = data_struct[tau_2_siteidx][tau_2_tauidx][0]

    # Similar to above, but find the locs of closest kinks to ira and masha in im time

    # Case 1: Ira is ahead in imaginary time
    if tau_1_tauidx > tau_2_tauidx:
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
    return None

'----------------------------------------------------------------------------------'

def worm_spaceshift(data_struct,beta,is_worm_present,ira_loc,masha_loc):

    # Retrieve the site and tau indices of where ira and masha are located
    # ira_loc = [site_idx,tau_idx]
    tau_1_siteidx = ira_loc[0]
    tau_1_tauidx = ira_loc[1]
    tau_2_siteidx = masha_loc[0]
    tau_2_tauidx = masha_loc[1]

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
        if tau_1_tauidx > tau_2_tauidx : # Ira ahead
            del data_struct[tau_1_siteidx][tau_1_tauidx] # Deletes ira
            del data_struct[tau_2_siteidx][tau_2_tauidx] # Deletes masha
        else: # Ira behind
            del data_struct[tau_2_siteidx][tau_2_tauidx] # Deletes masha
            del data_struct[tau_1_siteidx][tau_1_tauidx] # Deletes ira

        print("Delete: ")
        print(data_struct[tau_1_siteidx])

        # Update the worm flag and ira_loc,masha_loc
        is_worm_present[0] = False
        del ira_loc[:]
        del masha_loc[:]

        return None

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def view_worldlines(data_struct,beta,figure_name):

    # Set initial configuration
    # Number of lattice sites
    L = len(data_struct)

    # Number of total particles in the lattice
    # N = 0
    # for site_idx in range(L):
    #     N += data_struct[site_idx][-1][1] # taulist[list[(tau,N,jump_dir)]]

    x = []
    for i in range(L):
        x.append(data_struct[i][0][1])
        del data_struct[i][0]

    print("Initial configuration: | x > = ", x)

    # Spread tau,N,dir data to different arrays (hopefully will help with visualization???)
    # Store kink times, particles after kinks, and kink directions for each site i
    tau_list = []
    N_list = []
    dirs_list = []
    for i in range(L):
        tau_i = []
        N_i = []
        dirs_i = []
        events = len(data_struct[i]) # Size of tuple for each site i
        for e in range(events):
            tau_i.append(data_struct[i][e][0])
            N_i.append(data_struct[i][e][1])
            dirs_i.append(data_struct[i][e][2])
        tau_list.append(tau_i)
        N_list.append(N_i)
        dirs_list.append(dirs_i)

    # Initialize figure
    plt.figure()

    # Strategy: -Plot worldlines for each site, ignoring kinks
    #           -Include kinks
    #           -Plot tau=0 segments for each site
    for i in range(L):
            # Occupation number of site i
            n = x[i]
            # Select line style depending on site occupation
            if n == 0: ls,lw = ':',1.3
            elif n == 1: ls,lw = '-',1.3
            elif n == 2: ls,lw = '-',4
            elif n == 3: ls,lw = '-',8

            # Draw line with no kinks if no kinks happened
            if len(tau_list[i]) == 0:
                plt.vlines(i,0,beta,linestyle=ls,linewidth=lw)
            else:
                tau_initial = 0
                tau_final = tau_list[i][0]
                plt.vlines(i,tau_initial,tau_final,linestyle=ls,linewidth=lw)

    # Plot middle segments for each site
    for i in range(L):
        L_tau = len(tau_list[i])
        for j in range(L_tau):
            if L_tau > 0:
                tau_initial = tau_list[i][j]
                if j + 1 < L_tau: # To avoid running out of range
                    tau_final = tau_list[i][j+1]
                else:
                    tau_final = beta
                n = N_list[i][j]
                if n == 0: ls,lw = ':',1
                elif n == 1: ls,lw = '-',1
                elif n == 2: ls,lw = '-',3
                elif n == 3: ls,lw = '-',5.5
                src_site = dirs_list[i][j][0]    # Index of source site
                dest_site = dirs_list[i][j][1]  # Index of destination site

                # Draw flat regions
                plt.vlines(i,tau_initial,tau_final,linestyle=ls,linewidth=lw)
                # Draw worm
                if src_site == dest_site:
                    #plt.plot(i,tau_initial,marker='_',ms=5,lw=5)
                    plt.hlines(tau_initial,i-0.06,i+0.06,lw=1)
                # Draw kinks
                #print(dirs_list)
                #print(N_list)
                src_site = dirs_list[i][j][0]    # Index of source site
                dest_site = dirs_list[i][j][1]  # Index of destination site

                # Wrap around site axis if kink connects first and last sites
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

    return None

# Test insert
data_struct = [ [[0,1,-26],[0.25,2,(1,0)],[0.5,1,(0,2)],[0.75,0,(0,1)]],
                [[0,1,-26],[0.25,0,(1,0)],[0.75,1,(0,1)]],
                [[0,1,-26],[0.5,2,(0,2)]] ]
beta = 1
is_worm_present = [False] # made flag a list so it can be passed "by reference"
ira_loc = []    # If there's a worm present, these will store
masha_loc = []  # the site_idx and tau_idx "by reference"
#print("Worm present: ",is_worm_present)
#print(data_struct)
worm_insert(data_struct,beta,is_worm_present,ira_loc,masha_loc)
#print(data_struct)
#print("Worm present: ",is_worm_present, "Ira at: ",ira_loc, "Masha at: ", masha_loc)

file_name = "worldlines_insert.pdf"
#view_worldlines(data_struct,beta,file_name)

# Test timeshift
worm_timeshift(data_struct,beta,is_worm_present,ira_loc,masha_loc)

file_name = "worldlines_timeshift.pdf"
#view_worldlines(data_struct,beta,file_name)

# Test delete
worm_delete(data_struct,beta,is_worm_present,ira_loc,masha_loc)
#print(is_worm_present,ira_loc,masha_loc)

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


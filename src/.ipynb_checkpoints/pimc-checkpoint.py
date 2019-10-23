# Functions to be used in main file (lattice_pimc.py)
import numpy as np
import bisect
import matplotlib.pyplot as plt
from scipy.stats import truncexpon

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

    # NEED TO DO THIS WITH TRUNCATED EXPONENTIAL #
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
        N_after_masha = n_i + 1
        N_after_ira = N_after_masha - 1
        masha_kink = [tau_2,N_after_masha,(i,i)]
        ira_kink = [tau_1,N_after_ira,(i,i)]
    else: # antiworm
        N_after_ira = n_i - 1
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
    print((ix,ik),(mx,mk))
    if is_worm:
        tau_min = data_struct[mx][mk-1][0]
        if ik+1 == len(data_struct[ix]): 
            tau_max = beta
        else: 
            tau_max = data_struct[ix][ik+1][0]
        n_before_worm = data_struct[mk][mk-1][1]
    else: # antiworm
        tau_min = data_struct[ix][ik-1][0] 
        if mk+1 == len(data_struct[mx]):
            tau_max = beta
        else:
             tau_max = data_struct[mx][mk+1][0]
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
        n_i = data_struct[mx][mk][1]   # particles in flat before delete
        dV = U*(1-n_i) - mu            # deleted energy minus energy of worm still there
        weight_ratio = np.exp(dV*dtau)/(n_i*eta**2)   # W_deleted/W_stillthere
    else: # delete antiworm
        n_i = data_struct[ix][ik][1]
        dV =  U*n_i + mu
        weight_ratio = np.exp(dV*dtau)/((n_i+1)*eta**2)
                   
    # Metropolis sampling
    # Accept
    p_tunable = 1 # p_iw/p_dw
    R = (p_tunable * p_L * p_flat * p_wormtype * p_wormlen * p_tau) * weight_ratio 
    if np.random.random() < R:
        # Delete the worm ends
        if is_worm: # worm
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
        tau_old = tau_1
    else:
        x = mx
        k = mk
        tau_old = tau_2
    
    # Determine the lower and upper bounds of the worm end to be timeshifted
    tau_max_idx = k + 1
    tau_min_idx = k - 1
    # Get tau_max
    if tau_max_idx == len(data_struct[x]):
        tau_max = beta  # This covers the case when there's no kinks after the worm end
    else:
        tau_max = data_struct[x][tau_max_idx][0] #actual times
    # Get tau_min
    tau_min = data_struct[x][tau_min_idx][0]
    
    # Determine if the worm end moved forward or backwards in imaginary time
    if tau_new >= tau_old: 
        shift_forward = True
    else:
        shift_forward = False

    # Get the diagonal energy differences between new and old configurations
    if shift_ira and shift_forward:             # ira forward
        n_i = data_struct[ix][ik][1]
        dV = U*n_i + mu
    elif shift_ira and not(shift_forward):      # ira backward
        n_i = data_struct[ix][ik-1][1]
        dV = U*(1-n_i) - mu
    elif not(shift_ira) and shift_forward:      # masha forward
        n_i = data_struct[mx][mk][1]
        dV = U*(1-n_i) - mu
    else:                                       # masha backward      
        n_i = data_struct[mx][mk-1][1]
        dV = U*n_i + mu  
               
    # From the truncated exponential distribution, choose new time of the worm end
    #loc = tau_min          # shift
    loc = 0
    b = tau_max - tau_min  # spread
    scale = 1/abs(dV)

    delta_tau = truncexpon.rvs(b=b,loc=loc,scale=scale,size=int(1))[0]
  
    # Do we add delta_tau to tau_min or subtract it from tau_max?
    

    # THERE"S IS NO METROPOLIS!!!!!!!!!!!!!
    # Rejection free
    weight_ratio = np.exp(-dV*(abs(tau_new-tau_old)))    # Z(C+)/Z(C)
       
    # Metropolis sampling
    #p_att_ratio = 2/truncexpon.pdf(tau_new,b=b,loc=loc,scale=scale) # p_att_-/p_att_+
    p_att_ratio = 2
    
    R = p_att_ratio*weight_ratio
    # Accept
    if np.random.random() < R:
        data_struct[x][k][0] = tau_new # Modify the time of the moved worm end
        return None

    # Reject
    else : return None

'----------------------------------------------------------------------------------'

def insert_gsworm_zero(data_struct, beta, ira_loc, masha_loc, U, mu, eta):
    
    if ira_loc != [] and masha_loc != []: return None
    
    return None

'----------------------------------------------------------------------------------'

def delete_gsworm_zero(data_struct, beta, ira_loc, masha_loc, U, mu, eta):

    if ira_loc != [] and masha_loc != []: return None

        
    return None
    
'----------------------------------------------------------------------------------'

def insert_gsworm_beta(data_struct, beta, ira_loc, masha_loc, U, mu, eta):
       
    if ira_loc != [] and masha_loc != []: return None

    return None
    
'----------------------------------------------------------------------------------'

def delete_gsworm_beta(data_struct, beta, ira_loc, masha_loc, U, mu, eta):
        
    if ira_loc != [] and masha_loc != []: return None

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

'----------------------------------------------------------------------------------'

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
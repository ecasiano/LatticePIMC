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

def worm_insert(data_struct, beta, head_loc, tail_loc, U, mu, eta):
    '''Inserts a worm or antiworm'''

    # Can only insert worm if there are no wormends present
    if head_loc != [] or tail_loc != [] : return None

    # Number of lattice sites
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    p_L = 1/L # probability of selecting site i

    # Randomly select a flat tau interval at which to possibly insert worm
    n_flats = len(data_struct[i])
    flat_min_idx = np.random.randint(n_flats)           # Index of lower bound of flat region
    tau_prev = data_struct[i][flat_min_idx][0]
    if flat_min_idx == n_flats - 1 : tau_next = beta     # In case that last flat is chosen
    else : tau_next = data_struct[i][flat_min_idx+1][0]
    p_flat = 1/n_flats                                  # prob. of selecting the flat interval
    dtau_flat = tau_next - tau_prev                       # length of the flat interval

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
        tau_t = tau_prev + np.random.random()*(dtau_flat - dtau_worm) # worm tail (creates a particle)
        tau_h = tau_t + dtau_worm                                    # worm head (destroys a particle)
    else: # antiworm
        tau_h = tau_prev + np.random.random()*(dtau_flat - dtau_worm)
        tau_t = tau_h + dtau_worm
    p_tau = 1/(dtau_flat-dtau_worm)     # prob. of inserting the worm end at the chosen time

    # Reject update if worm end is inserted at the bottom kink of the flat
    # (this will probably never happen in the 2 years I have left to complete my PhD :p )
    if tau_h == tau_prev or tau_t == tau_prev : return None

    # Reject update if both worm ends are at the same tau
    if tau_h == tau_t :
        return None

    # Build the worm end kinks to be inserted on i
    if insert_worm: # worm
        N_after_masha = n_i + 1
        N_after_ira = N_after_masha - 1
        masha_kink = [tau_t,N_after_masha,(i,i)]
        ira_kink = [tau_h,N_after_ira,(i,i)]
    else: # antiworm
        N_after_ira = n_i - 1
        N_after_masha = N_after_ira + 1
        ira_kink = [tau_h,N_after_ira,(i,i)]
        masha_kink = [tau_t,N_after_masha,(i,i)]

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
    R = 1 # For debugging
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
            tail_loc.extend([i,flat_min_idx+1])
            head_loc.extend([i,flat_min_idx+2])

        # Insert antiworm
        else:
            if flat_min_idx == n_flats - 1: # last flat
                data_struct[i].append(ira_kink)
                data_struct[i].append(masha_kink)
            else:
                data_struct[i].insert(flat_min_idx+1,ira_kink)
                data_struct[i].insert(flat_min_idx+2,masha_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            head_loc.extend([i,flat_min_idx+1])
            tail_loc.extend([i,flat_min_idx+2])
            
            return None

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def worm_delete(data_struct, beta, head_loc, tail_loc, U, mu, eta):

    # Can only propose worm deletion if both worm ends are present
    if head_loc == [] or tail_loc == [] : return None

    # Only delete if worm ends are on the same site and on the same flat interval
    if head_loc[0] != tail_loc[0] or abs(head_loc[1]-tail_loc[1]) != 1: return None

    # Retrieve the site and tau indices of where ira and masha are located
    # head_loc = [site_idx,tau_idx]
    hx = head_loc[0]
    hk = head_loc[1]
    tx = tail_loc[0]
    tk = tail_loc[1]
    
    # Identify the type of worm
    if hk > tk : is_worm = True   # worm
    else: is_worm = False         # antiworm
    
    # Calculate worm length
    tau_h = data_struct[hx][hk][0]
    tau_t = data_struct[tx][tk][0]
    dtau = np.abs(tau_h-tau_t)

    # Identify the lower and upper limits of the flat interval where the worm lives
    if is_worm:
        tau_prev = data_struct[tx][tk-1][0]
        if hk+1 == len(data_struct[hx]): 
            tau_next = beta
        else: 
            tau_next = data_struct[hx][hk+1][0]
        n_before_worm = data_struct[tk][tk-1][1]
    else: # antiworm
        tau_prev = data_struct[hx][hk-1][0] 
        if tk+1 == len(data_struct[tx]):
            tau_next = beta
        else:
             tau_next = data_struct[tx][tk+1][0]
        n_before_worm = data_struct[hx][hk-1][1]

    # Worm insert proposal probability
    p_L = 1/len(data_struct)           # prob of choosing site
    p_flat = 1/len(data_struct[hx])    # prob of choosing flat
    if n_before_worm == 0:             # prob of choosing worm/antiworm
        p_wormtype = 1 # Only a worm could've been inserted
    else:
        p_wormtype = 1/2
    p_wormlen = 1/(tau_next-tau_prev)    # prob of choosing wormlength
    p_tau = 1/((tau_next-tau_prev)-dtau) # prob of choosing the tau of the first wormend
    
    # Choose the appropriate weigh ratio based on the worm type
    if is_worm:
        n_i = data_struct[tx][tk][1]   # particles in flat before delete
        dV = U*(1-n_i) - mu            # deleted energy minus energy of worm still there
        weight_ratio = np.exp(dV*dtau)/(n_i*eta**2)   # W_deleted/W_stillthere
    else: # delete antiworm
        n_i = data_struct[hx][hk][1]
        dV =  U*n_i + mu
        weight_ratio = np.exp(dV*dtau)/((n_i+1)*eta**2)
                   
    # Metropolis sampling
    # Accept
    p_tunable = 1 # p_iw/p_dw
    R = (p_tunable * p_L * p_flat * p_wormtype * p_wormlen * p_tau) * weight_ratio 
    if np.random.random() < R:
        # Delete the worm ends
        if is_worm: # worm
            del data_struct[hx][hk] # Deletes ira
            del data_struct[tx][tk] # Deletes masha
        else: # antiworm
            del data_struct[tx][tk] # Deletes masha
            del data_struct[hx][hk] # Deletes ira

        # Update the locations ira and masha
        del head_loc[:]
        del tail_loc[:]

        return None

    # Reject
    else : return None
    
'----------------------------------------------------------------------------------'

def worm_timeshift(data_struct,beta,head_loc,tail_loc, U, mu):

    # Reject update if there is no worm present
    if head_loc == [] and tail_loc == [] : return None

    # Retrieve the site and tau indices of where ira and masha are located
    # head_loc = [site_idx,tau_idx]
    hx = head_loc[0]                # site index 
    hk = head_loc[1]                # kink index
    tx = tail_loc[0]
    tk = tail_loc[1]

    # Retrieve the actual times of head and tail
    tau_h = data_struct[hx][hk][0]
    tau_t = data_struct[tx][tk][0]
    
    # Randomly choose to shift HEAD or TAIL
    if np.random.random() < 0.5:
        shift_head = True
        #print("Moved Head")
    else:
        shift_head = False
        #print("Moved Tail")

    # Save the site and kink indices of the end that will be moved
    if shift_head == True :
        x = hx
        k = hk
    else:
        x = tx
        k = tk

    # Number of particles before and after the worm end to be shifted
    m_i = data_struct[x][k][1]       # after
    n_i = data_struct[x][k-1][1]     # before

    # Calculate the diagonal energy differences between old and new   configurations
    dV = (U/2)*(n_i*(n_i-1)-m_i*(m_i-1)) - mu*(n_i-m_i)
    #print("dV=",dV)
    #print("m_i=",m_i)
    #print("n_i=",n_i)

    # Determine the lower and upper bounds of the worm end to be timeshifted
    # Get tau_next
    if k+1 == len(data_struct[x]):
        tau_next = beta  # This covers the case when there's no kinks after the worm end
    else:
        tau_next = data_struct[x][k+1][0] #actual times
    # Get tau_prev
    tau_prev = data_struct[x][k-1][0]
    
    # Determine the "spread" of the truncated exponential distribution
    if tau_h > tau_t: # worm
        if shift_head:
            b = tau_next - tau_t
        else:
            b = tau_h - tau_prev
    else: # antiworm
        if shift_head:
            b = tau_t - tau_prev
            #print("\nShift head, b=%.4f"%b)
        else:
            b = tau_next - tau_h
            #print("\nShift tail")
            
    #print("tau_head: %.4f"%tau_h)
    #print("tau_tail: %.4f"%tau_t)
    #print("head_loc: ", head_loc)
    #print("tail_loc: ", tail_loc)
        
    #NOTE: Might need to include new conditional to determine b in the presence of antiworm
   
    # From the truncated exponential distribution, choose new time of the worm end
    loc = 0
    scale = 1/abs(dV)    
    r = truncexpon.rvs(b=b/scale,scale=scale,loc=loc,size=1)[0]
    if tau_t < tau_h: # worm
        if dV > 0:
            if shift_head:
                tau_new = tau_t + r
            else:
                tau_new = tau_h - r
        else: # dV > 0
            if shift_head:
                tau_new = tau_next - r        
            else:
                tau_new = tau_prev + r
    else: # antiworm
        if dV > 0:
            if shift_head:
                tau_new = tau_t - r
            else:
                tau_new = tau_h + r
        else: # dV > 0
            if shift_head:
                tau_new = tau_prev + r        
            else:
                tau_new = tau_next - r

    # Accept
    data_struct[x][k][0] = tau_new
        
    return None

'----------------------------------------------------------------------------------'

def insert_gsworm_zero(data_struct, beta, head_loc, tail_loc, U, mu, eta):
    
    # Cannot insert if there's two worm end presents
    if head_loc != [] and tail_loc != []: return None
    
    # Randomly choose a lattice site
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    p_L = 1/L # probability of selecting site i
    
    # Determine the upper and lower bound of the first flat interval of the site
    if len(data_struct[i]) == 1: # Worldline is flat throughout
        tau_next = beta
    else:
        tau_next = data_struct[i][1][0]
    tau_prev = 0

    # Randomly choose time at which to insert worm end
    tau_new = np.random.random()*tau_next
    
    # Decide between worm/antiworm based on the worm ends present
    if head_loc == [] and tail_loc != []: # can only insert worm head (worm)
        insert_head = True
        p_type = 1 # can only insert worm tail (antiworm)s
    elif head_loc != [] and tail_loc == []: # insert tail (antiworm from zero)
        insert_head = False
        p_type = 1
    elif data_struct[i][0][1] == 0 and head_loc == []: # can't insert tail if no particles
        insert_head = True
        p_type = 1
    else: # choose to insert either worm end with equal probability
        if np.random.random() < 0.5:
            insert_head = True
        else:
            insert_head = False
        p_type = 0.5
        
    # Build the structure representing the worm end to be inserted and the first flat
    m_i = data_struct[i][0][1] # particles after worm end (original number of particles)
    if insert_head:
        n_i = m_i + 1              # particles before worm head
    else:
        n_i = m_i - 1              # particles before antiworm tail
        
    worm_end_kink = [tau_new,m_i,(i,i)]
    first_flat = [0,n_i,(i,i)]
    
    # Build the Metropolis Ratio   
    R = 1
    if np.random.random() < 1: # Accept
        if len(data_struct[i]) == 1: # Worldline is flat throughout
            data_struct[i].append(worm_end_kink)
        else:
            data_struct[i].insert(1,worm_end_kink)
        
        data_struct[i][0] = first_flat # Modify the first flat
                
        # Save head and tail locations (site index, kink index)  
        if insert_head: # insert worm head (worm)
            head_loc.extend([i,1])
            # Reindex the other worm end if it was also on site i
            if tail_loc != []:
                if tail_loc[0] == i:
                    tail_loc[1] += 1
        else: # insert worm tail (antiworm)
            tail_loc.extend([i,1])
            # Reindex other worm end if necessary
            if head_loc != []:
                if head_loc[0] == i:
                    head_loc[1] += 1 

        return None
        
    else: # Reject
        return None

'----------------------------------------------------------------------------------'

def delete_gsworm_zero(data_struct, beta, head_loc, tail_loc, U, mu, eta):

    # Cannot delete if there are no worm ends present
    if head_loc == [] and tail_loc == []: return None
    
    # Retrieve the site and kink indices of the worm ends
    hx = head_loc[0] 
    hk = head_loc[1]
    tx = tail_loc[0]
    tk = tail_loc[1]
    
    # Only delete worms that originate at tau = 0
    if hk != 1 and tk != 1 : return None
    
    # Select worm end to delete
    if hk == 1 and tk != 1:     # Only the head is near tau=0
        delete_head = True
        prob = 1
    elif hk != 1 and tk == 1:   # Only the tail is near tau=0
        delete_head = False
        prob = 1
    else:                       # Both are near zero (but different sites)
        if np.random.random() < 0.5:
            delete_head = True
        else:
            delete_head = False
        prob = 0.5
        
    # Metropolis Sampling
    R = 1
    if np.random.random() < R: # Accept
        if delete_head:
            del data_struct[hx][hk]
            del head_loc[:]
            data_struct[hx][0][1] -= 1
            # Reindex if there was another end on the same worldline
            if hx == tx and tk > hk:
                tail_loc[1] -= 1
        
        else: # delete tail
            del data_struct[tx][tk]
            del tail_loc[:]
            data_struct[tx][0][1] += 1
            # Reindex if there was another end on the same worldline
            if hx == tx and hk > tk:
                head_loc[1] -= 1
                
        return None
    
    else: # Reject 
        return None
    
'----------------------------------------------------------------------------------'

def insert_gsworm_beta(data_struct, beta, head_loc, tail_loc, U, mu, eta):
    
    # Cannot insert if there's two worm end presents
    if head_loc != [] and tail_loc != []: return None
    
    # Randomly choose a lattice site
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    p_L = 1/L # probability of selecting site i
    
    # Count the number of kinks on site i to get the index of the last kink
    k_last = len(data_struct[i]) - 1
    
    # Determine the upper and lower bound of the first flat interval of the site
    tau_next = beta
    tau_prev = data_struct[i][k_last][0]

    # Randomly choose time at which to insert worm end
    tau_new = tau_prev + np.random.random()*(beta-tau_prev)
    
    # Decide between worm/antiworm insertion based on the worm ends present
    if head_loc == [] and tail_loc != []: # can only insert worm head (antiworm)
        insert_head = True
        p_type = 1 # can only insert worm tail (antiworm)
    elif head_loc != [] and tail_loc == []: # insert tail (worm from beta)
        insert_head = False
        p_type = 1
    elif data_struct[i][k_last][1] == 0 and tail_loc == []: # can't insert head if no particles
        insert_head = False
        p_type = 1
    else: # choose to insert either worm end with equal probability
        if np.random.random() < 0.5:
            insert_head = True
        else:
            insert_head = False
        p_type = 0.5
        
    # Build the structure representing the worm end to be inserted and the last flat
    n_i = data_struct[i][k_last][1] # particles before worm end (original number of particles)
    if insert_head:
        m_i = n_i - 1              # particles after worm head
    else:
        m_i = n_i + 1              # particles after antiworm tail
        
    worm_end_kink = [tau_new,m_i,(i,i)]
    
    # Build the Metropolis Ratio   
    R = 1
    if np.random.random() < 1: # Accept
        
        data_struct[i].append(worm_end_kink)
                        
        # Save head and tail locations (site index, kink index)  
        if insert_head: # insert worm head (worm)
            head_loc.extend([i,k_last+1])
        else: # insert worm tail (antiworm)
            tail_loc.extend([i,k_last+1])

        return None
        
    else: # Reject
        return None
    
'----------------------------------------------------------------------------------'

def delete_gsworm_beta(data_struct, beta, head_loc, tail_loc, U, mu, eta):

    # Cannot delete if there are no worm ends present
    if head_loc == [] and tail_loc == []: return None
    
    # Retrieve the site and kink indices of the worm ends
    hx = head_loc[0] 
    hk = head_loc[1]
    tx = tail_loc[0]
    tk = tail_loc[1]
    
    # Length of the 
    
    # Only delete worms that originate at tau = beta
    hk_len = len(data_struct[hx]) # Kinks of site where the head is
    tk_len = len(data_struct[tx]) # Kinks of site where the tail is
    if hk != hk_len-1 and tk != tk_len-1: return None
    
    # Select worm end to delete
    if hk == hk_len-1 and tk != tk_len-1:     # Only the head is near tau=beta
        delete_head = True
        prob = 1
    elif hk != hk_len-1 and tk == tk_len-1:   # Only the tail is near tau=beta
        delete_head = False
        prob = 1
    else:                       # Both are near beta (but different sites)
        if np.random.random() < 0.5:
            delete_head = True
        else:
            delete_head = False
        prob = 0.5
        
    # Metropolis Sampling
    R = 1
    if np.random.random() < R: # Accept
        if delete_head:
            del data_struct[hx][hk]
            hk_len -= 1
            #data_struct[hx][hk_len-1][1] += 1
            del head_loc[:]

        else: # delete tail
            del data_struct[tx][tk]
            tk_len -= 1
            #data_struct[tx][tk_len-1][1] -= 1
            del tail_loc[:]

                
        return None
    
    else: # Reject 
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
# head_loc = []    # If there's a worm present, these will store
# tail_loc = []  # the site_idx and tau_idx "by reference"

# M = int(1E+03)
# ctr00, ctr01, ctr02, ctr03, ctr04 = 0, 0, 0, 0, 0
# # Plot original configuration
# file_name = "worldlines_0%d_00.pdf"%ctr00
# #view_worldlines(data_struct,beta,file_name)
# print(" --- Progress --- ")
# for m in range(M):
#     # Test insert/delete worm and plot it
#     worm(data_struct,beta,head_loc,tail_loc)
#     file_name = "worldlines_0%d_01.pdf"%ctr01
#     #view_worldlines(data_struct,beta,file_name)
#     ctr01 += 1

#     # Test timeshift and plot it
#     worm_timeshift(data_struct,beta,is_worm_present,head_loc,tail_loc)
#     file_name = "worldlines_0%d_02.pdf"%ctr02
#     #view_worldlines(data_struct,beta,file_name)
#     ctr02 += 1

#     # Test spaceshift_before_insert and plot it
#     worm_spaceshift_before(data_struct,beta,is_worm_present,head_loc,tail_loc)
#     file_name = "worldlines_0%d_03.pdf"%ctr03
#     #view_worldlines(data_struct,beta,file_name)
#     ctr03 += 1

#     # Test spaceshift_after and plot it
#     worm_spaceshift_after(data_struct,beta,is_worm_present,head_loc,tail_loc)
#     file_name = "worldlines_0%d_04.pdf"%ctr04
#     #view_worldlines(data_struct,beta,file_name)
#     ctr04 += 1

#     # Test gsworm_insert
#     gsworm_insert(data_struct,beta,is_worm_present,head_loc,tail_loc)


#     # Progress
#     print("%.2f%%"%((m+1)/M*100))


    ##############################################
    # Forces worm instead of antiworm
    #if tau_t > tau_h:
    #    tmp = tau_h
    #   tau_h = tau_t
    #    tau_t = tmp

    # Forces insert antiworm instead of worm
    #if tau_h > tau_t:
    #    tmp = tau_t
    #    tau_t = tau_h
    #    tau_h = tmp
    ##############################################				                      
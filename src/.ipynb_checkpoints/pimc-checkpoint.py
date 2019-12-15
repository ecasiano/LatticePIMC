# Functions to be used in main file (lattice_pimc.py)
import numpy as np
import bisect
import matplotlib.pyplot as plt
from copy import deepcopy
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

def N_tracker(data_struct,beta):
    '''Count total particles in the worldline configuration'''
    
    # Number of lattice sites
    L = len(data_struct)
    
    # Add paths
    l = 0
    for i in range(L):
        N_flats = len(data_struct[i]) # Number of flat intervals on the site
        for k in range(N_flats):
            if k < N_flats-1:
                dtau = data_struct[i][k+1][0]-data_struct[i][k][0]
            else:
                dtau = beta-data_struct[i][k][0]
            
            n = data_struct[i][k][1] # particles in flat interval
            l += n*dtau
                
    # Track the total number of particles (must be between N-1 and N+1)
    N = l/beta
    
    return N

'----------------------------------------------------------------------------------'

def egs_pimc(data_struct,beta,U,mu):
    '''Calculates pimc ground state energy at time slice tau=beta/2'''
    
    # Number of lattice sites
    L = len(data_struct)
    
    # BoseHubbard Energy (for no hopping)
    egs = 0
    for i in range(L):
        N_flats = len(data_struct[i]) # Number of flats on site i
        for k in range(N_flats):
            if data_struct[i][k][0] <= beta/2:
                n_i = data_struct[i][k][1] # particles on i at beta/2
            else: break
        # Add on-site contribution to energy if no worms crossing beta/2
        # if data_struct[i][k][2][0] != data_struct[i][k][2][1]:
        egs += ( (U/2)*n_i*(n_i-1)-mu*n_i )
                    
    return egs

'----------------------------------------------------------------------------------'

def egs_theory(L,U,mu):
    '''Calculates BH model theoretical ground state energy with no hopping'''
    
    # NOTE: This works for unit filling (not sure if otherwise works)
    n_min = 0.5 + mu/U
    egs = L*U/2 * n_min * (n_min-1) - L*mu*n_min
    
    return egs

'----------------------------------------------------------------------------------'

def worm_insert(data_struct, beta, head_loc, tail_loc, U, mu, eta,
               canonical, N):
    '''Inserts a worm or antiworm'''

    # Can only insert worm if there are no wormends present
    if head_loc != [] or tail_loc != [] : return None

    # Number of lattice sites
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm 
    i = np.random.randint(L)

    # Randomly select a flat tau interval at which to possibly insert worm
    N_flats = len(data_struct[i])
    flat_min_idx = np.random.randint(N_flats)           # Index of lower bound of flat region
    tau_prev = data_struct[i][flat_min_idx][0]
    if flat_min_idx == N_flats - 1 : tau_next = beta     # In case that last flat is chosen
    else : tau_next = data_struct[i][flat_min_idx+1][0]
    tau_flat = tau_next - tau_prev                       # length of the flat interval

    # Randomly choose either to insert worm or, if possible, an antiworm
    n_i = data_struct[i][flat_min_idx][1]  # initial number of particles in the flat interval
    if n_i == 0 : # only worm can be inserted
        insert_worm = True
        p_type = 1
    else:
        if np.random.random() < 0.5:
            insert_worm = True
        else:
            insert_worm = False
        p_type = 0.5 # prob. of the worm being either a worm or antiworm

    # MEASURE THE DIFFERENCE IN DIAGONAL ENERGY. To ensure exponential DECAY of the 
    # update's weight, the difference will be taken always as dV = eps_w - eps, where eps_w is
    # the energy of the segment of path adjacent the moving worm end with more particles. 
    if insert_worm:   
        N_after_tail = n_i + 1
        N_after_head = n_i
    else:
        N_after_head = n_i - 1
        N_after_tail = n_i
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)

    # From the truncated exponential distribution, choose the length of the worm    
    #tau_worm  = np.random.random()*(tau_flat)
    loc = 0
    scale = 1/abs(dV)    
    b = tau_next - tau_prev
    tau_worm = truncexpon.rvs(b=b/scale,scale=scale,loc=loc,size=1)[0] # Worm length
    if not(insert_worm):     # Flip the distribution for antiworm insertion
        tau_worm = -tau_worm + b   # Antiworm length
        
    # Randomly choose the time where the first worm end will be inserted
    if insert_worm: # worm
        tau_t = tau_prev + np.random.random()*(tau_flat - tau_worm) # worm tail (creates a particle)
        tau_h = tau_t + tau_worm                                    # worm head (destroys a particle)
    else: # antiworm
        tau_h = tau_prev + np.random.random()*(tau_flat - tau_worm)
        tau_t = tau_h + tau_worm

    # Reject update if worm end is inserted at the bottom kink of the flat
    # (this will probably never happen in the 2 years I have left to complete my PhD :p )
    if tau_h == tau_prev or tau_t == tau_prev : return None

    # Reject update if both worm ends are at the same tau
    if tau_h == tau_t :
        return None

    # Build the worm end kinks to be inserted on i
    if insert_worm: # worm
        tail_kink = [tau_t,N_after_tail,(i,i)]
        head_kink = [tau_h,N_after_head,(i,i)]
    else: # antiworm
        head_kink = [tau_h,N_after_head,(i,i)]
        tail_kink = [tau_t,N_after_tail,(i,i)]
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
    
        # Insert worm
        if insert_worm:
            if flat_min_idx == N_flats - 1: # if selected flat is the last
                data_struct_tmp[i].append(tail_kink)
                data_struct_tmp[i].append(head_kink)
            else:
                data_struct_tmp[i].insert(flat_min_idx+1,tail_kink)
                data_struct_tmp[i].insert(flat_min_idx+2,head_kink)
        # Insert antiworm
        else:
            if flat_min_idx == N_flats - 1: # last flat
                data_struct_tmp[i].append(head_kink)
                data_struct_tmp[i].append(tail_kink)
            else:
                data_struct_tmp[i].insert(flat_min_idx+1,head_kink)
                data_struct_tmp[i].insert(flat_min_idx+2,tail_kink)
        
        N_check = N_tracker(data_struct_tmp,beta)
        if N_check <= N-1 or N_check >= N+1: return None 
        
    # Build the Metropolis ratio (R)
    p_dw,p_iw = 0.5,0.5       # tunable delete and insert probabilities       
    R = (p_dw/p_iw) * L * N_flats * (tau_flat - tau_worm) / p_type * eta**2 * N_after_tail 
    # Metropolis Sampling
    if np.random.random() < R:
        # Insert worm
        if insert_worm:
            if flat_min_idx == N_flats - 1: # if selected flat is the last
                data_struct[i].append(tail_kink)
                data_struct[i].append(head_kink)
            else:
                data_struct[i].insert(flat_min_idx+1,tail_kink)
                data_struct[i].insert(flat_min_idx+2,head_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            tail_loc.extend([i,flat_min_idx+1])
            head_loc.extend([i,flat_min_idx+2])

        # Insert antiworm
        else:
            if flat_min_idx == N_flats - 1: # last flat
                data_struct[i].append(head_kink)
                data_struct[i].append(tail_kink)
            else:
                data_struct[i].insert(flat_min_idx+1,head_kink)
                data_struct[i].insert(flat_min_idx+2,tail_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            head_loc.extend([i,flat_min_idx+1])
            tail_loc.extend([i,flat_min_idx+2])
            
            return None

    # Reject
    else:
        return None

'----------------------------------------------------------------------------------'

def worm_delete(data_struct, beta, head_loc, tail_loc, U, mu, eta,
               canonical, N):

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
    
    # Number of lattice sites and Number of flat regions on the worm site
    L = len(data_struct)
    N_flats = len(data_struct[hx])
    
    # Identify the type of worm
    if hk > tk : is_worm = True   # worm
    else: is_worm = False         # antiworm
    
    # Retrieve the times of the worm head and tail 
    tau_h = data_struct[hx][hk][0]
    tau_t = data_struct[tx][tk][0]

    # Identify the lower and upper limits of the flat interval where the worm lives
    if is_worm:
        tau_prev = data_struct[tx][tk-1][0]
        if hk == len(data_struct[hx])-1: 
            tau_next = beta
        else: 
            tau_next = data_struct[hx][hk+1][0]
        n_i = data_struct[tx][tk-1][1]  # number of particles outside of worm/antiworm
        N_after_tail = n_i+1
    else: # antiworm
        tau_prev = data_struct[hx][hk-1][0] 
        if tk == len(data_struct[tx])-1:
            tau_next = beta
        else:
             tau_next = data_struct[tx][tk+1][0]
        n_i = data_struct[hx][hk-1][1]
        N_after_tail = n_i

    # Calculate the length of the flat interval and the length of the worm/antiworm
    tau_flat = tau_next - tau_prev
    tau_worm = np.abs(tau_h-tau_t)
    
    # Worm insert probability of choosing between worm or antiworm
    if n_i == 0:
        p_type = 1 # Only a worm could've been inserted
    else:
        p_type = 1/2
        
    # Check if N is conserved on canonical simulations
    if canonical:
        data_struct_tmp = deepcopy(data_struct)
    
        # Delete the worm ends
        if is_worm: # worm
            del data_struct_tmp[hx][hk] # Deletes ira
            del data_struct_tmp[tx][tk] # Deletes masha
        else: # antiworm
            del data_struct_tmp[tx][tk] # Deletes masha
            del data_struct_tmp[hx][hk] # Deletes ira
        
        N_check = N_tracker(data_struct_tmp,beta)
        if N_check <= N-1 or N_check >= N+1: return None 
                   
    # Metropolis sampling
    p_dw, p_iw = 0.5,0.5 # p_iw/p_dw
    R = 1/ ( (p_dw/p_iw) * L * N_flats * (tau_flat - tau_worm) / p_type * eta**2 * N_after_tail )
    # Accept
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

def worm_timeshift(data_struct,beta,head_loc,tail_loc, U, mu, canonical, N):

    # Reject update if there are is no worm end present
    if head_loc == [] and tail_loc == [] : return None

    # Choose which worm end to move    
    if head_loc != [] and tail_loc == [] : # only head present
        hx = head_loc[0]                # site index 
        hk = head_loc[1]                # kink index
        tau_h = data_struct[hx][hk][0]
        shift_head = True
    elif head_loc == [] and tail_loc != [] : # only tail present
        tx = tail_loc[0]                # site index 
        tk = tail_loc[1]                # kink index
        tau_t = data_struct[tx][tk][0]
        shift_head = False
    else: # both worm ends present
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
        else:
            shift_head = False

    # Save the site and kink indices of the end that will be moved
    if shift_head == True :
        x = hx
        k = hk
    else:
        x = tx
        k = tk
    
    # Number of particles before and after the worm end to be shifted
    n_f = data_struct[x][k][1]       # after
    n_o = data_struct[x][k-1][1]     # before

    # MEASURE THE DIFFERENCE IN DIAGONAL ENERGY. To ensure exponential DECAY of the 
    # update's weight, the difference will be taken always as dV = eps_w - eps, where eps_w is
    # the energy of the segment of path adjacent the moving worm end with more particles.  
    if shift_head:                              
        dV = (U/2)*(n_o*(n_o-1)-n_f*(n_f-1)) - mu*(n_o-n_f)
    else:
        dV = (U/2)*(n_f*(n_f-1)-n_o*(n_o-1)) - mu*(n_f-n_o)
                    
    # Determine the lower and upper bounds of the worm end to be timeshifted
    # Get tau_next
    if k+1 == len(data_struct[x]):
        tau_next = beta  # This covers the case when there's no kinks after the worm end
    else:
        tau_next = data_struct[x][k+1][0] #actual times
    # Get tau_prev
    tau_prev = data_struct[x][k-1][0]
    
    # From the truncated exponential distribution, choose new time of the worm end
    loc = 0
    scale = 1/abs(dV)    
    b = tau_next - tau_prev
    r = truncexpon.rvs(b=b/scale,scale=scale,loc=loc,size=1)[0]

    if dV > 0:
        if shift_head:
            tau_new = tau_prev + r
        else:
            tau_new = tau_next - r
    else: # dV < 0
        #print("dV ", dV)
        if shift_head:
            tau_new = tau_next - r
        else:
            tau_new = tau_prev + r       
    
    # Accept
    tau_old = data_struct[x][k][0] # original time of the worm end
    data_struct[x][k][0] = tau_new
    if canonical:
        N_check = N_tracker(data_struct,beta)
        if N_check <= N-1 or N_check >= N+1:
            data_struct[x][k][0] = tau_old # reject update if N not conserved
        
    return None

'----------------------------------------------------------------------------------'

def insert_gsworm_zero(data_struct, beta, head_loc, tail_loc, U, mu, eta,
                      canonical,N):
    
    # Cannot insert if there's two worm ends present
    if head_loc != [] and tail_loc != []: return None
    
    # Randomly choose a lattice site
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    
    # Determine the upper and lower bound of the first flat interval of the site
    if len(data_struct[i]) == 1: # Worldline is flat throughout
        tau_next = beta
    else:
        tau_next = data_struct[i][1][0]
    # tau_prev = 0
    
    # Decide between worm/antiworm based on the worm ends present
    if head_loc == [] and tail_loc != []: # can only insert worm head (worm)
        insert_worm = True
        p_type = 1 # can only insert worm tail (antiworm)s
    elif head_loc != [] and tail_loc == []: # can only insert tail (antiworm from zero)
        insert_worm = False
        p_type = 1
    elif data_struct[i][0][1] == 0 and head_loc == []: # can't insert tail if no particles
        insert_worm = True
        p_type = 1
    else: # choose to insert either worm end with equal probability
        if np.random.random() < 0.5:
            insert_worm = True
        else:
            insert_worm = False
        p_type = 0.5 
        
    # MEASURE THE DIFFERENCE IN DIAGONAL ENERGY. To ensure exponential DECAY of the 
    # update's weight, the difference will be taken always as dV = eps_w - eps, where eps_w is
    # the energy of the segment of path adjacent the moving worm end with more particles. 
    n_i = data_struct[i][0][1] # particles after worm end (original number of particles)
    if insert_worm:
        N_after_tail = n_i + 1
        N_after_head = n_i
    else:
        N_after_head = n_i - 1
        N_after_tail = n_i
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
    
    # From the truncated exponential distribution, choose the length of the worm
    loc = 0
    scale = 1/abs(dV)    
    b = tau_next
    tau_worm = truncexpon.rvs(b=b/scale,scale=scale,loc=loc,size=1)[0] # Worm length
    if not(insert_worm):     # Flip the distribution for antiworm insertion
        tau_worm = -tau_worm + b   # Antiworm length
    
    # If there are two wormends present, the delete move has to randomly choose which to remove
    # This is important for detailed balance.
    if head_loc == [] and tail_loc == []:
        p_wormend = 1 # if no ends initially, we end up with only one end after insertion, delete chooses this one
    else:
        p_wormend = 0.5 # delete might have to choose between two ends
    
    # Build the kinks to be inserted to the data structture if the move is accepted
    if insert_worm:
        worm_end_kink = [tau_worm,N_after_head,(i,i)]  # kinks to be inserted to
        first_flat = [0,N_after_tail,(i,i)]           # the data structure
    else:
        worm_end_kink = [tau_worm,N_after_tail,(i,i)]
        first_flat = [0,N_after_head,(i,i)]
        
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
    
        if len(data_struct_tmp[i]) == 1: # Worldline is flat throughout
            data_struct_tmp[i].append(worm_end_kink)
        else:
            data_struct_tmp[i].insert(1,worm_end_kink)
        
        data_struct_tmp[i][0] = first_flat # Modify the first flat
        
        N_check = N_tracker(data_struct_tmp,beta)
        if N_check <= N-1 or N_check >= N+1: return None 
        
    # Build the Metropolis Ratio   
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    p_gsdw, p_gsiw = 0.5, 0.5
    R = (p_gsdw/p_gsiw) * L * p_wormend / p_type * eta * np.sqrt(N_after_tail) * C_post/C_pre
    if np.random.random() < R: # Accept
        if len(data_struct[i]) == 1: # Worldline is flat throughout
            data_struct[i].append(worm_end_kink)
        else:
            data_struct[i].insert(1,worm_end_kink)
        
        data_struct[i][0] = first_flat # Modify the first flat
                
        # Save head and tail locations (site index, kink index)  
        if insert_worm: # insert worm (just a head)
            head_loc.extend([i,1])
            # Reindex the other worm end if it was also on site i
            if tail_loc != []:
                if tail_loc[0] == i:
                    tail_loc[1] += 1
        else: # insert an antiworm (just a tail)
            tail_loc.extend([i,1])
            # Reindex other worm end if necessary
            if head_loc != []:
                if head_loc[0] == i:
                    head_loc[1] += 1 

        return None
        
    else: # Reject
        return None

'----------------------------------------------------------------------------------'

def delete_gsworm_zero(data_struct,beta,head_loc,tail_loc,U,mu,eta,canonical,N):

    # Cannot delete if there are no worm ends present
    if head_loc == [] and tail_loc == []: 
        return None
    
     # Can only propose if the worm ends don't form a worm
    if head_loc != [] and tail_loc != []:
        if head_loc[0] == tail_loc[0] and np.abs(head_loc[1]-tail_loc[1])==1: return None
    
     # Number of lattice sites
    L = len(data_struct)
    
    # Save the indices of worm end to be deleted
    if head_loc != [] and tail_loc == []: # only head present
        x = head_loc[0]                  # site index 
        k = head_loc[1]                  # kink index
        delete_head = True
        p_wormend = 1
        if k != 1:  # head not on first flat
            return None
    elif head_loc == [] and tail_loc != [] : # only tail present
        x = tail_loc[0]                     # site index 
        k = tail_loc[1]                     # kink index
        delete_head = False
        p_wormend = 1                       # probability of choosing this wormend
        if k != 1:      # tail not on first flat
            return None
    else: # both worm ends present
        x = head_loc[0]                # site index 
        k = head_loc[1]                # kink index
        delete_head = True
        p_wormend = 1
        if k != 1: # head not on first flat, try tail
            x = tail_loc[0]
            k = tail_loc[1]
            delete_head = False
            if k != 1: # tail no on last first either, stop.
                return None
        else:  # head on first flat, check tail too.
            tx = tail_loc[0]
            tk = tail_loc[1]
            if tk == 1: # if both ends on first flast, choose randomly
                if np.random.random() < 0.5:
                    x = tx
                    k = tk
                    delete_head = False
                    p_wormend = 0.5
       
    # Number of particles in flat after the near zero worm end
    n_i = data_struct[x][k][1]
    if delete_head:
        N_after_tail = n_i+1
    else:
        N_after_tail = n_i
        
    # Worm insert probability of choosing between worm or antiworm
    if n_i == 0:
        p_type = 1 # Only a worm could've been inserted
    else:
        p_type = 1/2
        
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        del data_struct_tmp[x][k]
        if delete_head:
            data_struct_tmp[x][0][1] -= 1
        else: # delete tail
            data_struct_tmp[x][0][1] += 1
    
        N_check = N_tracker(data_struct_tmp,beta)
        if N_check <= N-1 or N_check >= N+1: return None 

        
    # Metropolis Sampling
    p_gsdw,p_gsiw = 0.5,0.5
    C_post,C_pre = 0.5,0.5
    R = 1 / ( (p_gsdw/p_gsiw) * L * p_wormend / p_type * eta * np.sqrt(N_after_tail) * C_post/C_pre )
    if np.random.random() < R: # Accept
        del data_struct[x][k]
        
        if delete_head:
            data_struct[x][0][1] -= 1  # Modify the number of particles after deletion
            # Reindex if there was another end on the same worldline
            if head_loc != [] and tail_loc != []:
                if head_loc[0] == tail_loc[0] and tail_loc[1] > head_loc[1]:
                    tail_loc[1] -= 1
            del head_loc[:]

        
        else: # delete tail
            data_struct[x][0][1] += 1
            # Reindex if there was another end on the same worldline
            if head_loc != [] and tail_loc != []:
                if head_loc[0] == tail_loc[0] and tail_loc[1] < head_loc[1]:
                    head_loc[1] -= 1                
            del tail_loc[:]
            
        return None
    
    else: # Reject 
        return None
    
'----------------------------------------------------------------------------------'

def insert_gsworm_beta(data_struct, beta, head_loc, tail_loc, U, mu, eta,
                      canonical,N):
    
    # Cannot insert if there's two worm end presents
    if head_loc != [] and tail_loc != []: return None
    
    # Randomly choose a lattice site
    L = len(data_struct)

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    
    # Count the number of kinks on site i to get the index of the last kink
    k_last = len(data_struct[i]) - 1
    
    # Determine the upper and lower bound of the first flat interval of the site
    tau_prev = data_struct[i][k_last][0]
    
    # Decide between worm/antiworm insertion based on the worm ends present
    if head_loc == [] and tail_loc != []: # can only insert worm head (antiworm)
        insert_worm = False
        p_type = 1 # can only insert worm tail (antiworm)
    elif head_loc != [] and tail_loc == []: # insert tail (worm from beta)
        insert_worm = True 
        p_type = 1
    elif data_struct[i][k_last][1] == 0 and tail_loc == []: # can't insert head if no particles
        insert_worm = True
        p_type = 1
    else: # choose to insert either worm end with equal probability
        if np.random.random() < 0.5:
            insert_worm = True
        else:
            insert_worm = False
        p_type = 0.5  

    # MEASURE THE DIFFERENCE IN DIAGONAL ENERGY. To ensure exponential DECAY of the 
    # update's weight, the difference will be taken always as dV = eps_w - eps, where eps_w is
    # the energy of the segment of path adjacent the moving worm end with more particles. 
    n_i = data_struct[i][k_last][1] # particles after worm end (original number of particles)
    if insert_worm:
        N_after_tail = n_i + 1
        N_after_head = n_i
    else: # antiworm
        N_after_head = n_i - 1
        N_after_tail = n_i
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
    
    # From the truncated exponential distribution, choose the length of the worm 
    loc = 0
    scale = 1/abs(dV)    
    b = beta - tau_prev
    tau_worm = truncexpon.rvs(b=b/scale,scale=scale,loc=loc,size=1)[0] # Worm length
    if not(insert_worm):     # Flip the distribution for antiworm insertion
        tau_worm = -tau_worm + b   # Antiworm length
    
    # If there are two wormends present, the delete move has to randomly choose which to remove
    if head_loc == [] and tail_loc == []:
        p_wormend = 1 # if no ends initially, we end up with only one, delete only chooses this one
    else:
        p_wormend = 0.5 # delete might have to choose between two ends
              
    # Build the kinks to be inserted to the data structture if the move is accepted
    if insert_worm:
        worm_end_kink = [beta-tau_worm,N_after_tail,(i,i)]  # kinks to be inserted to
    else:
        worm_end_kink = [beta-tau_worm,N_after_head,(i,i)]
        
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        data_struct_tmp[i].append(worm_end_kink)
        
        N_check = N_tracker(data_struct_tmp,beta)
        if N_check <= N-1 or N_check >= N+1: return None 
        
    # Build the Metropolis Ratio   
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    p_gsdw, p_gsiw = 0.5, 0.5
    R = (p_gsdw/p_gsiw) * L * p_wormend / p_type * eta * np.sqrt(N_after_tail) * C_post/C_pre
    if np.random.random() < R: # Accept    
        data_struct[i].append(worm_end_kink)
                        
        # Save head and tail locations (site index, kink index)  
        if insert_worm: # insert worm
            tail_loc.extend([i,k_last+1])
        else: # insert antiworm
            head_loc.extend([i,k_last+1])

        return None
        
    else: # Reject
        return None
    
'----------------------------------------------------------------------------------'

def delete_gsworm_beta(data_struct,beta,head_loc,tail_loc,U,mu,eta,canonical,N):

    # Cannot delete if there are no worm ends present
    if head_loc == [] and tail_loc == []: return None
     
     # Can only propose if the worm ends don't form a worm
    if head_loc != [] and tail_loc != []:
        if head_loc[0] == tail_loc[0] and np.abs(head_loc[1]-tail_loc[1])==1: return None
    
    # Number of lattice sites
    L = len(data_struct)
    
    # Save the indices of worm end to be deleted
    if head_loc != [] and tail_loc == []: # only head present
        x = head_loc[0]                  # site index 
        k = head_loc[1]                  # kink index
        delete_head = True
        p_wormend = 1
        if k != len(data_struct[x])-1:  # head not on last flat
            return None
    elif head_loc == [] and tail_loc != [] : # only tail present
        x = tail_loc[0]                     # site index 
        k = tail_loc[1]                     # kink index
        delete_head = False
        p_wormend = 1                       # probability of choosing this wormend
        if k != len(data_struct[x])-1:      # tail not on last flat
            return None
    else: # both worm ends present
        x = head_loc[0]                # site index 
        k = head_loc[1]                # kink index
        delete_head = True
        p_wormend = 1
        if k != len(data_struct[x])-1: # head not on last flat, try tail
            x = tail_loc[0]
            k = tail_loc[1]
            delete_head = False
            if k != len(data_struct[x])-1: # tail no on last flat either, stop.
                return None
        else:  # head on last flat, check tail too.
            tx = tail_loc[0]
            tk = tail_loc[1]
            if tk == len(data_struct[x])-1: # if both ends on last flast, choose randomly
                if np.random.random() < 0.5:
                    x = tx
                    k = tk
                    delete_head = False
                    p_wormend = 0.5   
    
    # Number of particles in flat before the edge worm flat
    n_i = data_struct[x][k-1][1]
    if delete_head:
        N_after_tail = n_i
    else:
        N_after_tail = n_i+1
       
    # Worm insert (the reverse update) probability of choosing between worm or antiworm
    if n_i == 0:
        p_type = 1 # Only a worm could've been inserted
    else:
        p_type = 1/2
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        del data_struct_tmp[x][k]
    
        N_check = N_tracker(data_struct_tmp,beta)
        if N_check <= N-1 or N_check >= N+1: return None 
    
    # Metropolis Sampling
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    p_gsdw, p_gsiw = 0.5, 0.5
    R = 1 / ( (p_gsdw/p_gsiw) * L * p_wormend / p_type * eta * np.sqrt(N_after_tail) * C_post/C_pre )
    if np.random.random() < R: # Accept
        
        del data_struct[x][k]
        
        if delete_head:
            del head_loc[:]

        else: # delete tail
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
            else: ls,lw = '-',n

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
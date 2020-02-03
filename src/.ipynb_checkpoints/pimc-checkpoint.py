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

def create_data_struct(alpha,L):
    '''Generate the [tau,N,(src,dest)] data_struct from the configuration'''

    data_struct = []
    for i in range(L):
        data_struct.append([[0,alpha[i],(i,i)]])

    return data_struct

'----------------------------------------------------------------------------------'

def N_tracker(data_struct,beta,L):
    '''Count total particles in the worldline configuration'''
    
    # Add paths
    l = 0
    for i in range(L):
        N_flats = len(data_struct[i]) # Number of flat intervals on the site
        for k in range(N_flats):
            if k < N_flats-1:
                dtau = data_struct[i][k+1][0]-data_struct[i][k][0]
            else: # time difference between beta and last kink on the site
                dtau = beta-data_struct[i][k][0]
            
            n = data_struct[i][k][1] # particles in flat interval
            l += n*dtau
                
    # Track the total number of particles (must be between N-1 and N+1)
    N = l/beta
    
    return N

'----------------------------------------------------------------------------------'

def bh_egs(data_struct,beta,dtau,U,mu,t,L):
    ''''Count the kinks in the interval [0.4*beta,0.6*beta]'''
    
    # Store the configuration at beta/2 (need for diagonal energy)
    alpha = [0]*L
    
    # Iterate over every site and count the kinks in [beta/2-dtau,beta/2+dtau]
    n_kinks = 0 # number of kinks
    for i in range(L):
        N_flats = len(data_struct[i]) # Number of flats on site i
        for k in range(N_flats): # Ignore the initial values (known)
            tau = data_struct[i][k][0]
            if tau <= 0.5*beta:
                n_i = data_struct[i][k][1] # no. particles in the flat
                alpha[i] = n_i # Get Fock state at beta/2 (for diagonal energy)
                if tau >= 0.5*beta-dtau and tau <= 0.5*beta+dtau:
                    kink_src = data_struct[i][k][2][1] # source site of the kink
                    if i == kink_src: # This avoids double counting the kinks
                        n_kinks += 1                      
            else: break
                              
    # Calculate kinetic energy estimator
    kinetic = -n_kinks/beta
    
    # Calculate diagonal energy
    diagonal = 0  
    for i in range(L):
        n_i = alpha[i]
        diagonal += ( (U/2)*n_i*(n_i-1)-mu*n_i )                
    
    return kinetic,diagonal

'----------------------------------------------------------------------------------'

def n_pimc(data_struct,beta,L):
    '''Calculates total particle number at time slice tau=beta/2'''

    # Average particle number at slice beta/2 (for no hopping)
    n = 0
    for i in range(L):
        N_flats = len(data_struct[i]) # Number of flats on site i
        for k in range(N_flats):
            if data_struct[i][k][0] <= beta/2:
                n_i = data_struct[i][k][1] # particles on i at beta/2
            else: break
        n += n_i
               
    return n

'----------------------------------------------------------------------------------'

def n_i_pimc(data_struct,beta,L):
    '''Determine site occupation at time slice beta/2'''
    
    # Average particle number at slice beta/2 (for no hopping)
    n = [] # stores average particle number per site
    for i in range(L):
        N_flats = len(data_struct[i]) # Number of flats on site i
        for k in range(N_flats):
            if data_struct[i][k][0] <= beta/2:
                n_i = data_struct[i][k][1] # particles on i at beta/2
            else: break
        n.append(n_i)
               
    return n

'----------------------------------------------------------------------------------'

def check_worm(head_loc,tail_loc):
    '''Determine if the worldline configuration contains a worm end(s)'''
    
    if head_loc or tail_loc:
        return True
    
    else:
        return False

'----------------------------------------------------------------------------------'

def worm_insert(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,insert_worm_data,insert_anti_data):
    
    '''Inserts a worm or antiworm'''

    # Can only insert worm if there are NO wormends present
    if head_loc or tail_loc: return None

    # Randomly select a lattice site i on which to insert a worm or antiworm 
    i = np.random.randint(L)

    # Randomly select a flat tau interval at which to possibly insert worm
    N_flats = len(data_struct[i])            # Number of flats on site i
    k = np.random.randint(N_flats)           # Index of lower bound of chosen flat
    tau_prev = data_struct[i][k][0]
    if k == N_flats - 1 : tau_next = beta     # In case that the last flat is chosen
    else : tau_next = data_struct[i][k+1][0]
        
    # Calculate length of flat interval
    tau_flat = tau_next - tau_prev
    
    # Randomly choose where to insert worm head and tail on the flat interval
    tau_h = tau_prev + tau_flat*np.random.random()
    tau_t = tau_prev + tau_flat*np.random.random()

    # From the times of the ends, determine the type of proposed worm
    if tau_h > tau_t:
        insert_worm = True
    else: # antiworm
        insert_worm = False
    
    # Add to worm/antiworm PROPOSAL counters
    if insert_worm:
        insert_worm_data[1] += 1
    else:
        insert_anti_data[1] += 1
        
    # Determine the no. of particles after each worm end
    n_i = data_struct[i][k][1]  # initial number of particles in the flat interval
    if insert_worm:   
        N_after_tail = n_i + 1
        N_after_head = n_i
    else: # antiworm
        N_after_tail = n_i
        N_after_head = n_i - 1         
        
    # Reject antiworm insertion if there were no particles in the flat interval
    if n_i == 0 and not(insert_worm): return False
    
    # Reject update if worm end is inserted at the bottom kink of the flat
    if tau_h == tau_prev or tau_t == tau_prev : return False

    # Reject update if both worm ends are at the same tau
    if tau_h == tau_t : return False

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
            if k == N_flats - 1: # if selected flat is the last
                data_struct_tmp[i].append(tail_kink)
                data_struct_tmp[i].append(head_kink)
            else:
                data_struct_tmp[i].insert(k+1,tail_kink)
                data_struct_tmp[i].insert(k+2,head_kink)
        # Insert antiworm
        else:
            if k == N_flats - 1: # last flat
                data_struct_tmp[i].append(head_kink)
                data_struct_tmp[i].append(tail_kink)
            else:
                data_struct_tmp[i].insert(k+1,head_kink)
                data_struct_tmp[i].insert(k+2,tail_kink)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
    
    # Calculate the difference in diagonal energy dV = \epsilon_w - \epsilon
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
    
    # Build the Metropolis ratio (R)
    p_dw,p_iw = 0.5,0.5       # tunable delete and insert probabilities   
    R = eta**2 * N_after_tail * np.exp(-dV*(tau_h-tau_t)) * (p_dw/p_iw) * L * N_flats * tau_flat**2
        
    # Metropolis sampling
    if np.random.random() < R: # Accept
        # Insert worm
        if insert_worm:
            if k == N_flats - 1: # if selected flat is the last, use append
                data_struct[i].append(tail_kink)
                data_struct[i].append(head_kink)
            else:
                data_struct[i].insert(k+1,tail_kink)
                data_struct[i].insert(k+2,head_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            tail_loc.extend([i,k+1])
            head_loc.extend([i,k+2])

        # Insert antiworm
        else:
            if k == N_flats - 1: # last flat
                data_struct[i].append(head_kink)
                data_struct[i].append(tail_kink)
            else:
                data_struct[i].insert(k+1,head_kink)
                data_struct[i].insert(k+2,tail_kink)

            # Save ira and masha locations (site_idx, tau_idx)
            head_loc.extend([i,k+1])
            tail_loc.extend([i,k+2])
           
        # Add to ACCEPTANCE counters
        if insert_worm:
            insert_worm_data[0] += 1
        else: # insert antiworm
            insert_anti_data[0] += 1
        
        return True

    # Reject
    else:
        return False

'----------------------------------------------------------------------------------'

def worm_delete(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,delete_worm_data,delete_anti_data):

    # Can only propose worm deletion if both worm ends are present
    if not(head_loc) or not(tail_loc) : return None
    
    # Only delete if worm ends are on the same site and on the same flat interval
    if head_loc[0] != tail_loc[0] or abs(head_loc[1]-tail_loc[1]) != 1: return None

    # Retrieve the site and tau indices of where ira and masha are located
    # head_loc = [site_idx,tau_idx]
    hx = head_loc[0]
    hk = head_loc[1]
    tx = tail_loc[0]
    tk = tail_loc[1]
    
    # Retrieve the times of the worm head and tail 
    tau_h = data_struct[hx][hk][0]
    tau_t = data_struct[tx][tk][0]
    
    # No. of flat regions BEFORE worm insertion
    N_flats = len(data_struct[hx]) - 2
    
    # Identify the type of worm
    if  tau_h > tau_t : is_worm = True   # worm
    else: is_worm = False                # antiworm

    # Identify the lower and upper limits of the flat interval where the worm lives
    if is_worm:
        tau_prev = data_struct[tx][tk-1][0]
        if hk == len(data_struct[hx])-1: 
            tau_next = beta
        else: 
            tau_next = data_struct[hx][hk+1][0]
        n_i = data_struct[tx][tk-1][1]  # no. of particles originally on the flat
        N_after_tail = n_i+1
    else: # antiworm
        tau_prev = data_struct[hx][hk-1][0] 
        if tk == len(data_struct[tx])-1:
            tau_next = beta
        else:
            tau_next = data_struct[tx][tk+1][0]
        n_i = data_struct[hx][hk-1][1]
        N_after_tail = n_i
    N_after_head = N_after_tail - 1

    # Calculate the length of the flat interval and the length of the worm/antiworm
    tau_flat = tau_next - tau_prev
   
    # Add to delete worm/antiworm PROPOSAL counters
    if is_worm:
        delete_worm_data[1] += 1
    else: # delete antiworm
        delete_anti_data[1] += 1
        
    # Check if N would be conserved in canonical simulations
    if canonical:
        data_struct_tmp = deepcopy(data_struct)
    
        # Delete the worm ends
        if is_worm: # worm
            del data_struct_tmp[hx][hk] # Deletes ira
            del data_struct_tmp[tx][tk] # Deletes masha
        else: # antiworm
            del data_struct_tmp[tx][tk] # Deletes masha
            del data_struct_tmp[hx][hk] # Deletes ira

        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False 
            
    # Calculate diagonal energy difference
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
    
    # Build the Metropolis ratio (R)
    p_dw,p_iw = 0.5,0.5       # tunable delete and insert probabilities   
    R = eta**2 * N_after_tail * np.exp(-dV*(tau_h-tau_t)) * (p_dw/p_iw) * L * N_flats * tau_flat**2
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R
    
    # Metropolis sampling
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
        
        # Add to delete ACCEPTANCE counters
        if is_worm:
            delete_worm_data[0] += 1
        else: # delete antiworm
            delete_anti_data[0] += 1
        return True

    # Reject
    else : return False
    
'----------------------------------------------------------------------------------'

def worm_timeshift(data_struct,beta,head_loc,tail_loc,U,mu,L,N,canonical):

    # Reject update if there are is no worm end present
    if head_loc == [] and tail_loc == [] : return None

    # Choose which worm end to move   
    worm = True # flag that keeps track if we have worm or antiworm
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

    # For debugging
    # shift_head = False

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
    if k == len(data_struct[x])-1:
        tau_next = beta  # This covers the case when there's no kinks after the worm end
    else:
        tau_next = data_struct[x][k+1][0] #actual times
    # Get tau_prev
    tau_prev = data_struct[x][k-1][0]
    
    # From the truncated exponential distribution, choose new time of the worm end
    loc = 0
    b = tau_next - tau_prev
    if dV == 0: # uniform distribution
        r = b*np.random.random()
    else: # truncated exponential distribution
        scale = 1/abs(dV)    
        r = truncexpon.rvs(b=b/scale,scale=scale,loc=loc,size=1)[0] # time diff.
        
    if dV > 0:
        if shift_head:
            tau_new = tau_prev + r
        else:
            tau_new = tau_next - r
    else: # dV < 0
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
        
    return True

'----------------------------------------------------------------------------------'

def insertZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,insertZero_worm_data,insertZero_anti_data):
    
    # Cannot insert if there's two worm ends present
    if head_loc and tail_loc: return None

    # Randomly select site i on which to insert a zero worm or antiworm
    i = np.random.randint(L)
    
    # Determine the length of the first flat interval
    if len(data_struct[i]) == 1: # Worldline is flat throughout
        tau_flat = beta
    else:
        tau_flat = data_struct[i][1][0]
    
    # Count the number of particles originally in the flat
    n_i = data_struct[i][0][1]
        
    # Choose worm/antiworm insertion based on the worm ends present
    if not(head_loc) and not(tail_loc): # No worm ends present
        if n_i == 0: # can only insert worm if there's no particles
            insert_worm = True
            insertZero_worm_data[1] += 1 # insert worm PROPOSAL counter
            p_type = 1
        else: # insert worm or antiworm randomly
            if np.random.random() < 0.5:
                insert_worm = True
                insertZero_worm_data[1] += 1
            else: # antiworm
                insert_worm = False
                insertZero_anti_data[1] += 1  # insert anti PROPOSAL counter
            p_type = 0.5
    elif head_loc: # only worm head present, can only insert tail (antiworm)
        insertZero_anti_data[1] += 1
        if n_i == 0:
            return False # can't insert antiworm if no particles on flat
        else: # n_i != 0
            insert_worm = False # insert antiworm
            p_type = 1
    else: # only tail present, can only insert head (worm)
       insert_worm = True
       insertZero_worm_data[1] += 1
       p_type = 1            

    # Randomly choose where to insert worm end on the flat interval
    tau = tau_flat*np.random.random()
        
    # Determine the no. of particles after each worm end
    if insert_worm:
        N_after_tail = n_i + 1
        N_after_head = n_i
    else: # insert antiworm
        N_after_tail = n_i
        N_after_head = n_i - 1
        
    # Calculate diagonal energy difference dV = \epsilon_w - \epsilon
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
    
    # If there are two wormends present, the delete move has to randomly choose which to remove
    if not(head_loc) and not(tail_loc): # no ends initially
        p_wormend = 1 # we end up with only one end after insertion, deleteZero would choose this one
    else: # one worm end already present
        if insert_worm: # if insert worm (i.e, a head), the end present was a tail
            tk = tail_loc[1] # kink idx of the tail
            if tk != 1: # tail was present but not on first flat, deleteZerocannot choose it
                p_wormend = 1                
            else: # tail was present and on first flat, deleteZero can choose either end
                p_wormend = 0.5
        else: # if insert anti (i.e, a tail) the end present was a head
            hk = head_loc[1] # kink idx of the head
            if hk != 1: # head was present but not on first flat, delete cannot choose it
                p_wormend = 1                
            else: # head was present and on first flat, deleteBeta can choose either end
                p_wormend = 0.5
    
    # Build the kinks to be inserted to the data structure if the move is accepted
    if insert_worm:
        worm_end_kink = [tau,N_after_head,(i,i)]    # kinks to be inserted to
        first_flat = [0,N_after_tail,(i,i)]         # the data structure
    else:
        worm_end_kink = [tau,N_after_tail,(i,i)]
        first_flat = [0,N_after_head,(i,i)]
        
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
    
        if len(data_struct_tmp[i]) == 1: # Worldline is flat throughout
            data_struct_tmp[i].append(worm_end_kink)
        else:
            data_struct_tmp[i].insert(1,worm_end_kink)
        
        data_struct_tmp[i][0] = first_flat # Modify the first flat
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
        
    # Build the weigh ratio W'/W
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    if insert_worm:
        W = eta * np.sqrt(N_after_tail) * C_post/C_pre * np.exp(-dV*tau)
    else: # antiworm
        W = eta * np.sqrt(N_after_tail) * C_pre/C_post * np.exp(dV*tau)
        
    # Build the Metropolis Ratio  (R)  
    p_dz, p_iz = 0.5,0.5
    R = W * (p_dz/p_iz) * L * p_wormend * tau_flat / p_type

    # Metropolis sampling
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

        # Add to insertZero ACCEPTANCE counters
        if insert_worm:
            insertZero_worm_data[0] += 1
        else: # insert antiworm
            insertZero_anti_data[0] += 1
            
        return True
        
    else: # Reject
        return False

'----------------------------------------------------------------------------------'

def deleteZero(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,deleteZero_worm_data,deleteZero_anti_data):

    # Cannot delete if there are no worm ends present
    if not(head_loc) and not(tail_loc): return None
    
    # Cannot delete if there are no worm ends near zero
    if head_loc and tail_loc: # both worm ends present
        if head_loc[1] != 1 and tail_loc[1] != 1:
            return None
    elif head_loc: # only head present
        if head_loc[1] != 1:
            return None
    else: # only tail present
        if tail_loc[1] != 1:
            return None

    # Decide which worm end to delete
    if head_loc and tail_loc: # both worm ends present (at least one is on a first flat)
        if head_loc[1] == 1 and tail_loc[1] == 1: # both on first flat, choose randomly
            if np.random.random() < 0.5:
                delete_head = True
            else:
                delete_head = False # delete tail (antiworm)
            p_wormend = 0.5
        elif head_loc[1] == 1: # head on first flat, tail is thus not on a first flat
            delete_head = True # delete head (worm)
            p_wormend = 1
        else: # tail on first flat, head is thus not on a first flat
            delete_head = False # delete tail (antiworm)
            p_wormend = 1
    elif head_loc: # only head present 
        delete_head = True # delete head (worm)
        p_wormend = 1
    else: # only tail present
        delete_head = False # delete tail (antiworm)
        p_wormend = 1
        
    # Get the site and kink indices of the worm end to be deleted
    if delete_head:
        x = head_loc[0]
        k = head_loc[1]
    else: # delete tail
        x = tail_loc[0]
        k = tail_loc[1]
        
    # Get the time of the worm end
    tau = data_struct[x][k][0]
                    
    # Get tau_next
    if k == len(data_struct[x]) - 1: # worldline almost completely flat
        tau_next = beta
    else:
        tau_next = data_struct[x][k+1][0]
        
    # Calculate the length of the flat interval
    tau_flat = tau_next
       
    # Number of particles originally in the flat
    n_i = data_struct[x][k][1]
    
    # No. of particles after each worm end
    if delete_head: # delete worm
        N_after_tail = n_i+1
    else: # delete antiworm
        N_after_tail = n_i
    N_after_head = N_after_tail-1
        
    # Worm insert (the reverse update) probability of choosing between worm or antiworm
    if head_loc and tail_loc: # When insertBeta was proposed, there was one end already present
        if delete_head: # In the deleted head configuration, there must have still been a tail.
            p_type = 1 # Only a head could've been inserted
        if not(delete_head): # In the deleted tail config, there must have still been a head.
            p_type = 1 # Only a tail could've been inserted
    else: # When insertBeta was proposed, there were no worm ends present. Choose type randomly.
        if n_i==0: # If there were no particles on the flat, only a head could've been inserted.
            p_type = 1
        else: # If there were particles on the flat, either head or tail could've been inserted.
            p_type = 1/2
        
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        del data_struct_tmp[x][k]
        if delete_head:
            data_struct_tmp[x][0][1] -= 1
        else: # delete tail
            data_struct_tmp[x][0][1] += 1
    
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False 
  
    # Add to deleteZero PROPOSAL counters
    if delete_head: # delete head (delete worm)
        deleteZero_worm_data[1] += 1
    else: # delete tail (delete antiworm)
        deleteZero_anti_data[1] += 1
        
    # Calculate diagonal energy difference
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
        
    # Build the weigh ratio W'/W
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    if delete_head: # delete worm
        W = eta * np.sqrt(N_after_tail) * C_post/C_pre * np.exp(-dV*tau)
    else: # delete antiworm
        W = eta * np.sqrt(N_after_tail) * C_pre/C_post * np.exp(dV*tau)    
    
    # Build the Metropolis Ratio  (R)  
    p_dz, p_iz = 0.5,0.5
    R = W * (p_dz/p_iz) * L * p_wormend * tau_flat / p_type
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R    
    # Metropolis sampling
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
            
        # Add to deleteZero ACCEPTANCE counters
        if delete_head: # delete worm
            deleteZero_worm_data[0] += 1
        else: # delete antiworm
            deleteZero_anti_data[0] += 1
            
        return True
    
    else: # Reject 
        return False
    
'----------------------------------------------------------------------------------'

def insertBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,insertBeta_worm_data,insertBeta_anti_data):
    
    # Cannot insert if there's two worm end already present
    if head_loc and tail_loc: return None

    # Randomly select a lattice site i on which to insert a worm or antiworm
    i = np.random.randint(L)
    
    # Get the kink index of the last flat interval
    k_last = len(data_struct[i]) - 1
    
    # Determine the lower bound of the last flat of the site
    tau_prev = data_struct[i][k_last][0]
    
    # Determine the length of the last flat interval
    tau_flat = beta - tau_prev
    
    # Count the number of particles originally in the flat
    n_i = data_struct[i][k_last][1]
    
    # Choose worm/antiworm insertion based on the worm ends present
    if not(head_loc) and not(tail_loc): # No worm ends present
        if n_i == 0: # can only insert worm if there's no particles
            insert_worm = True
            insertBeta_worm_data[1] += 1 # insert worm PROPOSAL counter
            p_type = 1
        else: # if there's particles, insert worm or antiworm randomly
            if np.random.random() < 0.5:
                insert_worm = True
                insertBeta_worm_data[1] += 1
            else:
                insert_worm = False
                insertBeta_anti_data[1] += 1  # insert anti PROPOSAL counter
            p_type = 0.5
    elif tail_loc: # only worm tail present, can only insert head (antiworm)
        insertBeta_anti_data[1] += 1  # insert anti PROPOSAL counter
        if n_i == 0:
            return False
        else: # if there's particles, insert the antiworm
            insert_worm = False # insert antiworm
            p_type = 1
    else: # only head present, can only insert tail (worm)
       insert_worm = True
       insertBeta_worm_data[1] += 1
       p_type = 1   
     
    # Randomly choose where to insert worm end on the flat interval
    tau = tau_prev + tau_flat*np.random.random()
    
    # Determine the no. of particles after each worm end
    if insert_worm:
        N_after_tail = n_i + 1
        N_after_head = n_i # technically, head will not be inserted
    else: # antiworm
        N_after_head = n_i - 1
        N_after_tail = n_i # technically, tail will not be inserted
        
    # Calculate diagonal energy difference dV = \epsilon_w - \epsilon
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
    
    # If there are two wormends present, the delete move has to randomly choose which to remove
    if not(head_loc) and not(tail_loc): # no ends initially
        p_wormend = 1 # we end up with only one end after insertion, deleteZero would choose this one
    else: # one worm end already present
        if insert_worm: # if insert worm (i.e, a tail), the end present was a head
            hk_last = len(data_struct[head_loc[0]]) - 1 # last kink idx of head site
            hk = head_loc[1] # kink idx of the head
            if hk != hk_last: # head was present but not on last flat, deleteBeta cannot choose it
                p_wormend = 1                
            else: # head was present and on last flat, deleteBeta can choose either end
                p_wormend = 0.5
        else: # if insert anti (i.e, a head), the end present was a tail
            tk_last = len(data_struct[tail_loc[0]]) - 1 # last kink idx of tail site
            tk = tail_loc[1] # kink idx of the tail
            if tk != tk_last: # tail was present but not on last flat, delete cannot choose it
                p_wormend = 1                
            else: # head was present and on last flat, deleteBeta can choose either end
                p_wormend = 0.5
              
    # Build the kinks to be appended to the data structure if the move is accepted
    if insert_worm:
        worm_end_kink = [tau,N_after_tail,(i,i)]  # kinks to be inserted to
    else: # antiworm
        worm_end_kink = [tau,N_after_head,(i,i)]
        
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        data_struct_tmp[i].append(worm_end_kink)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False 
            
    # Build the weight ratio W'/W
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    if insert_worm:
        W = eta * np.sqrt(N_after_tail) * C_post/C_pre * np.exp(-dV*(beta-tau))
    else: # antiworm
        W = eta * np.sqrt(N_after_tail) * C_pre/C_post * np.exp(-dV*(tau-beta))

    # Build the Metropolis Ratio   
    p_db, p_ib = 0.5, 0.5
    R = W * (p_db/p_ib) * L * p_wormend * tau_flat / p_type

    # Metropolis sampling
    if np.random.random() < R: # Accept    
        data_struct[i].append(worm_end_kink)
                        
        # Save head and tail locations (site index, kink index)  
        if insert_worm: # insert worm
            tail_loc.extend([i,k_last+1])
        else: # insert antiworm
            head_loc.extend([i,k_last+1])

        # Add to insertZero ACCEPTANCE counters
        if insert_worm:
            insertBeta_worm_data[0] += 1
        else: # insert antiworm
            insertBeta_anti_data[0] += 1
            
        return True
        
    else: # Reject
        return False
    
'----------------------------------------------------------------------------------'

def deleteBeta(data_struct,beta,head_loc,tail_loc,U,mu,eta,L,N,canonical,deleteBeta_worm_data,deleteBeta_anti_data):

    # Cannot delete if there are no worm ends present
    if not(head_loc) and not(tail_loc): return None
    
    # Cannot delete if there are no worm ends near beta (last flat)
    if head_loc and tail_loc: # both worm ends present
        hk_last = len(data_struct[head_loc[0]]) - 1 # index of the last kink on the head site
        tk_last = len(data_struct[tail_loc[0]]) - 1 # index of the last kink on the tail site
        if head_loc[1] != hk_last and tail_loc[1] != tk_last:
            return None
    elif head_loc: # only head present
        hk_last = len(data_struct[head_loc[0]]) - 1
        if head_loc[1] != hk_last:
            return None
    else: # only tail present
        tk_last = len(data_struct[tail_loc[0]]) - 1
        if tail_loc[1] != tk_last:
            return None

    # Decide which worm end to delete
    if head_loc and tail_loc: # both worm ends present (at least one is on a last flat)
        if head_loc[1] == hk_last and tail_loc[1] == tk_last: # both on last, choose randomly
            if np.random.random() < 0.5:
                delete_head = True
            else:
                delete_head = False # delete tail (worm)
            p_wormend = 0.5
        elif head_loc[1] == hk_last: # head on last flat, tail is thus not on a last flat
            delete_head = True # delete head (antiworm)
            p_wormend = 1
        else: # tail on last flat, head is thus not on a last flat
            delete_head = False
            p_wormend = 1
    elif head_loc: # only head present (it must be on last flat if we made it here)
        delete_head = True # delete head (antiworm)
        p_wormend = 1
    else: # only tail present (on last flat)
        delete_head = False # delete tail (worm)
        p_wormend = 1
        
    # Get the site and kink indices of the worm end to be deleted
    if delete_head:
        x = head_loc[0]
        k = head_loc[1]
    else: # delete tail
        x = tail_loc[0]
        k = tail_loc[1]    
    
    # Get the time of the worm end
    tau = data_struct[x][k][0]
    
    # Get tau_prev
    tau_prev = data_struct[x][k-1][0]
        
    # Calculate the length of the flat interval
    tau_flat = beta - tau_prev
    
    # Number of particles originally in the flat
    n_i = data_struct[x][k-1][1]

    # No. of particles after each worm end
    if delete_head: # delete antiworm
        N_after_tail = n_i
    else: # delete worm
        N_after_tail = n_i+1
    N_after_head = N_after_tail-1
       
    # Worm insert (the reverse update) probability of choosing between worm or antiworm
    if head_loc and tail_loc: # When insertBeta was proposed, there was one end already present
        if delete_head: # In the deleted head configuration, there must have still been a tail.
            p_type = 1 # Only a head could've been inserted
        if not(delete_head): # In the deleted tail config, there must have still been a head.
            p_type = 1 # Only a tail could've been inserted
    else: # When insertBeta was proposed, there were no worm ends present. Choose type randomly.
        if n_i==0: # If there were no particles on the flat, only a tail could've been inserted.
            p_type = 1
        else: # If there were particles on the flat, either head or tail could've been inserted.
            p_type = 1/2 
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        del data_struct_tmp[x][k]
    
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False 
        
    # Add to deleteBeta PROPOSAL counters
    if delete_head: # delete head (antiworm)
        deleteBeta_anti_data[1] += 1
    else: # delete tail (worm)
        deleteBeta_worm_data[1] += 1
        
    # Calculate diagonal energy difference
    dV = (U/2)*(N_after_tail*(N_after_tail-1)-N_after_head*(N_after_head-1)) - mu*(N_after_tail-N_after_head)
            
    # Build the weight ratio W'/W
    C_post, C_pre = 0.5,0.5 # (sqrt) Probability amplitudes of trial wavefunction
    if not(delete_head): # delete tail (worm)
        W = eta * np.sqrt(N_after_tail) * C_post/C_pre * np.exp(-dV*(beta-tau))
    else: # delete head (antiworm)
        W = eta * np.sqrt(N_after_tail) * C_pre/C_post * np.exp(-dV*(tau-beta))

    # Build the Metropolis Ratio   
    p_db, p_ib = 0.5, 0.5
    R = W * (p_db/p_ib) * L * p_wormend * tau_flat / p_type
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R    
        
    # Metropolis sampling
    if np.random.random() < R: # Accept
        
        del data_struct[x][k]
        
        if delete_head:
            del head_loc[:]

        else: # delete tail
            del tail_loc[:]
              
        # Add to deleteBeta ACCEPTANCE counters
        if delete_head: # delete antiworm
            deleteBeta_anti_data[0] += 1
        else: # delete worm
            deleteBeta_worm_data[0] += 1
            
        return True
    
    else: # Reject 
        return False
    
'----------------------------------------------------------------------------------'

def insert_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,ikbh_data):
    
    # Update only possible if there is a worm head present
    if not(head_loc): return None
    
    # Need at least two sites for a spaceshift
    if L <= 1: return None
    
    # Add to PROPOSAL counter
    ikbh_data[1] += 1
    
    # Retrieve worm head indices (i:site,k:kink)
    i = head_loc[0]
    k = head_loc[1]
    
    # Randomly choose destination site (j) of the head
    if L == 2: # Only two sites
        j = i-1 # Hop head to the left
        if j==-1: # PBC's
            j = L-1 # head hops to last site
        p_site = 1 # probability of hopping to site j
    else: # 3 sites or more
        if np.random.random() < 0.5:
            j = i+1 # Hop head to the right
            if j == L: 
                j = 0
        else: # Head hops to the left
            j = i-1
            if j==-1:
                j = L-1 # head hops to last site
        p_site = 0.5
      
    # Retrieve the time of the worm head (and tail if present)
    tau_h = data_struct[i][k][0]
    if tail_loc:
        tau_t = data_struct[tail_loc[0]][tail_loc[1]][0]

    # Determine the lower bounds of the flat where head lives
    tau_prev_i = data_struct[i][k-1][0] # lower bound of head src site  
    
    # Retrieve the no. of particles before/after head
    n_i = data_struct[i][k][1] # after head
    n_wi = n_i+1 # before head
    
    # Determine the lower bounds of the flat where head will move to
    for idx in range(len(data_struct[j])):
        tau = data_struct[j][idx][0] # imaginary time
        n = data_struct[j][idx][1]   # particles in flat idx
        if tau < tau_h:
            tau_prev_j = tau
            n_j = n # Number of particles originally in the flat
            tau_prev_j_idx = idx
        else: break
    n_wj = n_j+1 # No. of particles on j after the particle hop
    
    # Determine the lowest time at which the kink can be inserted
    tau_min = max(tau_prev_i,tau_prev_j)
    
    # Randomly choose the time of the kink
    tau_kink = tau_min + np.random.random()*(tau_h-tau_min)
    if tau_kink == tau_min: return False
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_i,(i,j)]
        kink_j = [tau_kink,n_wj,(i,j)]
        head_kink_j = [tau_h,n_j,(j,j)]
        
        # Delete the worm end from site i
        del data_struct_tmp[i][k]
        
        # Insert kinks
        data_struct_tmp[i].insert(k,kink_i)
        data_struct_tmp[j].insert(tau_prev_j_idx+1,head_kink_j)
        data_struct_tmp[j].insert(tau_prev_j_idx+1,kink_j)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
        
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((dV_i-dV_j)*(tau_h-tau_kink))
    
    # Build the Metropolis ratio (R)
    p_dkbh,p_ikbh = 0.5,0.5
    R = W * (p_dkbh/p_ikbh) * (tau_h-tau_min)/p_site
    
    # Metropolis sampling
    if np.random.random() < R: # Accept
        
        # Add to ACCEPTANCE counter
        ikbh_data[0] += 1
    
        # Delete the worm end from site i
        del data_struct[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_i,(i,j)]
        kink_j = [tau_kink,n_wj,(i,j)]
        head_kink_j = [tau_h,n_j,(j,j)]
                
        # Insert kinks
        data_struct[i].insert(k,kink_i)
        data_struct[j].insert(tau_prev_j_idx+1,head_kink_j)
        data_struct[j].insert(tau_prev_j_idx+1,kink_j)
                       
        # Readjust head indices
        head_loc[0] = j
        head_loc[1] = tau_prev_j_idx+2
        
        # Readjust tail indices if on head dest site and at later time
        if tail_loc:
            if tail_loc[0] == j and tau_t > tau_h:
                tail_loc[1] += 2 # kink and head insertion raises tail idx by two
        
        return True
    
    else: # Reject
        return False
    
'----------------------------------------------------------------------------------'
 
def delete_kink_before_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,dkbh_data):
    
    # Update only possible if there is worm head present
    if not(head_loc): return None
    
    # Need at least two sites for a spaceshift
    if L <= 1: return None
    
    # Retrieve the head indices
    j = head_loc[0] # site (also destination site of the kink)
    k = head_loc[1] # kink
    
    # Retrieve the source site of the kink before the head
    i = data_struct[j][k-1][2][0]

    # Update only possible if there's an actual kink before the head
    if i == j: return None # i.e, the kink cannot be worm end or initial element   
    
    # Retrieve the time of the head (and tail if present)
    tau_h = data_struct[j][k][0]
    if tail_loc:
        tau_t = data_struct[tail_loc[0]][tail_loc[1]][0]
    
    # Retrieve the time of the kink
    tau_kink = data_struct[j][k-1][0]
    
    # Retrieve the time of the "kink" before the kink (i.e, the lower bound)
    tau_prev_j = data_struct[j][k-2][0]
   
    # Retrieve the no. of particles after/before worm head
    n_j = data_struct[j][k][1] # after worm head
    n_wj = n_j+1 # before worm head
    
    # Determine the lower bound of the flat region of the kink src site (i)
    for idx in range(len(data_struct[i])):
        tau = data_struct[i][idx][0] # imaginary time
        n = data_struct[i][idx][1]   # particles in the flat
        if tau < tau_kink:
            tau_prev_i = tau
            n_wi = n # Particles before the kink on the src site
            tau_prev_i_idx = idx
        else: break
    n_i = n_wi-1 # No. of particles on i after the particle hop
    
    # Determine the upper bound of the flat on site i.
    if tau_prev_i_idx+1 == len(data_struct[i])-1:
        tau_next_i = beta
    else:
        tau_next_i = data_struct[i][tau_prev_i_idx+2][0]

    # Deletion cannnot interfere w/ kinks on other site
    if tau_h >= tau_next_i: return None
    
    # Add to PROPOSAL counter
    dkbh_data[1] += 1

    # Determine the lowest time at which the kink could've been inserted
    tau_min = max(tau_prev_i,tau_prev_j)
    
    # Determine the probability that the inverse move had of hopping the head to site j
    if L == 2: # only 2 sites
        p_site = 1
    else: # more than 2 sites
        p_site = 0.5
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the kink structure on both sites
        del data_struct_tmp[j][k] # deletes the worm head from j
        del data_struct_tmp[j][k-1] # deletes the kink from j
        del data_struct_tmp[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm head kink to be moved to i
        head_kink_i = [tau_h,n_i,(i,i)]
        
        # Insert the worm kink on i
        data_struct_tmp[i].insert(tau_prev_i_idx+1,head_kink_i)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
    
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((dV_i-dV_j)*(tau_h-tau_kink))
    
    # Build the Metropolis ratio (R)
    p_dkbh,p_ikbh = 0.5,0.5
    R = W * (p_dkbh/p_ikbh) * (tau_h-tau_min)/p_site
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R    
        
    # Metropolis Sampling
    if np.random.random() < R: # Accept

        # Add to ACCEPTANCE counter
        dkbh_data[0] += 1
        
        # Delete the kink structure on both sites
        del data_struct[j][k] # deletes the worm head from j
        del data_struct[j][k-1] # deletes the kink from j
        del data_struct[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm head kink to be moved to i
        head_kink_i = [tau_h,n_i,(i,i)]
        
        # Insert the worm kink on i
        data_struct[i].insert(tau_prev_i_idx+1,head_kink_i)
        
        # Readjust head indices
        head_loc[0] = i
        head_loc[1] = tau_prev_i_idx+1

        # Readjust tail indices if on site j and at later time
        if tail_loc:
            if tail_loc[0] == j and tau_t > tau_h:
                tail_loc[1] -= 2 # kink and head insertion raises tail idx by two
        
        return True
    
    else: # Reject
        return False
    
'----------------------------------------------------------------------------------'

def insert_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,ikah_data):

    # Update only possible if there is a worm head present
    if not(head_loc): return None
    
    # Need at least two sites for a spaceshift
    if len(data_struct) <= 1: return None
    
    # Add to PROPOSAL counter
    ikah_data[1] += 1
    
    # Retrieve worm head indices (i:site,k:kink)
    i = head_loc[0]
    k = head_loc[1]
    
    # Randomly choose destination site (j) of the head
    if len(data_struct) == 2: # Only two sites
        j = i-1
        p_site = 1 # probability of hopping to site j
    else: # 3 sites or more
        if np.random.random() < 0.5:
            j = i+1 # Head hops to the right
            if j == L: # PBC's
                j = 0
        else: # Head hops to the left
            j = i-1
        p_site = 0.5
    
    # Need to make the j exclusively positive for plotting purposes
    if j==-1:
        j = L-1 # head hops to last site
      
    # Retrieve the time of the worm head (and tail if present)
    tau_h = data_struct[i][k][0]
    if tail_loc:
        tau_t = data_struct[tail_loc[0]][tail_loc[1]][0]

    # Determine the upper bounds of the flat where head lives
    if k == len(data_struct[i])-1: # head is the last "kink" on the site
        tau_next_i = beta
    else:
        tau_next_i = data_struct[i][k+1][0] # upper bound of head src site  
    
    # Retrieve the no. of particles before/after head
    n_i = data_struct[i][k][1] # after head
    n_wi = n_i+1 # before head
    
    # Determine the lower bound of the flat where head will move to
    for idx in range(len(data_struct[j])):
        tau = data_struct[j][idx][0] # imaginary time
        n = data_struct[j][idx][1]   # particles in flat idx
        if tau > tau_h:
            break
        else:
            tau_prev_j = tau
            n_wj = n # Number of particles originally in the flat
            tau_prev_j_idx = idx     
    n_j = n_wj-1 # No. of particles on j after the particle hop
    
    # Update is rejected if there were no particles on j
    if n_wj == 0: return False
    
    # Determine the upper bound of the flat where head will move to
    tau_next_j_idx = tau_prev_j_idx+1
    if tau_prev_j_idx == len(data_struct[j])-1:
        tau_next_j = beta
    else:
        tau_next_j = data_struct[j][tau_next_j_idx][0]

    # Determine the highest time at which the kink can be inserted
    tau_max = min(tau_next_i,tau_next_j)
    
    # Randomly choose the time of the kink
    tau_kink = tau_h + np.random.random()*(tau_max-tau_h)
    if tau_kink == tau_h: return False
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the worm end from site i
        del data_struct_tmp[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_i,(i,j)]
        kink_j = [tau_kink,n_wj,(i,j)]
        head_kink_j = [tau_h,n_j,(j,j)]
                
        # Insert kinks
        data_struct_tmp[i].insert(k,kink_i) # kink on i
        data_struct_tmp[j].insert(tau_next_j_idx,kink_j) # kink on j
        data_struct_tmp[j].insert(tau_next_j_idx,head_kink_j) # head kink on j
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
        
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((-dV_i+dV_j)*(tau_kink-tau_h))
    
    # Build the Metropolis ratio (R)
    p_dkah,p_ikah = 0.5,0.5
    R = W * (p_dkah/p_ikah) * (tau_max-tau_h)/p_site
    
    # Metropolis sampling
    if np.random.random() < R: # Accept
        
        # Add to ACCEPTANCE counter
        ikah_data[0] += 1
    
        # Delete the worm end from site i
        del data_struct[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_i,(i,j)]
        kink_j = [tau_kink,n_wj,(i,j)]
        head_kink_j = [tau_h,n_j,(j,j)]
                
        # Insert kinks
        data_struct[i].insert(k,kink_i) # kink on i
        data_struct[j].insert(tau_next_j_idx,kink_j) # kink on j
        data_struct[j].insert(tau_next_j_idx,head_kink_j) # head kink on j
                       
        # Readjust head indices
        head_loc[0] = j
        head_loc[1] = tau_next_j_idx
        
        # Readjust tail indices if on head dest site and at later time
        if tail_loc:
            if tail_loc[0] == j and tau_t > tau_h:
                tail_loc[1] += 2 # kink and head insertion raises tail idx by two
        
        return True
    
    else: # Reject
        return False
    
'----------------------------------------------------------------------------------'

def delete_kink_after_head(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,dkah_data):

    # Update only possible if there is worm head present
    if not(head_loc): return None
    
    # Need at least two sites for a spaceshift
    if L <= 1: return None
    
    # Cannot do update if there's nothing after the head
    if head_loc[1] == len(data_struct[head_loc[0]])-1: return None
    
    # Retrieve the head indices
    j = head_loc[0] # site (also destination site of the kink)
    k = head_loc[1] # kink
    
    # Retrieve the source site of the kink after the head
    i = data_struct[j][k+1][2][0]

    # Update only possible if there's an actual kink (not wormend) after the head
    if i == j: return None  
    
    # Retrieve the time of the head (and tail if present)
    tau_h = data_struct[j][k][0]
    if tail_loc:
        tau_t = data_struct[tail_loc[0]][tail_loc[1]][0]
    
    # Retrieve the time of the kink
    tau_kink = data_struct[j][k+1][0]
    
    # Retrieve the time of the "kink" after the kink (i.e, the upper bound)
    if k+1 == len(data_struct[j])-1:
        tau_next_j = beta
    else:
        tau_next_j = data_struct[j][k+2][0]
   
    # Retrieve the no. of particles after/before worm head
    n_j = data_struct[j][k][1] # after worm head
    n_wj = n_j+1 # before worm head
    
    # Determine the lower bound of the flat region of the kink SRC site (i)
    for idx in range(len(data_struct[i])):
        tau = data_struct[i][idx][0] # imaginary time
        n = data_struct[i][idx][1]   # particles in the flat
        if tau < tau_kink:
            tau_prev_i = tau
            n_wi = n # Particles before the kink on the src site
            tau_prev_i_idx = idx
        else: break
    n_i = n_wi-1 # No. of particles on i after the particle hop
    
    # Deletion cannnot interfere w/ kinks on other site
    if tau_h <= tau_prev_i: return None
    
    # Add to PROPOSAL counter
    dkah_data[1] += 1
    
    # Determine tau_next_i
    if tau_prev_i_idx+1 == len(data_struct[i])-1:
        tau_next_i = beta
    else:
        tau_next_i = data_struct[i][tau_prev_i_idx+2][0]
    
    # Determine the highest time at which the kink could've been inserted
    tau_max = min(tau_next_i,tau_next_j)
    
    # The inverse move (IKAH), had to choose which site head would hop to
    if L == 2: # only two sites
        p_site = 1
    else: # More than two sites
        p_site = 0.5
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the kink structure on both sites
        del data_struct_tmp[j][k+1] # deletes the kink from j
        del data_struct_tmp[j][k] # deletes the worm head from j
        del data_struct_tmp[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm head kink to be moved to i
        head_kink_i = [tau_h,n_i,(i,i)]
        
        # Insert the worm kink on i
        data_struct_tmp[i].insert(tau_prev_i_idx+1,head_kink_i)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
    
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((-dV_i+dV_j)*(tau_kink-tau_h))
    
    # Build the Metropolis ratio (R)
    p_dkah,p_ikah = 0.5,0.5
    R = W * (p_dkah/p_ikah) * (tau_max-tau_h)/p_site
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R    
    # Metropolis Sampling
    if np.random.random() < R: # Accept
        
        # Add to ACCEPTANCE counter
        dkah_data[0] += 1
        
        # Delete the kink structure on both sites
        del data_struct[j][k+1] # deletes the kink from j
        del data_struct[j][k] # deletes the worm head from j
        del data_struct[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm head kink to be moved to i
        head_kink_i = [tau_h,n_i,(i,i)]
        
        # Insert the worm kink on i
        data_struct[i].insert(tau_prev_i_idx+1,head_kink_i)
        
        # Readjust head indices
        head_loc[0] = i
        head_loc[1] = tau_prev_i_idx+1

        # Readjust tail indices if on site j and at later time
        if tail_loc:
            if tail_loc[0] == j and tau_t > tau_h:
                tail_loc[1] -= 2 # kink and head insertion raises tail idx by two
        
        return True
    
    else: # Reject
        return False

'----------------------------------------------------------------------------------'

def insert_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,ikbt_data):

    # Update only possible if there is a worm tail present
    if not(tail_loc): return None
    
    # Need at least two sites for a spaceshift
    if len(data_struct) <= 1: return None
            
    # Add to PROPOSAL counter
    ikbt_data[1] += 1
    
    # Retrieve worm tail indices (i:site,k:kink)
    i = tail_loc[0]
    k = tail_loc[1]
    
    # Randomly choose destination site (j) of the head
    if L == 2: # Only two sites
        j = i-1 # hop tail to the left
        if j==-1: # PBC's
            j = L-1 # tail hops to last site
        p_site = 1 # probability of hopping to site j
    else: # 3 sites or more
        if np.random.random() < 0.5:
            j = i+1 # hop tail to the right
            if j == L: # PBC's
                j = 0
        else: # hop tail to the left
            j = i-1
            if j==-1: # PBC's
                j = L-1 # tail hops to last site
        p_site = 0.5
                
    # Retrieve the time of the worm tail (and head if present)
    tau_t = data_struct[i][k][0]
    if head_loc:
        tau_h = data_struct[head_loc[0]][head_loc[1]][0]
        
    # Determine the lower bounds of the flat where tail lives
    tau_prev_i = data_struct[i][k-1][0] # lower bound of tail src site  
    
    # Retrieve the no. of particles before/after tail
    n_wi = data_struct[i][k][1] # after tail
    n_i = n_wi-1 # before tail
    
    # Determine the lower bound of the flat where tail will move to
    for idx in range(len(data_struct[j])):
        tau = data_struct[j][idx][0] # imaginary time
        n = data_struct[j][idx][1]   # particles in flat idx
        if tau < tau_t:
            tau_prev_j = tau
            n_wj = n # Number of particles originally in the flat
            tau_prev_j_idx = idx
        else: break
    n_j = n_wj-1 # No. of particles on j after the particle hop
    
    # Update is rejected if there were no particles on j
    if n_wj == 0: return False
        
    # Determine the lowest time at which the kink can be inserted
    tau_min = max(tau_prev_i,tau_prev_j)
    
    # Randomly choose the time of the kink
    tau_kink = tau_min + np.random.random()*(tau_t-tau_min)
    if tau_kink == tau_min: return False # very unlikely, but possible
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the worm end from site i
        del data_struct_tmp[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_wi,(j,i)]
        kink_j = [tau_kink,n_j,(j,i)]
        tail_kink_j = [tau_t,n_wj,(j,j)]
                
        # Insert kinks
        data_struct_tmp[i].insert(k,kink_i)
        data_struct_tmp[j].insert(tau_prev_j_idx+1,tail_kink_j)
        data_struct_tmp[j].insert(tau_prev_j_idx+1,kink_j)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False    
    
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((-dV_i+dV_j)*(tau_t-tau_kink))
    
    # Build the Metropolis ratio (R)
    p_dkbt,p_ikbt = 0.5,0.5
    R = W * (p_dkbt/p_ikbt) * (tau_t-tau_min)/p_site
    
    # Metropolis sampling
    if np.random.random() < R: # Accept
        
        # Add to ACCEPTANCE counter
        ikbt_data[0] += 1
    
        # Delete the worm end from site i
        del data_struct[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_wi,(j,i)]
        kink_j = [tau_kink,n_j,(j,i)]
        tail_kink_j = [tau_t,n_wj,(j,j)]
                
        # Insert kinks
        data_struct[i].insert(k,kink_i)
        data_struct[j].insert(tau_prev_j_idx+1,tail_kink_j)
        data_struct[j].insert(tau_prev_j_idx+1,kink_j)
                       
        # Readjust head indices
        tail_loc[0] = j
        tail_loc[1] = tau_prev_j_idx+2
        
        # Readjust head indices if on tail dest site and at later time
        if head_loc:
            if head_loc[0] == j and tau_h > tau_t:
                head_loc[1] += 2 # kink and tail insertion raises head idx by two
        
        return True
    
    else: # Reject
        return False
    
'----------------------------------------------------------------------------------'

def delete_kink_before_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,dkbt_data):
    
    # Update only possible if there is a worm tail present
    if not(tail_loc): return None
    
    # Need at least two sites for a spaceshift
    if len(data_struct) <= 1: return None
    
    # Retrieve the tail indices
    j = tail_loc[0] # site (also destination site of the kink)
    k = tail_loc[1] # kink
    
    # Retrieve the dest site (i) of the kink before the tail
    i = data_struct[j][k-1][2][1]

    # Update only possible if there's an actual kink before the tail
    if i == j: return None # i.e, the kink cannot be worm end or initial element   
    
    # Retrieve the time of the tail (and head if present)
    tau_t = data_struct[j][k][0]
    if head_loc:
        tau_h = data_struct[head_loc[0]][head_loc[1]][0]
    
    # Retrieve the time of the kink
    tau_kink = data_struct[j][k-1][0]
    
    # Retrieve the time of the "kink" before the kink (i.e, the lower bound)
    tau_prev_j = data_struct[j][k-2][0]
   
    # Retrieve the no. of particles after/before worm tail
    n_wj = data_struct[j][k][1] # after worm tail
    n_j = n_wj-1 # before worm tail
    
    # Determine the lower bound of the flat region of the kink on i
    for idx in range(len(data_struct[i])):
        tau = data_struct[i][idx][0] # imaginary time
        n = data_struct[i][idx][1]   # particles in the flat
        if tau < tau_kink:
            tau_prev_i = tau
            n_i = n # Particles before the kink on the src site
            tau_prev_i_idx = idx
        else: break
    n_wi = n_i+1 # No. of particles on i after the particle hop
    
    # Determine the upper bound of the flat on site i.
    if tau_prev_i_idx+1 == len(data_struct[i])-1:
        tau_next_i = beta
    else:
        tau_next_i = data_struct[i][tau_prev_i_idx+2][0]

    # Deletion cannnot interfere w/ kinks on other site
    if tau_t >= tau_next_i: return None
       
    # Add to PROPOSAL counter
    dkbt_data[1] += 1
    
    # Determine the lowest time at which the kink could've been inserted
    tau_min = max(tau_prev_i,tau_prev_j)
    
    # Determine probability of particle hopping left or right
    if len(data_struct) > 2: # 3 or more lattice sites
        p_site = 0.5
    else: # only 2 sites
        p_site = 1
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the kink structure on both sites
        del data_struct_tmp[j][k] # deletes the worm tail from j
        del data_struct_tmp[j][k-1] # deletes the kink from j
        del data_struct_tmp[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm tail kink to be moved to i
        tail_kink_i = [tau_t,n_wi,(i,i)]
        
        # Insert the worm kink on i
        data_struct_tmp[i].insert(tau_prev_i_idx+1,tail_kink_i)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
    
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((-dV_i+dV_j)*(tau_t-tau_kink))
    
    # Build the Metropolis ratio (R)
    p_dkbt,p_ikbt = 0.5,0.5
    R = W * (p_dkbt/p_ikbt) * (tau_t-tau_min)/p_site
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R    
    # Metropolis Sampling
    if np.random.random() < R: # Accept

        # Add to ACCEPTANCE counter
        dkbt_data[0] += 1
        
        # Delete the kink structure on both sites
        del data_struct[j][k] # deletes the worm tail from j
        del data_struct[j][k-1] # deletes the kink from j
        del data_struct[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm tail kink to be moved to i
        tail_kink_i = [tau_t,n_wi,(i,i)]
        
        # Insert the worm kink on i
        data_struct[i].insert(tau_prev_i_idx+1,tail_kink_i)
        
        # Readjust tail indices
        tail_loc[0] = i
        tail_loc[1] = tau_prev_i_idx+1

        # Readjust head indices if on site j and at later time
        if head_loc:
            if head_loc[0] == j and tau_h > tau_t:
                head_loc[1] -= 2 # kink and tail deletion lowers head idx by two
        
        return True
    
    else: # Reject
        return False
        
'----------------------------------------------------------------------------------'

def insert_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,ikat_data):
    
    # Update only possible if there is a worm tail present
    if not(tail_loc): return None
    
    # Need at least two sites for a spaceshift
    if len(data_struct) <= 1: return None
            
    # Add to PROPOSAL counter
    ikat_data[1] += 1
    
    # Retrieve worm tail indices (i:site,k:kink)
    i = tail_loc[0]
    k = tail_loc[1]
    
    # Randomly choose destination site (j) of the head
    if L == 2: # Only two sites
        j = i-1 # hop tail to the left
        if j==-1: # PBC's
            j = L-1 # tail hops to last site
        p_site = 1 # probability of hopping to site j
    else: # 3 sites or more
        if np.random.random() < 0.5:
            j = i+1 # hop tail to the right
            if j == L: # PBC's
                j = 0
        else: # hop tail to the left
            j = i-1
            if j==-1: # PBC's
                j = L-1 # tail hops to last site
        p_site = 0.5
        
    # Retrieve the time of the worm tail (and head if present)
    tau_t = data_struct[i][k][0]
    if head_loc:
        tau_h = data_struct[head_loc[0]][head_loc[1]][0]
        
    # Determine the upper bounds of the flat where tail lives
    if k == len(data_struct[i])-1: # head is the last "kink" on the site
        tau_next_i = beta
    else:
        tau_next_i = data_struct[i][k+1][0] # upper bound of head src site 

    # Determine the lower bounds of the flat where tail lives
    tau_prev_i = data_struct[i][k-1][0] # lower bound of tail src site 
    
    # Retrieve the no. of particles before/after tail
    n_wi = data_struct[i][k][1] # after tail
    n_i = n_wi-1 # before tail
        
    # Determine the lower bound of the flat where tail will move to
    for idx in range(len(data_struct[j])):
        tau = data_struct[j][idx][0] # imaginary time
        n = data_struct[j][idx][1]   # particles in flat idx
        if tau < tau_t:
            tau_prev_j = tau
            n_j = n # Number of particles originally in the flat
            tau_prev_j_idx = idx
        else: break
    n_wj = n_j+1 # No. of particles on j after the particle hops      
        
    # Determine the upper bound of the flat where head will move to
    tau_next_j_idx = tau_prev_j_idx+1
    if tau_prev_j_idx == len(data_struct[j])-1:
        tau_next_j = beta
    else:
        tau_next_j = data_struct[j][tau_next_j_idx][0]
        
    # Determine the highest time at which the kink can be inserted
    tau_max = min(tau_next_i,tau_next_j)
    
    # Randomly choose the time of the kink
    tau_kink = tau_t + np.random.random()*(tau_max-tau_t)
    if tau_kink == tau_t: return False    
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the worm end from site i
        del data_struct_tmp[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_wi,(j,i)]
        kink_j = [tau_kink,n_j,(j,i)]
        tail_kink_j = [tau_t,n_wj,(j,j)]
                
        # Insert kinks
        data_struct_tmp[i].insert(k,kink_i) # kink on i
        data_struct_tmp[j].insert(tau_next_j_idx,kink_j) # kink on j
        data_struct_tmp[j].insert(tau_next_j_idx,tail_kink_j) # tail kink on j
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
        
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((dV_i-dV_j)*(tau_kink-tau_t))
    
    # Build the Metropolis ratio (R)
    p_dkat,p_ikat = 0.5,0.5
    R = W * (p_dkat/p_ikat) * (tau_max-tau_t)/p_site     
    
    # Metropolis sampling
    if np.random.random() < R: # Accept
        
        # Add to ACCEPTANCE counter
        ikat_data[0] += 1
    
        # Delete the worm end from site i
        del data_struct[i][k]
        
        # Build the kinks to be inserted to each site
        kink_i = [tau_kink,n_wi,(j,i)]
        kink_j = [tau_kink,n_j,(j,i)]
        tail_kink_j = [tau_t,n_wj,(j,j)]
                
        # Insert kinks
        data_struct[i].insert(k,kink_i) # kink on i
        data_struct[j].insert(tau_next_j_idx,kink_j) # kink on j
        data_struct[j].insert(tau_next_j_idx,tail_kink_j) # tail kink on j
                       
        # Readjust tail indices
        tail_loc[0] = j
        tail_loc[1] = tau_next_j_idx
        
        # Readjust head indices if on tail dest site and at later time
        if head_loc:
            if head_loc[0] == j and tau_h > tau_t:
                head_loc[1] += 2 # kink and tail insertion raises head idx by two
        
        return True
    
    else: # Reject
        return False

'----------------------------------------------------------------------------------'

def delete_kink_after_tail(data_struct,beta,head_loc,tail_loc,t,U,mu,eta,L,N,canonical,dkat_data):

    # Update only possible if there is worm tail present
    if not(tail_loc): return None
    
    # Need at least two sites for a spaceshift
    if L <= 1: return None
    
    # Cannot delete if there's nothing after the tail
    if tail_loc[1] == len(data_struct[tail_loc[0]])-1: return None
    
    # Retrieve the tail indices
    j = tail_loc[0] # site (also src site of the kink)
    k = tail_loc[1] # kink
    
    # Retrieve the dest site (i) of the kink after the tail
    i = data_struct[j][k+1][2][1]

    # Update only possible if there's an actual kink (not wormend) after the tail
    if i == j: return None   
    
    # Retrieve the time of the tail (and head if present)
    tau_t = data_struct[j][k][0]
    if head_loc:
        tau_h = data_struct[head_loc[0]][head_loc[1]][0]
    
    # Retrieve the time of the kink
    tau_kink = data_struct[j][k+1][0]
    
    # Retrieve the time of the "kink" after the kink (i.e, the upper bound)
    if k+1 == len(data_struct[j])-1:
        tau_next_j = beta
    else:
        tau_next_j = data_struct[j][k+2][0]
   
    # Retrieve the no. of particles after/before worm tail
    n_wj = data_struct[j][k][1] # after worm tail
    n_j = n_wj-1 # before worm tail
    
    # Determine the lower bound of the flat region of the kink DEST site (i)
    for idx in range(len(data_struct[i])):
        tau = data_struct[i][idx][0] # imaginary time
        n = data_struct[i][idx][1]   # particles in the flat
        if tau < tau_kink:
            tau_prev_i = tau
            n_i = n # Particles before the kink on site i
            tau_prev_i_idx = idx
        else: break
    n_wi = n_i+1 # No. of particles on i after the particle hop
    
    # Deletion cannnot interfere w/ kinks on other site
    if tau_t <= tau_prev_i: return None
    
    # Add to PROPOSAL counter
    dkat_data[1] += 1
    
    # Determine tau_next_i
    if tau_prev_i_idx+1 == len(data_struct[i])-1:
        tau_next_i = beta
    else:
        tau_next_i = data_struct[i][tau_prev_i_idx+2][0]
    
    # Determine the highest time at which the kink could've been inserted
    tau_max = min(tau_next_i,tau_next_j)
    
    # The inverse move (IKAT), had to choose which site tail would hop to
    if L == 2: # only two sites
        p_site = 1
    else: # More than two sites
        p_site = 0.5
    
    # Check if the update would violate conservation of total particle number
    if canonical: # do the check for Canonical simulation
        data_struct_tmp = deepcopy(data_struct)
        
        # Delete the kink structure on both sites
        del data_struct_tmp[j][k+1] # deletes the kink from j
        del data_struct_tmp[j][k] # deletes the worm tail from j
        del data_struct_tmp[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm tail kink to be moved to i
        tail_kink_i = [tau_t,n_wi,(i,i)]
        
        # Insert the worm kink on i
        data_struct_tmp[i].insert(tau_prev_i_idx+1,tail_kink_i)
        
        N_check = N_tracker(data_struct_tmp,beta,L)
        if N_check <= N-1 or N_check >= N+1: return False
    
    # Calculate the diagonal energy difference on both sites
    dV_i = (U/2)*(n_wi*(n_wi-1)-n_i*(n_i-1)) - mu*(n_wi-n_i)
    dV_j = (U/2)*(n_wj*(n_wj-1)-n_j*(n_j-1)) - mu*(n_wj-n_j)
    
    # Calculate the weight ratio W'/W
    W = t * np.sqrt(n_wj*n_wi) * np.exp((dV_i-dV_j)*(tau_kink-tau_t))
    
    # Build the Metropolis ratio (R)
    p_dkat,p_ikat = 0.5,0.5
    R = W * (p_dkat/p_ikat) * (tau_max-tau_t)/p_site
    if R == 0:
        R = 1000000 # "infinity"
    else:
        R = 1/R
        
    # Metropolis Sampling
    if np.random.random() < R: # Accept
        
        # Add to ACCEPTANCE counter
        dkat_data[0] += 1
        
        # Delete the kink structure on both sites
        del data_struct[j][k+1] # deletes the kink from j
        del data_struct[j][k] # deletes the worm tail from j
        del data_struct[i][tau_prev_i_idx+1] # deletes the kink from i
    
        # Build the worm tail kink to be moved to i
        tail_kink_i = [tau_t,n_wi,(i,i)]
        
        # Insert the worm kink on i
        data_struct[i].insert(tau_prev_i_idx+1,tail_kink_i)
        
        # Readjust tail indices
        tail_loc[0] = i
        tail_loc[1] = tau_prev_i_idx+1

        # Readjust head indices if on site j and at later time
        if head_loc:
            if head_loc[0] == j and tau_h > tau_t:
                head_loc[1] -= 2 # kink and head deletion lowers tail idx by two        
        return True
    
    else: # Reject
        return False
    
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
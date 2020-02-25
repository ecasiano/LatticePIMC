def tau_resolved_energy(data_struct,beta,dtau,U,mu,t,L):
    
    '''Calculates the kinetic and diagonal energies'''


    # Generate tau_slices 
    start = dtau
    tau_slices = []
    window_size = 2*dtau
    for i in range(int(beta/window_size)):
        tau_slices.append(start)
        start += window_size

    # Generate tau slices
    tau_slices = np.linspace(0,beta,13) # [0*beta,0.1*beta,...,1*beta]
    tau_slices = tau_slices[1:-1] # do not measure at end points
    
    # Check that measurement window doesn't overlap w/ adjacent slices
    if dtau > tau_slices[1]-tau_slices[0]:
        print("WARNING: dtau overlaps adjacent time slices. Decrease dtau.")
    
    # Define the size of measurement window
    window_size = 2*dtau 
        
    # Initialize arrays that will save energies measured @ tau slices
    diagonal = np.zeros_like(tau_slices)
    kinetic = np.zeros_like(tau_slices)

    for i in range(L):
        for idx,tau_slice in enumerate(tau_slices):
            for k in range(data_struct[i]):
                
                tau = data_struct[i][k][0]
                n_i = data_struct[i][k][1]
                                
                if tau > tau_slice-dtau and tau < tau_slice+dtau and tau != 0:
                    kinetic[idx] += 1
                else: # Proceed to measure at next tau_slice
                    break
        
    #print(kinetic)
    ##sys.exit()
    
    return kinetic,diagonal
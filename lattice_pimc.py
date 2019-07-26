# Lattice Path Integral Monte Carlo
# Emanuel Casiano-Diaz

import numpy as np
import argparse
import line_profiler
import memory_profiler
from src import *

n_kinks = 2
# Initialize random bosonic configuration
L,N = 4,2   # Lattice size, total particles
x = random_boson_config(L,N)
x_list = x   # NOTE: May eventually need a 2d-array w/ the configs at each tau
print("\n")
print("The initial random configuration is: | x > = ", x)
print("\n")

# Initialize the data structure
data_struct = []
for i in range(L):
    data_struct.append([[0,x[i],(-1,-1)]]) # (tau,n,jump_direction)

# Final imaginary time
beta = 1       # beta = (1/(K_B*T))

# Set the Bose-Hubbard parameters
t = 1.0
U = 2.0

# Number of desired accept/reject steps
M = 100

# Temporary loops controlling the max amount of kinks
flag = False
for m in range(M):
    particle_jump(data_struct,beta)
    for i in range(L):
        if len(data_struct[i]) == n_kinks + 1 : flag=True
    if flag == True : break

#for m in range(M):
#    particle_jump(data_struct,beta)

# Remove the initializing tuple corresponding to tau = 0 from each site tuple
for site_tuples in data_struct:
    del site_tuples[0]

# Print out results
i = 0
for site_tuples in data_struct:
    print("i = %d"%i)
    for tau_tuples in site_tuples:
        print(tau_tuples)
    print("\n")
    i += 1

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

# Print to check that arrays were spread as intended
# print("\n")
# print("tau_list =  ", tau_list)
# print("N_list =  ", N_list)
# print("dirs_list =  ", dirs_list)

# Visualization
import matplotlib.pyplot as plt
plt.figure()

# Strategy: Plot worldlines for each site, ignoring kinks
#           Include kinks
# Plot tau=0 segments for each site
for i in range(L):
#        print(i," initial", x[i])
        # Occupation of site i
        n = x[i]
        # Select line style depending on site occupation
        if n == 0: ls,lw = ':',1.3
        elif n == 1: ls,lw = '-',1.3
        elif n == 2: ls,lw = '-',4

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
            if j + 1 < L_tau:
                tau_final = tau_list[i][j+1]
            else:
                tau_final = beta
            n = N_list[i][j]
            if n == 0: ls,lw = ':',1.3
            elif n == 1: ls,lw = '-',1.3
            elif n == 2: ls,lw = '-',4
            plt.vlines(i,tau_initial,tau_final,linestyle=ls,linewidth=lw)

            # Draw kinks
            src_site = dirs_list[i][j][0]    # Index of source site
            dest_site = dirs_list[i][j][1]  # Index of destination site

            # Wrap around site axis if kink connects first and last sites
            if (src_site == 0 and dest_site == L-1):
                plt.hlines(tau_list[i][j],-0.5,0,linewidth=1.5)
                plt.hlines(tau_list[i][j],L-1,L-1+0.5,linewidth=1.5)

            elif (src_site == L-1 and dest_site == 0):
                plt.hlines(tau_list[i][j],-0.5,0,linewidth=1.5)
                plt.hlines(tau_list[i][j],L-1,L-1+0.5,linewidth=1.5)

            else:
                plt.hlines(tau_list[i][j],src_site,dest_site,linewidth=1.5)


            #plt.hlines(tau_list[i][j],-0.5,0,linewidth=1.5)
            #plt.hlines(tau_list[i][j],L-1,L-1+0.5,linewidth=1.5)

            #plt.hlines(tau_list[i][j],src_site,dest_site,linewidth=1.5)

# Plot final segments for each site
for i in range(L):
        if len(tau_list[i]) > 0:
            n = N_list[i][-1]
 #           print(i," final", N_list[i][-1])
            tau_initial = tau_list[i][-1]
            tau_final = 1
            if n == 0: ls,lw = ':',1.3
            elif n == 1: ls,lw = '-',1.3
            elif n == 2: ls,lw = '-',4
            plt.vlines(i,tau_initial,tau_final,linestyle=ls,linewidth=lw)

plt.xticks(range(0,L))
plt.xlim(-0.5,L-1+0.5)
plt.ylim(0,1)
plt.tick_params(axis='y',which='both',left=False,right=False)
plt.tick_params(axis='x',which='both',top=False,bottom=False)
plt.xlabel(r"$i$")
plt.ylabel(r"$\tau/\beta$")
#plt.savefig("worldlines.pdf")
#plt.savefig("worldlines_bad.pdf")
plt.show()

# Format the data file
#with open("worldlines_%d_%d_%.4f_%d.dat"%(L,N,U,M),"w+") as data:
#    np.savetxt(data,np.array((tau_list,N_list,source_list,dest_list)).T,delimiter=" ",fmt="%.16f %d %d %d",header="%s %14s %3s %3s"%("tau","N","source","dest"))

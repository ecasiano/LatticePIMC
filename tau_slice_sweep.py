# Sweep over various values of tau_slice can calculate 
# canonical ground state energy
import numpy as np
import subprocess

# Set the pimc parameters
L = str(4)
N = str(4)
U = str(0)
mu = str(-2.4)
M = str(int(1.5E+06))
beta = 5.0
tau_slices = np.linspace(0,beta,20)
beta = str(beta)

for tau_slice in tau_slices:
    tau_slice = str(tau_slice)
    subprocess.call(['python','main.py',L,N,U,mu,'--beta',beta,'--tau-slice',tau_slice,'--M',M])
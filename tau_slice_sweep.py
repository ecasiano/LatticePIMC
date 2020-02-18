# Sweep over various values of tau_slice can calculate 
# canonical ground state energy
import numpy as np
import subprocess

# Set the pimc parameters
L = str(2)
N = str(2)
U = str(1)
mu = str(1)
M = str(int(5E+04))
beta = 1.0
dtau = 0.1
tau_slices = np.arange(0,beta+2*dtau,2*dtau)
beta = str(beta)
dtau = str(dtau)
for tau_slice in tau_slices:
    print("SLICE:",tau_slice)
    tau_slice = str(tau_slice)
#     subprocess.call(['python','main.py',L,N,U,mu,'--beta',beta,'--tau-slice',tau_slice,'--M',M])
    subprocess.call(['python','main.py',L,N,U,mu,'--beta',beta,'--tau-slice',tau_slice,'--M',M,'--canonical',
                    '--dtau',dtau])
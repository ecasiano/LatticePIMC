# Benchmark insertion/deletion of kinks
import pimc
import numpy as np
import time
from worldline import Kink,Worldline,Worm

# Bose-Hubbard Parameters (1D)
L = int(1)
N = L
D = 1
beta = 1

# Randomly generate Fock state at tau=0
alpha_0 = pimc.random_boson_config(L,D,N)

'------------------------------------------------------------------------------'
'''Linked list version'''

print("-----Linked List Version-----")
# Initialize list that will store worldlines of each site (paths)
start = time.time()
paths = []

# Initialize list that will store kink handles (helper_kinks)
max_num_kinks = int(2E+07)
num_kinks = 0 # Kinks IN USE
helper_kinks = [Kink(None,None,None,None,None)] * max_num_kinks

# Fill out paths and helper kinks with each site's initial kinks
for site,n in enumerate(alpha_0):
	paths.append(Worldline(n,site))
	helper_kinks[site] = paths[site].first
	num_kinks += 1

# Grow the number of kinks in the worldline
test_size = int(1000)
for i in range(test_size):

	# Randomly choose prev_kink from helper array
	r = int(np.random.random()*num_kinks)
	prev_kink = helper_kinks[r]

	# Determine length of flat interval
	if prev_kink.next is not None:
		tau_flat = prev_kink.next.tau - prev_kink.tau 
	else: 
		tau_flat = beta-prev_kink.tau

	tau = prev_kink.tau + np.random.random()*tau_flat
	n = int(np.random.random())*L
	src = int(np.random.random())*L
	dest = int(np.random.random())*L
	label = num_kinks+1
	new_kink = Kink(tau,n,src,dest,label)

	# Insert new_kink to worldline
	paths[0].insert(prev_kink,new_kink)

	# Insert kink to helper array
	helper_kinks[num_kinks] = new_kink
	num_kinks += 1

end = time.time()
print("Time elapsed to create structure with %d \
initial kinks: %.4f seconds"%(test_size,end-start))

'''----------------------------------------------'''

# Alternate between insert and delete many times
start = time.time()
insertions = int(500000)
for i in range(insertions):

	'''Insertion'''

	# Sample prev_kink (lower bound of flat interval) from helper array
	r = int(np.random.random()*num_kinks)
	prev_kink = helper_kinks[r] 

	# Determine length of flat interval
	if prev_kink.next is not None:
		tau_flat = prev_kink.next.tau - prev_kink.tau 
	else: 
		tau_flat = beta-prev_kink.tau

	# Generate kink data randomly
	tau = prev_kink.tau + np.random.random()*tau_flat
	n = 17
	src = 17
	dest = 17
	label = i+1

	# Grab first available kink from helper array and set it's values
	new_kink = helper_kinks[num_kinks]
	new_kink.tau = tau
	new_kink.n = n
	new_kink.src = src
	new_kink.dest = dest
	new_kink.label = label

	# Insert new_kink to worldline
	paths[0].insert(prev_kink,new_kink)

	# Swap the new kink such that it's the last used kink
	# helper_kinks[num_kinks-1],helper_kinks[num_kinks] = \
	# helper_kinks[num_kinks],helper_kinks[num_kinks-1]

	# Increase number of kinks in use
	num_kinks += 1

	'''Deletion'''

	# Sample kink to be deleted from helper array
	r = L + int(np.random.random()*(num_kinks-L)) # Cannot delete initial kinks

	# Delete kink from worldline
	paths[0].delete(helper_kinks[r])

	# In helper_array, swap deleted kink with last used kink
	helper_kinks[r],helper_kinks[num_kinks-1] = \
	helper_kinks[num_kinks-1],helper_kinks[r]

	# Decrease number of kinks in use
	num_kinks -= 1

end = time.time()
print("Time elapsed to perform %d inserts/deletes on \
worldline with %d initial kinks: %.4f seconds"%(insertions,test_size,end-start))

# print('\n',paths[0],'\n')
# print(helper_kinks[:num_kinks+2])

# '------------------------------------------------------------------------------'
'''Python list version'''
print("\n-----Python List Version-----")

# Initialize list that will contain kinks
start = time.time()
paths = []

# Fill out paths initial kinks
for site,n in enumerate(alpha_0):
	paths.append([[0,n,(site,site)]])

# Grow the number of kinks in the worldline
for i in range(test_size):

	# Randomly choose insertion index
	flats = len(paths[0])
	k = int(np.random.random()*flats)

	# print(paths)
	# print(flats)
	# print(k)
	# Determine length of flat interval
	if k!=flats-1:
		tau_flat = paths[0][k+1][0] - paths[0][k][0]
	else:
		tau_flat = beta - paths[0][k][0]

	tau = paths[0][k][0] + np.random.random()*tau_flat
	n = int(np.random.random())*L
	src = int(np.random.random())*L
	dest = int(np.random.random())*L

	# Build the new kink
	new_kink = [tau,n,(src,dest)]

	# Insert new_kink to worldline
	paths[0].insert(k+1,new_kink)

end = time.time()
print("Time elapsed create structure with %d \
initial kinks: %.4f seconds"%(test_size,end-start))

# Alternate between insert and delete many times
start = time.time()
for i in range(insertions):

	'''Insertion'''

	# Sample prev_kink (lower bound of flat interval) from helper array
	flats = len(paths[0])
	k = int(np.random.random()*flats)

	# Determine length of flat interval
	if k!=flats-1:
		tau_flat = paths[0][k+1][0] - paths[0][k][0]
	else:
		tau_flat = beta - paths[0][k][0]

	# Build the new kink
	new_kink = [tau,n,(src,dest)]

	# Insert new_kink to worldline
	paths[0].insert(k+1,new_kink)

	'''Deletion'''

	# Sample kink to be deleted from helper array
	flats = len(paths[0])
	k = int(np.random.random()*flats)

	# Delete kink from worldline
	del paths[0][k]

end = time.time()
print("Time elapsed to perform %d inserts/deletes on \
worldline with %d initial kinks: %.4f seconds"%(insertions,test_size,end-start))

# Old version:
# 3) Create data_struct (list version) for one site
# 4) Grow data_struct[0] to desired size
# 5) Randomly sample a flat region and use list.insert()
# 6) Randomly sample a flat region and delete lower bound kink



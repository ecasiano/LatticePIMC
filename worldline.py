# A complete working Python program to demonstrate all 
# insertion methods 
# References: https://www.geeksforgeeks.org/doubly-linked-list/,
#              https://dbader.org/blog/python-linked-list

import numpy as np

class Kink: 
  
    def __init__(self,tau,n,src,dest,label): 
        '''Kink constructor'''
        self.tau = tau
        self.n = n
        self.src = src
        self.dest = dest
        self.label = label
        self.next = None
        self.prev = None

    def __repr__(self):
        '''Return string representation of kink'''
        return f"Kink({self.tau},{self.n},{self.src},{self.dest},{self.label})"


class Worm:

    def __init__(self):
        self.head = None
        self.tail = None

class Worldline:

    def __init__(self,n,site):
        '''Initialize worldline'''
        self.first = Kink(0,n,site,site,site) # last site is the label
        self.last = self.first
        self.flats = 1

    def __repr__(self):
        '''String representation of worldline'''
        nodes = []
        curr = self.first
        while curr:
            nodes.append(repr(curr))
            curr = curr.next
        return '<' + ', '.join(nodes) + '>'

    def insert(self,prev_kink,new_kink): 
        '''Insert a new kink at time tau_new'''

        # Connect next of new kink to next_kink
        new_kink.next = prev_kink.next
  
        # Connect next of prev_kink to new_kink
        prev_kink.next = new_kink 
  
        # Conect prev of new_kink to prev_kink
        new_kink.prev = prev_kink 
  
        # Change previous of next_kink
        if new_kink.next is not None: 
            new_kink.next.prev = new_kink
        else: # New kink inserted at the end
            self.last = new_kink

        # Increase number of flats in the worldline by one
        self.flats += 1

        # return new_kink

    def delete(self,kink_to_remove):
        '''Delete a specific kink'''

        if kink_to_remove.next is not None:
            kink_to_remove.next.prev = kink_to_remove.prev
            kink_to_remove.prev.next = kink_to_remove.next
        else:
            kink_to_remove.prev.next = None
            self.last = kink_to_remove.prev

        # Unlink undesired kink
        kink_to_remove.next = None
        kink_to_remove.prev = None
        # del kink_to_remove

        # Decrease number of flats by one
        self.flats -= 1

    # def append(self,new_kink):
    #     '''Insert a kink at the end of the worldline'''

    #     # Get the last kink of the worldline
    #     last_kink = self.last

    #     # Connect last kink to new_kink
    #     last_kink.next = new_kink
    #     new_kink.prev = last_kink

    #     # Make the appended kink the last one
    #     new_kink.next = None
    #     self.last = new_kink

    #     # Increase number of flats by one
    #     self.N_flats += 1



'-----------------------------------------------------------'

# # Main

# # Create a kink object
# kink = Kink(0,1,0,0)

# print("A kink object is represented as",kink,'\n')

# # Create a wordline object
# i = 0   # Worldline site
# n_i = 1 # Initial number of particles on site i  
# worldline = Worldline(n_i,i)

# print("A worldline object is represented as",worldline,'\n')

# # Insert kink after trivial kink
# tau_new,n_new,src_new,dest_new = 0.78,2,1,0
# worldline.insert(tau_new,n_new,src_new,dest_new)

# print("Kink insertion:",worldline,'\n') 

# # Initialize worm
# worm = Worm()

# # Insert a worm between the other two kinks
# worm.head = worldline.insert(0.15*tau_new,n_i-1,i,i)  # head
# worm.tail = worldline.insert(0.2*tau_new,n_i,i,i)     # tail
# print("Worm insertion:",worldline,'\n')

# # Insert kink at the end
# worldline.insert(1.1*tau_new,1,0,1)
# print("Insert kink at the end:",worldline,'\n')

# # Append:
# worldline.append(0.999999999,2,1,0)
# print("Append kink:",worldline,'\n')

# # Delete the second to last kink
# worldline.remove(worldline.last.prev)
# print("Delete second to last kink: ", worldline,'\n')

# # Delete the worm head
# worm_head = worldline.first.next
# worldline.remove(worm_head)
# print("Delete worm head kink: ", worldline,'\n')

# # Retrieve the first and last kinks
# print("The first kink of the worldline is:",worldline.first,'\n')
# print("The last kink of the worldline is:",worldline.last,'\n')

# first_kink = worldline.first
# print(worldline.first==first_kink)
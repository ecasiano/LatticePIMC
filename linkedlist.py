# Define linked list objects

# Node class
class Node:

    #  Node object constructor
    def __init__(self,data):
        self.data = data # Data element in the node
        self.next = None # Initialize ptr to next node as None

# Linked list class
class LinkedList:

    # Linked list object constructor
    def __init__(self):
        self.head = None # The first node of the linked list

    # This function prints contents of linked list 
    # starting from head 
    def printList(self): 
        temp = self.head 
        while (temp): 
            print(temp.data) 
            temp = temp.next


# Main (to test out objects defined above)
if __name__ == '__main__':

	# Initialize an empty linked list
	llist = LinkedList()

	# Define some nodes
	llist.head = Node(0)
	second = Node(326)
	third = Node(1968)

	''' 
	Three nodes have been created. 
	We have references to these three blocks as head, 
	second and third 

	llist.head          second             third 
	     |                 |                 | 
	     |                 |                 | 
	+----+------+     +----+------+     +----+------+ 
	| 0  | None |     | 326| None |     |1968| None | 
	+----+------+     +----+------+     +----+------+ 
	'''

	# Link the first node to the second
	llist.head.next = second

	''' 
	Now next of first Node refers to second.  So they 
	both are linked. 

	llist.head         second              third 
	     |                |                  | 
	     |                |                  | 
	+----+------+     +----+------+     +----+------+ 
	| 0  |  o-------->| 326| null |     |1968| null | 
	+----+------+     +----+------+     +----+------+  
	'''

	# Link the second node to the third
	second.next = third

	''' 
	Now next of second Node refers to third.  So all three 
	nodes are linked. 

	llist.head         second              third 
	     |                |                  | 
	     |                |                  | 
	+----+------+     +----+------+     +----+------+ 
	| 0  |  o-------->| 326|  o-------->|1968| null | 
	+----+------+     +----+------+     +----+------+  
	'''

	llist.printList()
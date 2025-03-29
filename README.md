# ENPM661_Project03_Phase01

Project 3, Phase 1 A_star search algorithm
Group Members:
 - Chris Collins   UID: 110697305
 - Kyle Demmerle   UID: 121383341
 - Dan Zinobile    UID: 121354464

Dependencies:
    - pandas
    - numpy
    - matplotlib.pyplot
    - collections import deque 
    - cv2   (pip install opencv-contrib-python)
    - matplotlib.patches as patches
    - math
    - heapq
    - matplotlib import animation
    - os
    - time


GitHub: https://github.com/dzinobile/ENPM661_Project03_Phase01

We created our project 3, Phase 1 code in VS Code using code blocks. 

The first block imports the necessary libraries, and all the functions we wrote for Project 3, Phase 1

The second block is where a user can define a custom input case or if an invalid start/goal_state is entered, the code will prompt the user to input valid inputs:
    found_valid = True   
    start       = time.time()  
    algo        = "A_star"
    save_folder_path = ["Dropbox", "UMD", "ENPM_661 - Path Planning for Robots", "ENPM661_Project03_Phase01"]
    generate_random = False
    start_in = (10, 48, 30)
    goal_in  = (500, 220, 30)
    r        = 1

If found_valid is set to False, the code will prompt the user to enter a custom start/goal state
start_in/goal_in can be edited to define a valid custom case (or if invalid will be prompted to enter valid)
generate_random 


The code is executed as follows:
    Step 1: Create Map with Obstacles, Map with Obstacles + 5mm clearence and Cost Matrix
    Step 2: Get Start/ Goal State, either from user or generate random valid start/goal state
    Step 3: Run A* Search and plot HeatMap of Cost Matrix and "Heat Map" of V Matrix
    Step 4: Create Output Video of solution path and explored path


A* Algorithm:

    Data Structure Selection:
        We use a priority queue (heapq) as our Open List to add nodes to visit as we search.
        Priority queues are type of queue where each element is associated with priority and element with the highest priority is popped before an element with lower priority -- Priority queues are implemented as binary heaps that have one of the following properties:
         - Min-Heap Property: For every node in the tree, the key is Less than or equal to the keys of its children
         - Max-Heap Property: For every node in the tree, the key is Greater than or equal to the keys of its children
        Priority queues work well with algorithms like Dijkstra/A* where we need to visit nodes in order of their cost or (cost_to_come+cost_to_go)

        We use a set to track visited nodes in our Closed List: cost_to_come to track the cost to each node, assuming cost to traverse pixels is = to r from 1 to 10
        We use a dictionary to map child nodes to parent nodes to back-track our path from goal to start state
        We use a list for explored_path to track all nodes visited in order to visualize the path taken by the algorithm

    Algorithm:
        1. Initialize Open List (priority queue) and Closed List (cost_to_come dictionary)
        2. Add start state to Open List with cost to come + euclideon distance to reach goal
        3. While Open List is not empty:
            1. Pop node with lowest cost_to_come + cost_to_go from Open List
            2. Check if node is within 1.5 mm of goal state, if it is, generate path and break loop
            3. Check if node has higher cost than previously found cost, if it does, skip and continue
            4. Add node to Closed List
            5. Generate possible moves from current node
            6. For each possible move:
                1. Check if move is valid and not an obstacle
                2. Calculate cost to reach next node
                3. Check if next node has not been visited or if new cost is lower than previous cost to reach node
                4. If so, update cost_to_come, parent, and cost_matrix and add node back to Open List
                5. If not, skip and continue
            7. If no solution found, return None

    Parameters:
        start_state:        Initial state of point robot as tuple of (x, y) coordinates
        goal_state:         Goal state of point robot as tuple of (x, y) coordinates
        map_data:           Map with obstacles
        cost_matrix:        Cost matrix with obstacles as -1 and free space as infinity
        obstacles:          Set of obstacle coordinates

    Returns:     
        solution_path:      List of states from the start state to goal state
        cost_to_come:       Dictionary of cost to reach each node
        parent:             Dictionary mapping child states to parent states
        cost_matrix:        Cost matrix with updated costs to reach each node
        explored_path:      List of all nodes expanded by the algorithm in search
        V:                  Visited nodes matrix with 1 for visited nodes and 0 for unvisited nodes
        goal_state_reached: Goal state reached by the algorithm


Note: Looking forward we thought it may be important to check the orientation while we perform our A* search, this results in a large number of nodes being explored
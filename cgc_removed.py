# def action_set(node):
#     r = 5
#     theta_deg = np.linspace(0, 360, 12, endpoint=False)
#     theta     = np.deg2rad(theta_deg)
#     actions = []
#     for i in range(12):
#         x = int(node[0] + r * np.cos(theta[i]))
#         y = int(node[1] + r * np.sin(theta[i]))
#         dist = np.sqrt((x - node[0])**2 + (y - node[1])**2)
#         actions.append([(x, y, theta_deg[i]), r])

#     return actions



# %% START OF DEBUGGING
found_valid = True
start       = time.time()  
algo       = "A_star"
save_folder_path = ["Dropbox", "UMD", "ENPM_661 - Path Planning for Robots", "Project 3"]
generate_random = False
start_in = (5, 48, 30)
goal_in  = (175, 2, 30)
r = 1

# Step 1: Create Map with Obstacles, Map with Obstacles + 2mm clearence and Cost Matrix
(map_data, map_with_clearance, cost_matrix, obstacles 
)= get_map_cost_matrix() 
map_data_wit_clearence   = map_with_clearance.copy()   # Copy map_data with obstacles and clearance

end = time.time()

# Step 2: Get Start/ Goal State, either from user or generate random valid start/ goal state
if generate_random: # Generate Random Start/ Goal State
    while not found_valid:
        start_state = generate_random_state(map_data_wit_clearence, obstacles)
        goal_state  = generate_random_state(map_data_wit_clearence, obstacles)
        if is_valid_move(start_state, map_data_wit_clearence) and is_valid_move(goal_state, map_data_wit_clearence):
            found_valid = True
            print("Start State: ", start_state)
            print("Goal State: ", goal_state)
            break

        if not is_valid_move(start_state, map_data_wit_clearence):
            start_state = generate_random_state(map_data_wit_clearence, obstacles)
            
        if not is_valid_move(goal_state, map_data_wit_clearence):
            goal_state  = generate_random_state(map_data_wit_clearence, obstacles)


else: # Use User Provided Start/ Goal State
    start_state, goal_state = check_validity_with_user(map_data_wit_clearence, start_in, goal_in)# Use User Provided Start/ Goal State

# Step 3: Run Search Algorithm
start = time.time()

solution_path = None
pq            = [] # Open List
cost_to_come  = {} # Closed List
explored_path = [] # List of all nodes expanded in search
parent        = {start_state: None}  # Dictionary to map child->parent to backtrack path to goal state
f_start       = euclidean_distance(start_state, goal_state) # Heuristic function for start state 
thresh        = 0.5
V             = np.zeros(
                    (int(map_data.shape[0]/thresh),
                    int(map_data.shape[1]/thresh),
                    12)) # Visited Nodes



start_state, x_v_idx, y_v_idx, theta_v_idx    = round_and_get_v_index(start_state)
print(start_state)

start_v_idx                      = (x_v_idx, y_v_idx, theta_v_idx)
cost_to_come[start_v_idx]        = 0.0       # cost_to_come is our Closed List
cost_matrix[y_v_idx, x_v_idx]    = f_start   # we'll store cost to reach node + heuristic cost to reach goal
V[y_v_idx, x_v_idx, theta_v_idx] = 1

heapq.heappush(pq, (f_start, start_state))   # pq is our Open List

while pq:
    curr_f, curr_node = heapq.heappop(pq) # Pop node with lowest cost from priority queue

    curr_node_round, curr_x_v_idx, curr_y_v_idx, curr_theta_v_idx = round_and_get_v_index(curr_node) # Round to nearest half 
    curr_cost_node = (curr_x_v_idx, curr_y_v_idx, curr_theta_v_idx)
    
    if euclidean_distance(curr_node, goal_state) <= 1.5:              # If goal state reached, generate path from start to gaol and break the loop
        solution_path = generate_path(parent, goal_state)
        print("Found Solution to Goal:")
        print(goal_state)
        print("Cost: ", cost_to_come[curr_cost_node])
        break

    if curr_f > cost_to_come[(curr_x_v_idx, curr_y_v_idx, curr_theta_v_idx)] + euclidean_distance(curr_node, goal_state):   # If we've found lower cost for this node, 
        continue                                # skip and don't expand this node
    # else:                                     # Only add node to explored path if it is visited and expanded
    #     explored_path.append(curr_node)       # If we've found a lower cost for the node, then we have already explored it

    possible_moves = [
                        move_theta_0(      curr_node, r), 
                        move_diag_up_30(   curr_node, r), 
                        move_diag_up60(    curr_node, r),
                        move_diag_down_30( curr_node, r),
                        move_diag_down60(  curr_node, r),
                        ]
        

    for next_node, next_cost in possible_moves:   # For each move, check if it is valid and not an obstacle
        next_node_round, next_x_v_idx, next_y_v_idx, next_theta_v_idx = round_and_get_v_index(next_node)
        next_cost_node = (next_x_v_idx, next_y_v_idx, next_theta_v_idx)

        valid_move   = is_valid_move(next_node_round, map_data)
        not_obstacle = (math.floor(next_node_round[0]), math.floor(next_node_round[1])) not in obstacles

        if valid_move and not_obstacle:     # Check if next node is valid and not on top of an obstacle
            
            # We don't use our heuristic function here, we just use the cost to come to the current node + cost to reach next node
            # This is the parameter we want to minimize, but we use the heuristic function to prioritize our queue
            new_cost = cost_to_come[curr_cost_node] + next_cost

            # Check if next has not been visited or if new cost is lower than previous cost to reach node
            # For cases where we've found a lower cost to reach a node, we update the cost_to_come, parent, and cost_matrix
            # and add the node back-in to the priority queue without removing the old node, if the old node is reached again
            # we skip it with the continue statement above

            visited = check_if_visited(V, next_cost_node, r)
            if not visited:  #or new_cost < cost_to_come[next_cost_node]: 
                explored_path.append(curr_node)
                cost_to_come[next_cost_node] = new_cost
                parent[next_node]            = curr_node
                # Add Heurstic cost to reach goal to cost to come to current node for prioritization
                f_next                   = new_cost + euclidean_distance(next_node, goal_state)
                heapq.heappush(pq, (f_next, next_node))
                cost_matrix[next_y_v_idx, next_x_v_idx] = new_cost
                V[next_y_v_idx, next_x_v_idx, next_theta_v_idx] = 1

if solution_path is None:
    print("No Solution Found")

print("A_star Expanded States: ", len(explored_path))



# %%
# Run A* Algorithm
(solution_path, cost_to_come, parent, cost_matrix, explored_path
) = a_star(start_state, goal_state, map_data, cost_matrix, obstacles, r=r)

# Plot Heat Map of Cost Matrix
plot_cost_matrix(cost_matrix, start_state, goal_state, title=f"Cost Matrix Heatmap {algo}" )

if solution_path:
    # Create Videos of Solution Path and Explored Path
    solution_path_video(map_data , solution_path, save_folder_path, algo=algo)
    explored_path_video(map_data , explored_path, save_folder_path, algo=algo)

else:
    print("No solution found, aborting video generation")
end = time.time()
print("Time to Find Path, Plot Cost Matrix, and create videos: ", round((end-start), 2), " seconds")

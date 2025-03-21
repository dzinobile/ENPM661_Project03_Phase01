# import dependencies
import numpy as np
import cv2
import heapq
import itertools # for interleaving final animation frames

# Github link:
# https://github.com/dzinobile/planning/tree/main/project_2

# define map parameters
h = 50
w = 180
# buffer = 5
map = np.zeros((h, w, 3), dtype=np.uint8)

valid = False
while not valid:
    clearance = int(input("Enter clearance 0 - 6[mm]: "))
    radius = int(input("Enter robot radius 1 - 6[mm]: "))
    buffer = clearance+radius
    if clearance < 0 or clearance > 6:
        print("choose clearance between 0 and 6")
    elif radius < 1 or radius > 6:
        print("choos radius between 1 and 6")
    else:
        valid = True


# set boundary around border of map
map_limits = map.copy()
map_limits = cv2.cvtColor(map_limits, cv2.COLOR_BGR2GRAY)
map_limits = cv2.rectangle(map_limits, (0,0), (w,h), (255), buffer)
locations = np.where(map_limits == (255))

# initialize boundary sets and add the border boundary points
bfs_boundary = set(zip(locations[1], locations[0]))
dij_boundary = set(zip(locations[1], locations[0]))

# delete variables that are no longer used
del map_limits
del locations

# function for adding boundary points around obstacles
def add_boundary(inpt_x,inpt_y):
    # add given point to the boundary sets
    bfs_boundary.add((inpt_x,inpt_y))
    dij_boundary.add((inpt_x,inpt_y))

    # add all points in a box around given point to boundary sets
    for x in range(inpt_x-buffer, inpt_x+(buffer+1)):
        for y in range(inpt_y-buffer, inpt_y+(buffer+1)):
            bfs_boundary.add((x,y))
            dij_boundary.add((x,y))

# define obstacles
for y in range(17,35):
    # define obstacle E
    for x in range(20,40):
        if 16 < y <= 20:
            if 21 < x <= 39:
                # top horizontal of E
                map[y,x] = (255,0,0)
                add_boundary(x,y)

        if 20 < y <= 23:
            if 21 < x <= 25:
                # between top and middle horizontal of E
                map[y,x] = (255,0,0)
                add_boundary(x,y)

        if 23 < y <=27:
            if 21 < x <= 39:
                # middle horizontal of E
                map[y,x] = (255,0,0)
                add_boundary(x,y)

        if 27 < y <= 30:
            if 21 < x <= 25:
                # between middle and bottom horizontal of E
                map[y,x] = (255,0,0)
                add_boundary(x,y)

        if 30 < y <= 34:
            if 21 < x <= 39:
                #bottom horizontal of E
                map[y,x] = (255,0,0)
                add_boundary(x,y)
    
    # define obstacle N
    for x in range(40,60):
        if 41 < x <= 45:
            # first vertical of N
            map[y,x] = (255,0,0)
            add_boundary(x,y)

            # second vertical of N
        if 55 < x <= 59:
            map[y,x] = (255,0,0)
            add_boundary(x,y)

            # diagonal of N determined by equations of two lines
        if 45 < x <= 55:
            if (1.2*x)-38 < y <= (1.2*x)-32:
                map[y,x] = (255,0,0)
                add_boundary(x,y)
    
    # define obstacle P
    for x in range(60, 80):

        # main vertical of P
        if 61 < x <=67:
            map[y,x] = (255,0,0)
            add_boundary(x,y)

        # round part of P determined by ellipse equation
        if 67 < x <= 79:
            if (((x-67)**2)/(12**2))+(((y-22)**2)/(6**2)) <= 1:
                map[y,x] = (255,0,0)
                add_boundary(x,y)
    
    # define obstacle M
    for x in range(80,100):

        # first vertical of M
        if 81 < x <=85:
            map[y,x] = (255,0,0)
            add_boundary(x,y)
        
        # first angle of M determined by equations of two lines
        if 85 < x <= 90:
            if ((9/5)*x)-137 < y <= ((9/5)*x)-129:
                map[y,x] = (255,0,0)
                add_boundary(x,y)

        # second angle of M determined by equations of two lines
        if 90 < x <= 95:
            if ((-9/5)*x)+187 < y <= ((-9/5)*x)+195:
                map[y,x] = (255,0,0)
                add_boundary(x,y)
        
        # last vertical of M 
        if 95 < x <= 99:
            map[y,x] = (255,0,0)
            add_boundary(x,y)
    
    # define obstacle 6
    for x in range(100,120):

        # draw main circle per equation
        if ((x-110)**2)+(y-28.5)**2 <= 5.5**2:
            # add both 6's spaced 20 pixels apart
            map[y,x] = (255,0,0)
            map[y,x+20] = (255,0,0)
            add_boundary(x,y)
            add_boundary(x+20,y)

        # draw second circle at tip of "tail" of 6
        if ((x-110.85)**2)+(y-18)**2 <= 2**2:
            map[y,x] = (255,0,0)
            map[y,x+20] = (255,0,0)
            add_boundary(x,y)
            add_boundary(x+20,y)
        
        # draw part of a ring defined by two circles and two lines
        # to define the "tail" of 6
        if x-92.85 < y <=28.5:
            if 12.85**2 < (((x-121.35)**2)+((y-28.5)**2)) <= 16.85**2:
                map[y,x] = (255,0,0)
                map[y,x+20] = (255,0,0)
                add_boundary(x,y)
                add_boundary(x+20,y)

    # define obstacle 1
    for x in range(140,160):
        if 16 < y <= 20.62:
            if 144 < x <= 152:
                # top horizontal of 1
                map[y,x] = (255,0,0)
                add_boundary(x,y)
        if 20.62 < y <= 30:
            if 148 < x <= 152:
                # main vertical of 1
                map[y,x] = (255,0,0)
                add_boundary(x,y)
        if 30 < y <= 34:
            if 144 < x < 156:
                # bottom horizontal of 1
                map[y,x] = (255,0,0)
                add_boundary(x,y)
  
# define start and end points
valid = False
while not valid:
    start_x = int(input("Enter start x: "))
    start_y = 50-int(input("Enter start y: "))
    start_theta = int(input("Enter multiple of 30 for start angle [deg]: "))
    end_x = int(input("Enter goal x: "))
    end_y = 50-int(input("Enter goal y: "))
    end_theta = int(input("Enter multiple of 30 for goal angle [deg]: "))
    step_size = int(input("Enter step size from 1 to 10: "))
    
    start = (start_x, start_y, start_theta)
    end = (end_x, end_y, end_theta)
    if start_x <= 0 or start_x >= w:
        print("Start position out of bounds")
    elif start_y <= 0 or start_y >= h:
        print("Start position out of bounds")
    elif start_theta%30 != 0:
        print("Start angle must be multiple of 30")
    elif end_x <= 0 or end_x >= w:
        print("Goal position out of bounds")
    elif end_y <= 0 or end_y >= h:
        print("Goal position out of bounds")
    elif end_theta%30 != 0:
        print("Goal angle must be multiple of 30")
    elif start in bfs_boundary or start in dij_boundary:
        print("Invalid start position")
    elif end in bfs_boundary or end in dij_boundary:
        print("Invalid goal position")
    elif step_size < 1 or step_size > 10:
        print("Invalid step size")
    else:
        valid = True


# initialize lists for a star search
open_list = []
closed_list = []

# confirm open and closed lists are heapq
heapq.heapify(closed_list)
heapq.heapify(open_list)

# add start node to open list
#(total, cost to come, cost to go, parent, node)
start ctg = np.linalg.norm()
heapq.heappush(open_list, (0, start, start))


# move function for a star search 
def move(node,angle):
    parent_cost, parent = node[0], node[2] # cost and location of given parent node

    # find child location and cost based on given direction
    if direction == "up":
        child_cost, child = parent_cost+1, (parent[0],parent[1]-1)
    elif direction == "down":
        child_cost, child = parent_cost+1, (parent[0], parent[1]+1)
    elif direction == "left":
        child_cost, child = parent_cost+1, (parent[0]-1,parent[1])
    elif direction == "right":
        child_cost, child = parent_cost+1, (parent[0]+1,parent[1])
    elif direction == "up_right":
        child_cost, child = parent_cost+1.4, (parent[0]+1,parent[1]-1)
    elif direction == "down_right":
        child_cost, child = parent_cost+1.4, (parent[0]+1,parent[1]+1)
    elif direction == "down_left":
        child_cost, child = parent_cost+1.4, (parent[0]-1,parent[1]+1)
    elif direction == "up_left":
        child_cost, child = parent_cost+1.4, (parent[0]-1,parent[1]-1)
    
    # determine if child node has already been popped to closed list
    # if it has, the cost to come will be higher than the node that was popped
    if child in dij_boundary:
        return # skip if already popped
    
    # determine if there is currently a matching node with a higher cost to come
    for item in open_list:
        if item[2] == child:
            if item[0] > child_cost:
                open_list.remove(item) # replace node with new node if more efficient path found
            else:
                return # skip adding node if old node is more efficient
        
    # otherwise, push new node to open list
    heapq.heappush(open_list, (child_cost,parent,child))
    return 


i = 0 # initialize index counter at 0

# implement BFS search 
while end not in bfs_boundary:
    for direction in ["up","up_right","right","down_right","down","down_left","left","up_left"]:
        bfs_move(i,direction)
    i += 1

# backtrack through bfs list to find each subsequent parent
bfs_path = [bfs_list[-1]]
while bfs_path[-1][0] > 0:
    parent_idx = bfs_path[-1][0]
    bfs_path.append(bfs_list[parent_idx])

bfs_path.reverse() # reorder path to go from start to finish

# implement DIJ search
while end not in dij_boundary:
    parent_node = heapq.heappop(open_list) # pop lowest cost-to-come node from open list
    dij_boundary.add(parent_node[2]) # add coordinates of popped node to boundary list
    heapq.heappush(closed_list,parent_node) # push popped node to closed list
    
    # add children of popped node to open list
    for direction in ["up","up_right","right","down_right","down","down_left","left","up_left"]:
        dij_move(parent_node,direction)

# backtrack through closed list to find path
dij_path = []
path_node = end
while path_node != start:
    for item in closed_list:
        if item[2] == path_node: 
            dij_path.append(item)
            path_node = item[1]

dij_path.sort() # reorder path to go from start to finish

# create color list from red to blue evenly distributed across closed list
dij_rainbow = [(0,0,255)] # first color is red
for i in range(0,len(closed_list)):
    # divide color list into thirds
    ratio = i / (len(closed_list))

    # last third, decrease the amount of green
    if ratio < 1/3:
        B= 255
        G = int(255*(ratio*3))
        R = 0
    
    # middle third, decrease red, increase blue
    elif ratio < 2/3:
        B = int(255*(2-3*ratio))
        G = 255
        R = int(255*(3*ratio-1))
    
    # first third, increase the amount of green
    else:
        B = 0
        G = int(255*(3-3*ratio))
        R = 255
    dij_rainbow.append((B,G,R))

# create color list from red to blue evenly distributed across bfs list
bfs_rainbow = [(0,0,255)] # first color is red
for i in range(0,len(bfs_list)):

    # divide list into thirds
    ratio = i / (len(bfs_list))

    # last third, decrease the amount of green
    if ratio < 1/3:
        B= 255
        G = int(255*(ratio*3))
        R = 0
    
    # middle third, increase blue, decrease red
    elif ratio < 2/3:
        B = int(255*(2-3*ratio))
        G = 255
        R = int(255*(3*ratio-1))
    
    # first third, increase green
    else:
        B = 0
        G = int(255*(3-3*ratio))
        R = 255
    bfs_rainbow.append((B,G,R))

# initial indices for rainbow color lists
bfs_r_i = -1
dji_r_i = -1

# create board for displaying animation of both searches
board_display = np.zeros(((h*2)+40, w, 3), dtype=np.uint8)

# add white banners for showing titles
for y in range(20):
    board_display[y] = 255,255,255 

for y in range((h+20),(h+40)):
    board_display[y] = 255,255,255

# add titles
cv2.putText(board_display, "BFS Search",(5,19),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1,cv2.LINE_AA)
cv2.putText(board_display, "DJI Search",(5,89),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1,cv2.LINE_AA)

# duplicate map onto top and bottom sections for animation
for y in range(h):
    for x in range(w):
        board_display[(y+20,x)] = map[(y,x)]
        board_display[(y+90,x)] = map[(y,x)]

# draw circles indicating start and goal points
bfs_start = (start[0],start[1]+20)
dji_start = (start[0],start[1]+90)
bfs_end = (end[0],end[1]+20)
dji_end = (end[0],end[1]+90)
cv2.circle(board_display, bfs_start, 1, (255,255,255),-1)
cv2.circle(board_display, bfs_end, 1, (255,255,255),-1)
cv2.circle(board_display, dji_start, 1, (255,255,255),-1)
cv2.circle(board_display, dji_end, 1, (255,255,255),-1)

# animate bfs and dij searches 
for bfs_item, dij_item in itertools.zip_longest(bfs_list,closed_list): # interleave items from lists
    
    # update color of pixels based on rainbow lists 
    if bfs_item:
        rc = (bfs_item[1][1]+20, bfs_item[1][0])
        board_display[rc] = bfs_rainbow[bfs_r_i]
        bfs_r_i -= 1
    if dij_item:
        rc = (dij_item[2][1]+90, dij_item[2][0])
        board_display[rc] = dij_rainbow[dji_r_i]
        dji_r_i -= 1
    
    # re-draw circles to show start and goal points
    cv2.circle(board_display, bfs_start, 1, (255,255,255),-1)
    cv2.circle(board_display, bfs_end, 1, (255,255,255),-1)
    cv2.circle(board_display, dji_start, 1, (255,255,255),-1)
    cv2.circle(board_display, dji_end, 1, (255,255,255),-1)

    # display frame
    frame = board_display.copy()
    frame = cv2.resize(frame,(0,0), fx=3, fy=3)
    cv2.imshow('animation',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# animate final paths
for bfs_item, dij_item in itertools.zip_longest(bfs_path, dij_path): # interleave items from path lists
    
    # change pixels along path to white
    if bfs_item:
        rc = (bfs_item[1][1]+20,bfs_item[1][0])
        board_display[rc] = (255,255,255)
    if dij_item:
        rc = (dij_item[2][1]+90,dij_item[2][0])
        board_display[rc] = (255,255,255)
    
    # re-draw circles for start and end points
    cv2.circle(board_display, bfs_start, 1, (255,255,255),-1)
    cv2.circle(board_display, bfs_end, 1, (255,255,255),-1)
    cv2.circle(board_display, dji_start, 1, (255,255,255),-1)
    cv2.circle(board_display, dji_end, 1, (255,255,255),-1)

    # display frame
    frame = board_display.copy()
    frame = cv2.resize(frame, (0,0), fx=3, fy=3)
    cv2.imshow('animation',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# hold final frame until key is pressed and destroy all windows 
cv2.waitKey(0)
cv2.destroyAllWindows()
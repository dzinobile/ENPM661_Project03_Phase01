# Import libraries
import numpy as np
import cv2
import heapq


# Scaling function to convert mm to pixels
def scale(input):
    return input*4

# Create blank map
h = scale(50) 
w = scale(180)
map = np.zeros((h, w, 3), dtype=np.uint8)

# Get user input for clearance and radius
while True:
    clearance = scale(int(input("Enter clearance between 1 and 7 [mm]: ")))
    radius = scale(int(input("Enter robot radius between 1 and 7 [mm]: ")))
    buffer = clearance+radius
    if clearance < scale(1) or clearance > scale(7):
        print("Error: clearance must be between 1 and 7")
    elif radius < scale(1) or radius > scale(7):
        print("Error: radius must be between 1 and 7")
    else:
        break

#Create border boundary around map
map_limits = map.copy()
map_limits = cv2.cvtColor(map_limits, cv2.COLOR_BGR2GRAY)
map_limits = cv2.rectangle(map_limits, (0,0), (w,h), (255), 1)
locations = np.where(map_limits == (255))

boundary = set()
for i in range(0,len(locations[0])):
    boundary.add((locations[1][i], locations[0][i]))

del map_limits
del locations

# Function to add obstacles and buffer to boundary
def add_boundary(inpt_x,inpt_y):
    # add given point to the boundary sets
    boundary.add((inpt_x,inpt_y))

    # add all points in a circle around given point to boundary sets
    for x in range(inpt_x-buffer, inpt_x+(buffer+1)):
        for y in range(inpt_y-buffer, inpt_y+(buffer+1)):
            if (((x-inpt_x)**2)+((y-inpt_y)**2) <= (buffer)**2):
                boundary.add((x,y))
                

# Define obstacles
for y in range(scale(17),scale(35)):
    # define obstacle E
    for x in range(scale(20),scale(40)):
        if scale(16) < y <= scale(20):
            if scale(21) < x <= scale(39):
                # top horizontal of E
                add_boundary(x,y)
                map[y,x] = (255,0,0)

        if scale(20) < y <= scale(23):
            if scale(21) < x <= scale(25):
                # between top and middle horizontal of E
                add_boundary(x,y)
                map[y,x] = (255,0,0)

        if scale(23) < y <= scale(27):
            if scale(21) < x <= scale(39):
                # middle horizontal of E
                add_boundary(x,y)
                map[y,x] = (255,0,0)

        if scale(27) < y <= scale(30):
            if scale(21) < x <= scale(25):
                # between middle and bottom horizontal of E
                add_boundary(x,y)
                map[y,x] = (255,0,0)

        if scale(30) < y <= scale(34):
            if scale(21) < x <= scale(39):
                #bottom horizontal of E
                add_boundary(x,y)
                map[y,x] = (255,0,0)
    # define obstacle N
    for x in range(scale(40),scale(60)):
        if scale(41) < x <= scale(45):
            # first vertical of N
            add_boundary(x,y)
            map[y,x] = (255,0,0)

            # second vertical of N
        if scale(55) < x <= scale(59):
            add_boundary(x,y)
            map[y,x] = (255,0,0)

            # diagonal of N determined by equations of two lines
        if scale(45) < x <= scale(55):
            if ((1.2*x)-scale(38))< y <= ((1.2*x)-scale(32)):
                add_boundary(x,y)
                map[y,x] = (255,0,0)
    
    # define obstacle P
    for x in range(scale(60), scale(80)):

        # main vertical of P
        if scale(61) < x <= scale(67):
            add_boundary(x,y)
            map[y,x] = (255,0,0)

        # round part of P determined by ellipse equation
        if scale(67) < x <= scale(79):
            if (((x-scale(67))**2)/((scale(12))**2))+(((y-scale(22))**2)/((scale(6))**2)) <= 1:
                add_boundary(x,y)
                map[y,x] = (255,0,0)
    
    # define obstacle M
    for x in range(scale(80),scale(100)):

        # first vertical of M
        if scale(81) < x <= scale(85):
            add_boundary(x,y)
            map[y,x] = (255,0,0)
        
        # first angle of M determined by equations of two lines
        if scale(85) < x <= scale(90):
            if (((9/5)*x)-scale(137)) < y <= (((9/5)*x)-scale(129)):
                add_boundary(x,y)
                map[y,x] = (255,0,0)

        # second angle of M determined by equations of two lines
        if scale(90) < x <= scale(95):
            if (((-9/5)*x)+scale(187)) < y <= (((-9/5)*x)+scale(195)):
                add_boundary(x,y)
                map[y,x] = (255,0,0)
        
        # last vertical of M 
        if scale(95)< x <= scale(99):
            add_boundary(x,y)
            map[y,x] = (255,0,0)
    
    # define obstacle 6
    for x in range(scale(100),scale(120)):

        # draw main circle per equation
        if ((x-scale(110))**2)+(y-scale(28.5))**2 <= (scale(5.5))**2:
            # add both 6's spaced 20 pixels apart
            add_boundary(x,y)
            add_boundary(x+scale(20),y)
            map[y,x] = (255,0,0)
            map[y,x+scale(20)] = (255,0,0)

        # draw second circle at tip of "tail" of 6
        if ((x-scale(110.85))**2)+(y-scale(18))**2 <= (scale(2))**2:
            add_boundary(x,y)
            add_boundary(x+scale(20),y)
            map[y,x] = (255,0,0)
            map[y,x+scale(20)] = (255,0,0)

        # draw part of a ring defined by two circles and two lines
        # to define the "tail" of 6
        if x-scale(92.85) < y <= scale(28.5):
            if (scale(12.85))**2 < (((x-scale(121.35))**2)+((y-scale(28.5))**2)) <= (scale(16.85))**2:
                add_boundary(x,y)
                add_boundary(x+scale(20),y)
                map[y,x] = (255,0,0)
                map[y,x+scale(20)] = (255,0,0)

    # define obstacle 1
    for x in range(scale(140),scale(160)):
        if scale(16) < y <= scale(20.62):
            if scale(144) < x <= scale(152):
                # top horizontal of 1
                add_boundary(x,y)
                map[y,x] = (255,0,0)

        if scale(20.62) < y <= scale(30):
            if scale(148) < x <= scale(152):
                # main vertical of 1
                add_boundary(x,y)
                map[y,x] = (255,0,0)
        if scale(30) < y <= scale(34):
            if scale(144) < x < scale(156):
                # bottom horizontal of 1
                add_boundary(x,y)
                map[y,x] = (255,0,0)

# Color in buffer zone
for x,y in boundary:
    if map[y,x][0] == 0:
        map[y,x] = (0,255,0)

# Get user inputs for start/end positions and step size
while True:
    valid = True 

    start_x = scale(int(input("Enter start x [mm]: ")))
    start_y = h - scale(int(input("Enter start y [mm]: "))) # Subtract from height for origin at bottom left
    start_t = 360 - (round(((int(input("Enter start angle [deg]: ")))/30),0))*30 # Round to nearest multiple of 30 and subtract from 360 for origin at bottom left
    start_xy = (start_x,start_y)
    start = (start_x,start_y,start_t)
    
    end_x = scale(int(input("Enter goal x [mm]: ")))
    end_y = h - scale(int(input("Enter goal y [mm]: "))) # Subtract from height for origin at bottom left
    end_t = 0 # Ignoring goal angle for this assignment, default value given
    end_xy = (end_x,end_y)
    end = (end_x,end_y,end_t)

    step_size = scale(int(input("Enter step size from 1 - 10 [mm]: ")))

    # Create error message
    message = "Error: "
    if start_x <= 0 or start_x >= w:
        message = message + "\n start x out of map bounds"
        valid = False
    if start_y <= 0 or start_y >= h:
        message = message + "\n start y out of map bounds"
        valid = False
    if end_x <= 0 or end_x >= w:
        message = message + "\n goal x out of map bounds"
        valid = False
    if end_y <= 0 or end_y >= h:
        message = message + "\n goal y out of map bounds"
        valid = False
    if start_xy in boundary:
        message = message + "\n start position inside buffer zone"
        valid = False
    if end_xy in boundary:
        message = message + "\n end position inside buffer zone"
        valid = False
    if step_size < scale(1) or step_size > scale(10):
        message = message + "\n step size outside of range"
        valid = False

    if valid:
        break # Break loop if no errors

    print(message)


# Function to find euclidian distance between points
def distance(p1,p2):
    dist = np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    dist = int(dist)
    return dist

# Function to check if path between points crosses buffer zone
def line_cross(p1,p2):
    l_map = np.zeros((h, w, 1), dtype=np.uint8) # Create blank map
    cv2.line(l_map,p1,p2,(255),1) # Create white line between input points
    locations = np.where(l_map == (255)) # Find locations of white pixels
    for i in range(0,len(locations[0])):
        if (locations[1][i], locations[0][i]) in boundary: 
            return True # If line crosses over boundary, return True

# Initialize open and closed lists
open_list = []
closed_list = []
heapq.heapify(closed_list)
heapq.heapify(open_list)

# Push start node to open list
start_ctg = distance(start,end)
heapq.heappush(open_list, (start_ctg, 0, start_ctg, start, start))

# A star algorithm 
def move(node,angle):

    # Parent node information
    p_ctc = node[1]
    p_x = node[4][0]
    p_y = node[4][1]
    p_coord = node[4]
    p_xy = (p_x, p_y)

    # Child node information
    c_x = int(p_x+(step_size*np.cos(np.deg2rad(angle))))
    c_y = int(p_y+(step_size*np.sin(np.deg2rad(angle))))
    c_t = angle
    c_coord = (c_x, c_y, c_t)
    c_xy = (c_x, c_y)

    # Calculate child node cost to come, cost to go, and total cost
    c_ctc = p_ctc+step_size
    c_ctg = distance(c_coord,end)
    c_tot = c_ctc+c_ctg

    # Ignore node if path crosses buffer zone
    if line_cross(p_xy,c_xy):
        return
    
    # Replace node in open list with child node if lower total cost node found within 0.5 mm
    for item in open_list:
        item_xy = (item[4][0],item[4][1])        
        if distance(c_xy,item_xy) <= scale(0.5):
            if item[0] > c_tot:
                open_list.remove(item)
            else:
                return
            
    # Push child node to open list
    heapq.heappush(open_list, (c_tot, c_ctc, c_ctg, p_coord, c_coord))
    return

# Initial distance from goal
dist_to_goal = start_ctg

# Execute A star algorithm
while True:
    parent_node = heapq.heappop(open_list) # Pop lowest cost node from open list
    pxy = (parent_node[4][0],parent_node[4][1])
    boundary.add(parent_node[4]) # Add node location to boundary set
    heapq.heappush(closed_list, parent_node)
    dist_to_goal = distance(pxy,end_xy)
    

    # print(dist_to_goal)

    for item in [-2, -1, 0, 1, 2]:
        ang = parent_node[4][2] + (30*item)
        ang = ang%360
        move(parent_node,ang)
    
    if dist_to_goal < scale(1.5):
        break

# Find final path
final_path = []
path_node = closed_list[-1][4]
while path_node != start:
    for item in closed_list:
        if item[4] == path_node: 
            final_path.append(item)
            path_node = item[3]

final_path.sort() # reorder path to go from start to finish

# Animate search
map_display = map.copy()

for item in closed_list:
    par_xy = (item[3][0],item[3][1])
    chi_xy = (item[4][0],item[4][1])
    cv2.line(map_display,par_xy,chi_xy,(255,255,255),1)
    cv2.circle(map_display,start_xy,int(scale(1.5)),(0,0,255),1)
    cv2.circle(map_display,end_xy,int(scale(1.5)),(0,0,255),-1)
    frame = map_display.copy()
    cv2.imshow('animation',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for item in final_path:
    par_xy = (item[3][0],item[3][1])
    chi_xy = (item[4][0],item[4][1])
    cv2.line(map_display,par_xy,chi_xy,(0,0,255),1)
    frame = map_display.copy()
    cv2.imshow('animation',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
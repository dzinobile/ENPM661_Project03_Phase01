import numpy as np
import cv2
import heapq

# Scaling function to convert mm to pixels
scale_factor = 4

def scale(input):
    return input * scale_factor

# Create blank map
h = scale(50)
w = scale(180)
map = np.zeros((h, w, 3), dtype=np.uint8)

# Get user input for clearance and radius
while True:
    clearance = scale(int(input("Enter clearance between 1 and 7 [mm]: ")))
    radius = scale(int(input("Enter robot radius between 1 and 7 [mm]: ")))
    buffer = clearance + radius
    if clearance < scale(1) or clearance > scale(7):
        print("Error: clearance must be between 1 and 7")
    elif radius < scale(1) or radius > scale(7):
        print("Error: radius must be between 1 and 7")
    else:
        break

# Create border boundary around map
map_limits = map.copy()
map_limits = cv2.cvtColor(map_limits, cv2.COLOR_BGR2GRAY)
map_limits = cv2.rectangle(map_limits, (0, 0), (w, h), (255), 1)
locations = np.where(map_limits == (255))
boundary = set()
for i in range(0, len(locations[0])):
    boundary.add((locations[1][i], locations[0][i]))
del map_limits
del locations

# Define obstacles
for y in range(scale(17), scale(35), scale_factor):
    # Define obstacle E
    for x in range(scale(20), scale(40), scale_factor):
        if scale(16) < y <= scale(20):
            if scale(21) < x <= scale(39):
                # Top horizontal of E
                boundary.add((x, y))
                map[y, x] = (255, 0, 0)
        # Add other obstacle definitions similarly...

# Function to add obstacles and buffer to boundary
def add_boundary(inpt_x, inpt_y):
    boundary.add((inpt_x, inpt_y))
    for x in range(inpt_x - buffer, inpt_x + (buffer + 1)):
        for y in range(inpt_y - buffer, inpt_y + (buffer + 1)):
            if ((x - inpt_x) ** 2) + ((y - inpt_y) ** 2) <= (buffer) ** 2:
                boundary.add((x, y))

# Get user inputs for start/end positions and step size
while True:
    valid = True
    start_x = input("Enter start x [mm]: ")
    if start_x == '':
        print("Error")
        valid = False
    if valid:
        start_x = scale(int(start_x))
        start_y = h - scale(int(input("Enter start y [mm]: ")))  # Subtract from height for origin at bottom left
        start_t = 360 - (round(((int(input("Enter start angle [deg]: "))) / 30), 0)) * 30  # Round to nearest multiple of 30 and subtract from 360 for origin at bottom left
        start_xy = (start_x, start_y)
        start = (start_x, start_y, start_t)

        end_x = scale(int(input("Enter goal x [mm]: ")))
        end_y = h - scale(int(input("Enter goal y [mm]: ")))  # Subtract from height for origin at bottom left
        end_t = 0  # Ignoring goal angle for this assignment, default value given
        end_xy = (end_x, end_y)
        end = (end_x, end_y, end_t)

        step_size = scale(int(input("Enter step size from 1 - 10 [mm]: ")))  # Create error message

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
            break  # Break loop if no errors
    print(message)

# Function to find Euclidean distance between points
def distance(p1, p2):
    dist = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    dist = int(dist)
    return dist

# Function to check if path between points crosses buffer zone
def line_cross(p1, p2):
    l_map = np.zeros((h, w, 1), dtype=np.uint8)  # Create blank map
    cv2.line(l_map, p1, p2, (255), 1)  # Create white line between input points
    locations = np.where(l_map == (255))  # Find locations of white pixels
    for i in range(0, len(locations[0])):
        if (locations[1][i], locations[0][i]) in boundary:
            return True  # If line crosses over boundary, return True

# Initialize open and closed lists
open_list = []
closed_list = []

# Push start node to open list
start_ctg = distance(start[:2], end[:2])  # Heuristic cost
start_g = 0  # Path cost
start_f = start_g + start_ctg  # Total estimated cost
heapq.heappush(open_list, (start_f, start_g, start_ctg, start, start))

# A star algorithm
def move(node):
    # Parent node information
    p_ctc = node[1]  # Path cost
    p_x = node[4][0]
    p_y = node[4][1]
    p_coord = node[4]

    # Child node information
    for angle in [-2, -1, 0, 1, 2]:
        c_x = int(p_x + (step_size * np.cos(np.deg2rad(angle + node[4][2]))))
        c_y = int(p_y + (step_size * np.sin(np.deg2rad(angle + node[4][2]))))
        c_t = (angle + node[4][2]) % 360
        c_coord = (c_x, c_y, c_t)
        c_xy = (c_x, c_y)

        # Calculate child node cost to come, cost to go, and total cost
        c_ctc = p_ctc + step_size
        c_ctg = distance(c_coord[:2], end[:2])
        c_tot = c_ctc + c_ctg

        child_node = (c_tot, c_ctc, c_ctg, p_coord, c_coord)

        # Ignore node if path crosses buffer zone
        if c_x < 1 or c_x > w:
            continue
        if c_y < 1 or c_y > h:
            continue
        if line_cross(p_coord[:2], c_xy):
            continue

        # Check if node is already explored
        if (c_xy, c_t) in [(n[4][:2], n[4][2]) for n in closed_list]:
            continue

        # Add child node to open list if not explored or if it has a lower cost
        if (c_xy, c_t) in [(n[4][:2], n[4][2]) for n in open_list]:
            for i, n in enumerate(open_list):
                if n[4][:2] == c_xy and n[4][2] == c_t:
                    if n[0] > c_tot:
                        open_list[i] = child_node
                    else:
                        continue
        else:
            heapq.heappush(open_list, child_node)

# Execute A star algorithm
while open_list:
    parent_node = heapq.heappop(open_list)  # Pop lowest cost node from open list
    heapq.heappush(closed_list, parent_node)

    # Check if we've reached the goal
    if distance(parent_node[4][:2], end[:2]) < scale(1.5):
        break

# Find final path
final_path = []
path_node = closed_list[-1][4]
while path_node != start:
    for item in closed_list:
        if item[4] == path_node:
            final_path.append(item)
            path_node = item[3]
final_path.sort()  # Reorder path to go from start to finish

# Animate search
map_display = map.copy()
for item in closed_list:
    par_xy = (item[3][0], item[3][1])
    chi_xy = (item[4][0], item[4][1])
    cv2.line(map_display, par_xy, chi_xy, (255, 255, 255), 1)
cv2.circle(map_display, start[:2], int(scale(1.5)), (0, 0, 255), 1)
cv2.circle(map_display, end[:2], int(scale(1.5)), (0, 0, 255), -1)

for item in final_path:
    par_xy = (item[3][0], item[3][1])
    chi_xy = (item[4][0], item[4][1])
    cv2.line(map_display, par_xy, chi_xy, (0, 0, 255), 1)

cv2.imshow('animation', map_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
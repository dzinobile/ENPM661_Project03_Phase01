# Import libraries
import numpy as np
import cv2
import heapq

# Scaling function to convert mm to pixels
def scale(input):
    return input*2

# Create blank map
h = scale(50) 
w = scale(180)
map = np.zeros((h, w, 3), dtype=np.uint8)

# Get user input for clearance and radius
while True:
    clearance = scale(int(input("Enter clearance between 1 and 7 [mm]: ")))
    radius = scale(int(input("Enter robot radius between 1 and 7 [mm]: ")))
    buffer = clearance+radius
    if clearance < 1 or clearance > 7:
        print("Error: clearance must be between 1 and 7")
    elif radius < 1 or radius > 7:
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
                map[y,x] = (0,255,0)

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


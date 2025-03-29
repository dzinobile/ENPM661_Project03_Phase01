#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:43:50 2025

@author: kyle
"""

#############################
# Dependencies
#############################
import pygame
import time
import copy
import math
import heapq
import numpy as np
from queue import PriorityQueue
from collections import deque

###########################
# Global Variables
###########################

# Screen dimensions
rows, cols = (615,250)

# Define Lists
OL = []
CL = {}
index_ctr = 0
solution = []
thresh = 0.5
clearance = 5
parent = {}
costsum = {}
V = np.zeros((int(rows/thresh), int(cols/thresh), 12))
C2C = np.zeros((int(rows/thresh), int(cols/thresh)))

# Define colors
pallet = {"white":(255, 255, 255), 
          "black":(0, 0, 0), 
          "green":(4,217,13), 
          "blue":(61,119,245), 
          "red":(242,35,24)
          }


# Check if x_prime is in the Open List
# Pops all items from OL, checks each to see if coordinates have already
# been visited, appends popped item to temp_list, then puts all items 
# back into OL
# Returns: True is present, False if not
def CheckOpenList(coords, open_list):
    temp_list = []
    present = False
    
    while open_list:
        item = heapq.heappop(open_list)
        if item[1] == coords:
            present = True
        heapq.heappush(temp_list, item)
        
    return present, temp_list

# Check if x_prime is in the Closed List
# Closed list is a dictionary with each node's coords used as a key
# If x_prime coordinates are not in the closed list, a KeyError is thrown, 
# which is caught by the try-except block.
# Returns: True if present, False if not
def CheckClosedList(coords, closed_list):
    
    try:
        if(closed_list[coords]):
            return True
    except KeyError:
        return False

def get_theta_index(theta):
    theta = theta % 360
    look_up_dict = {
        0:   0,
        30:  1,
        60:  2,
        90:  3,
        120: 4,
        150: 5,
        180: 6,
        210: 7,
        240: 8,
        270: 9,
        300: 10,
        330: 11
    }
    return look_up_dict[theta]

def round_and_get_v_index(node):
   #  Round x, y coordinates to nearest half to ensure we are on the center of a pixel
   
   x           = round(node[0] * 2, 1) / 2
   y           = round(node[1] * 2, 1) / 2
   theta       = node[2]
   x_v_idx     = int(x * 2)
   y_v_idx     = int(y * 2)
   theta_v_idx = get_theta_index(theta)

   return (x, y, theta), x_v_idx, y_v_idx, theta_v_idx

def get_xy(node, move_theta, r):
    
    theta = node[2] + move_theta
    x     = node[0] + r * np.cos(np.deg2rad(theta))
    y     = node[1] + r * np.sin(np.deg2rad(theta))

    return (x, y, theta)

def move_theta_0(node, r):
    theta =  0
    return get_xy(node, theta, r), r


def move_diag_up_30(node, r):
    theta = 30
    return get_xy(node, theta, r), r

def move_diag_up_60(node, r):
    theta = 60
    return get_xy(node, theta, r), r

def move_diag_down_30(node, r):
    theta = -30
    return get_xy(node, theta, r), r

def move_diag_down_60(node, r):
    theta = -60
    return get_xy(node, theta, r), r


# Define the object space for all letters/numbers in the maze
# Also used for determining if a set of (x,y) coordinates is present in 
# the action space
# Returns: True if in Object Space, False if not
def InObjectSpace(x, y):
        
    # Define Object space for E
    if (((45<=x<=60) and (65<=y<=200)) or \
    ((60<=x<=84) and ((65<=y<=90) or (120<=y<=145) or (175<=y<=200)))):
        return True
    
    # Define Object Space for N
    elif (((99<=x<=114) and (65<=y<=200)) or \
    ((y-2.92*x+228.75 <= 0) and (y-2.92*x+272.5 >= 0) and (114<=x<=147) and (65<=y<=200)) or \
    ((147<=x<=162) and (65<=y<=200))):
        return True
    
    # Define Object Space for P
    elif (((177<=x<=192) and (65<=y<=200)) or ((((x-198)**2 + (y-99)**2 - 1225)<0)) and 192<=x<=235):
        return True
    
    # Define Object Space for M
    elif((((250<=x<=265) or (342<=x<=357)) and (65<=y<=200)) or \
        ((y-3.04*x+700.87<=0) and (y-3.04*x+746.52>=0) and (266<=x<=340) and (65<=y<=200)) or \
        ((y+3.04*x-1146.87<=0) and (y+3.04*x-1100.52>=0) and (267<=x<=341) and (65<=y<=200))):
        return True
    
    # Define Object Space for 6
    elif(((((x-410)**2+(y-160)**2 - 1444)<=0)and(((x-410)**2+(y-160)**2 - 100)>=0)) or \
        ((372<=x<=390) and (65<=y<=165)) or \
        ((387<=x<=410) and (65<=y<=75))):
        return True
    
    # Define Object Space for 6
    elif (((((x-501)**2+(y-160)**2 - 1444)<=0)and(((x-501)**2+(y-160)**2 - 100)>=0)) or \
          ((463<=x<=481) and (65<=y<=165)) or \
          ((478<=x<=501) and (65<=y<=75))):
        return True
    
    # Define Object Space for 1
    elif((554<=x<=569) and (65<=y<=200)):
        return True
    
    # Define Object Space for walls
    elif((x==0 and 0<=y<=249) or (x==614 and (0<=y<=249)) or \
        ((0<=x<=614) and y==0) or ((0<=x<=614) and y==249)):
        return True

    # Default case, non-object space    
    else:
        return False

# Backtrace the solution path from the goal state to the initial state
# Add all nodes to the "solution" queue
def GeneratePath(CurrentNode, parent, start_state):
    solution = []
    backtracing = True
    
    solution.append(CurrentNode)
    node = parent[CurrentNode]
    
    while backtracing:
        if(node == start_state):
            solution.append(node)
            backtracing = False
        else:
            solution.append(node)
            node = parent[node]
            
    solution.reverse()
    return solution

def euclidean_distance(node, goal_state):
    # Calculate Euclidean Distance between current node and goal state
    # Euclidean Distance is the straight line distance between two points
    # distance metric used in A* Search

    return math.sqrt((goal_state[0] - node[0])**2 + (goal_state[1] - node[1])**2)

# Draw the initial game board, colors depict:
# White: In object space
# Green: Buffer Zone
# Black: Action Space
# Turn clearance and robot radius used to calculate the total buffer zone that will be placed arround the walls and objects
# Robot will still be depicted on the screen as a "point-robot" as the center-point of the robot is the most import part for calculations
#
# Inputs:
#    rows:    x-axis size
#    cols:    y-axis size
#    pxarray: pixel array for screen that allows for drawing by point
#    pallet:  color dictionary with rgb codes
#    C2C:     (obsolete, not used here)
#    clear:   clearance needed for turns in mm
#    r:       robot raidus in mm
#
# Outputs: none
def DrawBoard(rows, cols, pxarray, pallet, C2C, clear, r):
    buff_mod = clear+r
    for x in range(0,rows):
        for y in range(0,cols):
            in_obj = InObjectSpace(x,y)
            if (in_obj):
                pxarray[x,y] = pygame.Color(pallet["white"])
            else:
                if(((InObjectSpace(x+buff_mod,y)) or\
                   (InObjectSpace(x-buff_mod,y)) or\
                   (InObjectSpace(x,y+buff_mod)) or\
                   (InObjectSpace(x,y-buff_mod)) or\
                   (InObjectSpace(x+buff_mod,y+buff_mod)) or\
                   (InObjectSpace(x-buff_mod,y-buff_mod)) or\
                   (InObjectSpace(x+buff_mod,y-buff_mod)) or\
                   (InObjectSpace(x-buff_mod,y+buff_mod))) and ((buff_mod<x<(613-buff_mod) and (buff_mod<y<(248-buff_mod))))):
                    pxarray[x,y] = pygame.Color(pallet["green"])
                elif(0<x<=buff_mod or (614-buff_mod)<=x<614 or 0<y<=buff_mod or (248-buff_mod)<=y<249):
                     pxarray[x,y] = pygame.Color(pallet["green"])
                else:
                    pxarray[x,y] = pygame.Color(pallet["black"])

def FillCostMatrix(C2C, pxarray, pallet, thresh):
    for x in range(0, int(rows/thresh)):
        for y in range(0, int(cols/thresh)):
            if((pxarray[int(math.floor(x*thresh)), int(math.floor(y*thresh))] == screen.map_rgb(pallet["white"])) or \
               (pxarray[int(math.floor(x*thresh)), int(math.floor(y*thresh))] == screen.map_rgb(pallet["green"]))):
                C2C[x,y] = -1
            else:
                C2C[x,y] = np.inf
                
#%%                    
def A_Star(start_node, goal_node, OL, parent, V, C2C, costsum, step):

    solution_path = []
    
    start_state, x_v_idx, y_v_idx, theta_v_idx = round_and_get_v_index(start_node[1])
    start_cost_state = (x_v_idx, y_v_idx, theta_v_idx)
    C2C[x_v_idx, y_v_idx] = 0.0
    costsum[start_cost_state] = 0.0 + euclidean_distance(start_node[1], goal_node)

    heapq.heappush(OL, start_node)
    
    while OL:
        node = heapq.heappop(OL)
        
        # Take popped node and center it, along with finding the index values for the V matrix
        fixed_node, x_v_idx, y_v_idx, theta_v_idx = round_and_get_v_index(node[1])
        
        # Add popped node to the closed list
        V[x_v_idx, y_v_idx, theta_v_idx] = 1
        pxarray[int(round(fixed_node[0])),int(round(fixed_node[1]))] = pygame.Color(pallet["blue"])
        pygame.display.update()
        
        # Check if popped node is within the goal tolerance region
        # If so, end exploration and backtrace to generate the soluion path
        # If not, apply action set to node and explore the children nodes
        if(euclidean_distance(fixed_node, goal_node) <= 1.5 ):
            solution_path = GeneratePath(fixed_node, parent, start_state)
            return True, solution_path
        else:
            actions = [move_diag_up_60(fixed_node, step),
                       move_diag_up_30(fixed_node, step),
                       move_theta_0(fixed_node, step),
                       move_diag_down_30(fixed_node, step),
                       move_diag_down_60(fixed_node, step)
                       ]
            
            # Walk through each child node created by action set and determine if it has been visited or not
            for child_node, child_cost in actions:
                child_node_fixed, child_x_v_idx, child_y_v_idx, child_theta_v_idx = round_and_get_v_index(child_node)
                child_cost_node = (child_x_v_idx, child_y_v_idx, child_theta_v_idx)
                
                # Check if node is in obstacle space or buffer zone
                try:
                    if((pxarray[int(child_node_fixed[0]), int(child_node_fixed[1])] == screen.map_rgb(pallet["white"])) or \
                       (pxarray[int(child_node_fixed[0]), int(child_node_fixed[1])] == screen.map_rgb(pallet["green"]))): continue
                except IndexError:
                    continue  # Attempted move was outside bounds of the map
                
                # Check if node is in visited list
                # If not, check if in open list using cost matrix C2C
                if(V[child_cost_node] == 0):
                    
                    # If child is not in open list, create new child node
                    if(C2C[child_x_v_idx, child_y_v_idx]  == np.inf):
                       cost2go = euclidean_distance(child_node_fixed, goal_node)  # Calculate Cost to Go using heuristic equation Euclidean Distance
                       cost2come = C2C[x_v_idx, y_v_idx] + step  # Calculate Cost to Come using parent Cost to Come and step size
                       parent[child_node_fixed] = fixed_node     # Add child node to parent dictionary 
                       
                       C2C[child_x_v_idx, child_y_v_idx] = cost2come   # Update cost matrix with newly calculate Cost to Come
                       costsum[child_cost_node] = cost2come + cost2go  # Calculate the total cost sum and add to reference dictionary (this will be used when determiniing optimal path)
                       child = [costsum[child_cost_node], child_node_fixed]  # Create new child node --> [total cost, (x, y, theta)]... Total cost is used as priority determinant in heapq
                       heapq.heappush(OL, child)   # push child node to heapq
                       
                # Child was in visited list, see if new path is most optimal
                else:
                    cost2go = euclidean_distance(child_node_fixed, goal_node)
                    cost2come = C2C[x_v_idx, y_v_idx] + step
                    
                    # Compare previously saved total cost estimate to newly calculated
                    # If new cost is lower than old cost, update in cost matrix and reassign parent 
                    if(costsum[child_cost_node] > (cost2come + cost2go)):  
                        parent[child_node_fixed] = fixed_node
                        C2C[child_x_v_idx, child_y_v_idx] = cost2come
                        costsum[child_cost_node] = cost2come + cost2go
    return False, solution_path

#%%
# Initialize pygame
pygame.init()

# Define screen size
window_size = (rows, cols)
screen = pygame.display.set_mode(window_size)
pxarray = pygame.PixelArray(screen)

#################################################
# Prompt User for starting and goal states
def GetUserInput():
    
    unanswered = True
    
    print("Welcome to the Point Robot Maze!")
    print("Before we get this show on the road, we need some information from you!\n")
    print("Robbie the Robot needs to navigate through a maze to get to get home...")
    print("Little Robbie is a bit forgetful though and needs your help remembering...")
    print("Where is he currently, and where is he going!\n")
    print("Robbie lives in a maze that is 180mm long and 50mm wide...")
    print("And he understands location using X and Y coordinates. So can you help?")
    print("---------------------------------------------------------------------------\n")
    while unanswered:
        start_x = float(input("Enter the starting x-coordinate: "))
        start_y = 249-float(input("Enter the starting y-coordinate: "))
        start_theta = float(input("Enter the starting orientation (0-360 degrees): "))
        
        # Check to see if start point falls in obstacle space, reprompt for new
        # coordinates if it does
        if(InObjectSpace(start_x, start_y)):
            print("Sorry, that start point appears to be object space!")
            continue
        
        goal_x = float(input("Enter the goal x-coordinate: "))
        goal_y = 249-float(input("Enter the goal y-coordinate: "))
        
        # Check to see if start point falls in obstacle space, reprompt for new
        # coordinates if it does
        if(InObjectSpace(goal_x, goal_y)):
            print("Sorry, that goal point appears to be in object space!")
            continue
        
        step_size = int(input("Enter step size from 1 to 10: "))
        if(step_size < 1 or step_size > 10):
            print("Sorry, that step size is not valid!")
            continue
        
        radius = int(input("Please enter a radius for Robbie the Robot greater than 0: "))
        if(radius < 0):
            print("Sorry, that radius is not valid!")
            continue
        
        start_node = [0.0, (start_x, start_y, start_theta)]
        goal = (goal_x,goal_y)
     
        unanswered = False

    return start_node, goal, step_size, radius

# Collect input from user:
# start_node: starting coordinates and orientation with total cost (float)
# goal_node:  desired end coordinates as tuple (float)
# step:       step size/movement length in mm (int)
# rradius:    size of robot radius in mm (int)
start_node, goal_node, step, rradius = GetUserInput()

# Draw board with objects and buffer zone
# rows:      size x-axis (named at one point, and forgot to change)
# cols:      size y-axis
# pallet:    color library with RGB values for drawing on pixel array
# C2C:       obsolete, starting costs set in FillCostMatrix()
# clearance: turn clearance for robot in mm 
# rradius:   robot radius in mm
DrawBoard(rows, cols, pxarray, pallet, C2C, clearance, rradius)

# Update the screen
pygame.display.update()

# Grab start time before running the solver
start_time = time.perf_counter()

# Start running solver
running = True
while running:
    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    FillCostMatrix(C2C, pxarray, pallet, thresh)
    
    # Start A_Star algorithm solver, returns game state of either SUCCESS (True) or FAILURE (false)
    alg_state, solution = A_Star(start_node, goal_node, OL, parent, V, C2C, costsum, step)
    
    if alg_state == False:
        print("Unable to find solution")
        end_time = time.perf_counter()
        running = False
    else:
        print("Solution found! Mapping now")
        end_time = time.perf_counter()
        running = False
    
    # Update the screen
    pygame.display.update()

# Calculate run time for solver
runtime = end_time - start_time

print("Time required to solve maze: ", runtime, " seconds")

# Draw start and goal points; start point will be filled, goal point will be hollow
pygame.draw.circle(screen, pygame.Color(pallet["red"]), (int(start_node[1][0]), start_node[1][1]), radius=5.0, width=0) # Start node    
pygame.draw.circle(screen, pygame.Color(pallet["red"]), (int(goal_node[0]), goal_node[1]), radius=5.0, width=1) # Goal node

# Draw solution path
for item in solution:
    pygame.draw.circle(screen, pygame.Color(pallet["red"]), (int(round(item[0])), int(round(item[1]))), radius=1.0, width=0)
    pygame.display.update()
    
# Freeze screen on completed maze screen until user quits the game
# (press close X on pygame screen)
running = True
while running:
    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # quit pygame
            pygame.quit()


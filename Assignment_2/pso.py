import random

# DO NOT import any other modules.
# DO NOT change the prototypes of any of the functions.
# Sample test cases given
# Grading will be based on hidden tests


# Cost function to be optimised
# Takes a list of elements
# Return the total sum of squares of even-indexed elements and inverse squares of odd-indexed elements
def cost_function(X): # 0.25 Marks
    # Your code goes here
    add = 0
    for i in range(len(X)):
        if i%2:
            add += X[i]**(-2)
        else:
            add += X[i]**(2)
    return add
    


# Takes length of vector as input
# Returns 4 values - initial_position, initial_velocity, best_position and best_cost
# Initialises position to a list with random values between [-10, 10] and velocity to a list with random values between [-1, 1]
# best_position is an empty list and best cost is set to -1 initially
def initialise(length): # 0.25 Marks
    # your code goes here
    initial_position = []
    initial_velocity = []
    for i in range(length):
        initial_position.append(random.uniform(-10,10))
        initial_velocity.append(random.uniform(-1,1))
    return [initial_position, initial_velocity, [], -1]


# Evaluates the position vector based on the input func
# On getting a better cost, best_position is updated in-place
# Returns the better cost 
def assess(position, best_position, best_cost, func): # 0.25 Marks
    # Your code goes here
    cost = func(position)
    if cost < best_cost or best_cost == -1:
        best_position[:] = position
        best_cost = cost
    return best_cost


# Updates velocity in-place by the given formula for each element:
# vel = w*vel + c1*r1*(best_position-position) + c2*r2*(best_group_position-position)
# where r1 and r2 are random numbers between 0 and 1 (not same for each element of the list)
# No return value
def velocity_update(w, c1, c2, velocity, position, best_position, best_group_position): # 0.5 Marks
    # Code goes here
    for i in range(len(position)):
        r1 = random.random()
        r2 = random.random()
        velocity[i] = w*velocity[i] + c1*r1*(best_position[i]-position[i]) + c2*r2*(best_group_position[i]-position[i])
    


# Input - position, velocity, limits(list of two elements - [min, max])
# Updates position in-place by the given formula for each element:
# pos = pos + vel
# Position element set to limit if it crosses either limit value
# No return value
def position_update(position, velocity, limits): # 0.5 Marks
    # Code goes here
    for i in range(len(position)):
        position[i] = position[i] + velocity[i]
        if position[i]>limits[1]:
            position[i] = limits[1]
        elif position[i] < limits[0]:
            position[i] = limits[0]

    


# swarm is a list of particles each of which is a list containing current_position, current_velocity, best_position and best_cost
# Initialise these using the function written above
# In every iteration for every swarm particle, evaluate the current position using the assess function (use the cost function you have defined) and update the particle's best cost if needed
# Update the best group cost and best group position based on performance of that particle
# Then for every swarm particle, first update its velocity then its position
# Return the best position and cost for the group
def optimise(vector_length, swarm_size, w, c1, c2, limits, max_iterations, initial_best_group_position=[], initial_best_group_cost=-1): # 1.25 Marks
    # Your Code goes here
    swarm = []
    for i in range(swarm_size):
        swarm.append(initialise(vector_length))
    swarm = [list(i) for i in swarm]
    for i in range(max_iterations):
        #Iterating over swarm to find best group position and cost
        for j in range(swarm_size):
            swarm[j][3] = assess(swarm[j][0], swarm[j][2], swarm[j][3], lambda x: cost_function(x))
            if swarm[j][3] < initial_best_group_cost or initial_best_group_cost == -1:
                initial_best_group_position = swarm[j][2]
                initial_best_group_cost = swarm[j][3]
        #updating it
        for j in range(swarm_size):
            velocity_update(w, c1, c2, swarm[j][1], swarm[j][0], swarm[j][2], initial_best_group_position)
            position_update(swarm[j][0], swarm[j][1], limits)
    return [initial_best_group_position, initial_best_group_cost]
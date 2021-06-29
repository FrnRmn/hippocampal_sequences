import numpy as np


class corridor():

    def __init__(self, matrix, code, zones=None):
        self.matrix = matrix
        self.shapes = matrix.shape
        self.vector = matrix.reshape((-1,))
        self.code = code
        self.zones = zones
        self.agent = None

    #set the new given agent position in the environment
    def set_agent_position(self, new_position):
        self.agent = new_position

    #compute the proximal states that the agent can legally reach within one step: actions available are up, down, left, right
    def agent_available_proximal_states(self):
        agent_pos = (int(self.agent / self.shapes[1]), self.agent % self.shapes[1])
        up = np.array([agent_pos[0]-1, agent_pos[1]])
        down =  np.array([agent_pos[0]+1, agent_pos[1]])
        left =  np.array([agent_pos[0], agent_pos[1]-1])
        right =  np.array([agent_pos[0], agent_pos[1]+1])
        proximals = []
        for next_state in [up, down, left, right]:
            if next_state[0] < 0 or next_state[0] >= self.shapes[0] or next_state[1] < 0 or next_state[1] >= self.shapes[1]:    #next move outside the maze borders
                next_state_val = -10
            elif self.matrix[next_state[0],next_state[1]] == 0:     #next move inside a wall
                next_state_val = -1
            else:   #legal next move
                next_state_arr = [next_state[0]*self.shapes[1]+next_state[1]]
                proximals.append(next_state_arr)
        return np.array(proximals).reshape(-1)

    #check if the agent position matches with the reward position
    def check_reward(self, reward_pos):
        if self.agent == reward_pos:
            return 1
        else:
            return -0.1



class square_maze():

    def __init__(self, matrix, code, zones=None):
        self.matrix = matrix
        self.shapes = matrix.shape
        self.vector = matrix.reshape((-1,))
        self.code = code
        self.zones = zones
        self.agent = None

    #set the new given agent position in the environment
    def set_agent_position(self, new_position):
        self.agent = new_position

    #compute the proximal states that the agent can legally reach within one step: actions available are up, down, left, right
    def agent_available_proximal_states(self):
        agent_pos = (int(self.agent / self.shapes[1]), self.agent % self.shapes[1])
        up = np.array([agent_pos[0]-1, agent_pos[1]])
        down =  np.array([agent_pos[0]+1, agent_pos[1]])
        left =  np.array([agent_pos[0], agent_pos[1]-1])
        right =  np.array([agent_pos[0], agent_pos[1]+1])
        proximals = []
        for next_state in [up, down, left, right]:
            if next_state[0] < 0 or next_state[0] >= self.shapes[0] or next_state[1] < 0 or next_state[1] >= self.shapes[1]:  #next move outside the maze borders
                next_state_val = -10
            elif self.matrix[next_state[0],next_state[1]] == 0:     #next move inside a wall
                next_state_val = -1
            else:   #legal next move
                next_state_arr = [next_state[0]*self.shapes[1]+next_state[1]]
                proximals.append(next_state_arr)
        return np.array(proximals).reshape(-1)

    #check if the agent position matches with the reward position
    def check_reward(self, reward_pos):
        if self.agent == reward_pos:
            return 1
        else:
            return -0.1


        
class Arm8_maze():

    def __init__(self, matrix, code, zones=None):
        self.matrix = matrix
        self.shapes = matrix.shape
        self.vector = matrix.reshape((-1,))
        self.code = code
        self.zones = zones
        self.agent = None
        self.multidims_start, self.multidims_ends = self.extract_multidimensional_locations()

    # the reward is placed in a random arm
    def random_task(self):
        possible_rewards = np.where(self.vector == self.code['possible_rewards'])[0]
        return np.random.choice(possible_rewards, 1)

    # the reward is placed in a random arm and stays there for a certain number of episodes
    def random_stay_task(self, params, counter, previous = None):
        if counter == 0:
            previous = self.random_task()
            counter = np.random.randint(params["reward_change_min"],params["reward_change_max"],1)
        else:
            counter -= 1
        
        return previous, counter

    #set the new given agent position in the environment
    def set_agent_position(self, new_position):
        self.agent = new_position


    #look from the map which are the states that needs to be considered adjacent
    def extract_multidimensional_locations(self):
        start = np.where(self.vector == 4)[0]
        ends = np.where(self.vector == 3)[0]
        return start, ends

    #compute the proximal states that the agent can legally reach within one step: actions available are up, down, left, right
    def agent_available_proximal_states(self):
        agent_pos = (int(self.agent / self.shapes[1]), self.agent % self.shapes[1])
        up = np.array([agent_pos[0]-1, agent_pos[1]])
        down =  np.array([agent_pos[0]+1, agent_pos[1]])
        left =  np.array([agent_pos[0], agent_pos[1]-1])
        right =  np.array([agent_pos[0], agent_pos[1]+1])
        proximals = []
        for next_state in [up, down, left, right]:
            if next_state[0] < 0 or next_state[0] >= self.shapes[0] or next_state[1] < 0 or next_state[1] >= self.shapes[1]:    #next move outside the maze borders
                next_state_val = -10
            elif self.matrix[next_state[0],next_state[1]] == 0:      #next move inside a wall
                next_state_val = -1     
            else:   #legal next move
                next_state_arr = [next_state[0]*self.shapes[1]+next_state[1]]
                proximals.append(next_state_arr)

        if self.agent in self.multidims_start:          # connects states that needs to be considered adjacent
            for i in range(len(self.multidims_ends)):
                proximals.append([self.multidims_ends[i]])
        elif self.agent in self.multidims_ends:
            for i in range(len(self.multidims_start)):
                proximals.append([self.multidims_start[i]])

        return np.array(proximals).reshape(-1)

    #check if the agent position matches with the reward position
    def check_reward(self, reward_pos):
        if self.agent == reward_pos:
            return 1
        else:
            return -0.1

    #convert coordinates from a 2D array to a flattened 1D array index
    def conversion_2D_to_1D(self, coord_2D):
        return coord_2D[0] * self.shapes[1] + coord_2D[1]

    #convert 1D index to coordinates of a 2D array
    def conversion_1D_to_2D(self, indx_1D):
        return (int(indx_1D / self.shapes[1]), indx_1D % self.shapes[1])     




#T-maze (not used in the simulations)
class T_maze():

    def __init__(self, matrix, code, zones=None):
        self.matrix = matrix
        self.shapes = matrix.shape
        self.vector = matrix.reshape((-1,))
        self.code = code
        self.zones = zones
        self.agent = None

    def random_task(self):
        possible_rewards = np.where(self.vector == self.code['possible_rewards'])[0]
        return np.random.choice(possible_rewards, 1)

    def random_stay_task(self, counter, previous = None):
        if counter == 0:
            previous = self.random_task()
            counter = np.random.randint(5,10,1)
        else:
            counter -= 1
        
        return previous, counter

    def set_agent_position(self, new_position):
        self.agent = new_position

    def agent_available_proximal_states(self):
        agent_pos = (int(self.agent / self.shapes[1]), self.agent % self.shapes[1])
        up = np.array([agent_pos[0]-1, agent_pos[1]])
        down =  np.array([agent_pos[0]+1, agent_pos[1]])
        left =  np.array([agent_pos[0], agent_pos[1]-1])
        right =  np.array([agent_pos[0], agent_pos[1]+1])
        proximals = []
        for next_state in [up, down, left, right]:
            if next_state[0] < 0 or next_state[0] >= self.shapes[0] or next_state[1] < 0 or next_state[1] >= self.shapes[1]:
                next_state_val = -10
            elif self.matrix[next_state[0],next_state[1]] == 0:
                next_state_val = -1
            else:
                next_state_arr = [next_state[0]*self.shapes[1]+next_state[1]]
                proximals.append(next_state_arr)
        return np.array(proximals).reshape(-1)

    def check_reward(self, reward_pos):
        if self.agent == reward_pos:
            return 1
        else:
            return -0.1
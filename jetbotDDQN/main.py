import pygame
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

SCREEN_DIMENSION = 800
NUMBER_OF_LINES = 10
INTERVALS = 360/NUMBER_OF_LINES
STARTING_DEGREE = INTERVALS/2

WIDTH_OF_CAR = 25
LENGTH_OF_CAR = 50
RADAR_LEN = 500

FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
EPISODES = 500
LEARNING_RATE = 0.0001
MEM_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.99
MAX_STEPS = 3600

EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.99999
EXPLORATION_MIN = 0.01

pygame.init()
window = pygame.display.set_mode((SCREEN_DIMENSION, SCREEN_DIMENSION))

class Car():
    def __init__(self):
        self.speed = 5
        self.reward = 0
        self.crash = False
        self.reward_gain = False

        self.original_image = pygame.Surface((LENGTH_OF_CAR, WIDTH_OF_CAR))
        self.original_image.set_colorkey(WHITE)
        self.original_image.fill(BLACK)
        self.car_body = self.original_image

        self.position = pygame.math.Vector2((200, 100))
        self.direction = pygame.math.Vector2(self.speed, 0)
        self.numbers = 0, 0
        self.calculated_distances = []

        file = open("wallcoordinates.txt", "r")
        self.wall_coordinates = []
        for line in file:
            stripped_line = line.strip()
            self.wall_coordinates.append(stripped_line)

        file = open("pointcoordinates.txt", "r")
        self.point_coordinates = []
        for line in file:
            stripped_line = line.strip()
            self.point_coordinates.append(stripped_line)

        for i in range(len(self.wall_coordinates)):
            op = self.wall_coordinates[i].strip('][').split(', ')
            for j in range(len(op)):
                op[j] = int(op[j])
            self.wall_coordinates[i] = op
        
        for i in range(len(self.point_coordinates)):
            op = self.point_coordinates[i].strip('][').split(', ')
            for j in range(len(op)):
                op[j] = int(op[j])
            self.point_coordinates[i] = op
    
    def agent_movement(self, value):
        if value == 0:
            self.position += self.direction
        #if value == 0:
            #self.position -= self.direction
        if value == 1:
            self.direction.rotate_ip(-1)
        if value == 2:
            self.direction.rotate_ip(1)
    
    def rendering(self):
        window.fill(WHITE)
        self.angle = self.direction.angle_to((1, 0))
        rotated_car = pygame.transform.rotate(self.car_body, self.angle)
        window.blit(rotated_car, rotated_car.get_rect(center = (round(self.position.x), round(self.position.y))))
        self.numbers = rotated_car.get_rect(center = (round(self.position.x), round(self.position.y)))

        for i in range(len(self.wall_coordinates) - 1):
            if i % 2 == 0:
                pygame.draw.line(window, BLACK, self.wall_coordinates[i], self.wall_coordinates[i + 1], 1)

        for i in range(len(self.point_coordinates) - 1):
            if i % 2 == 0:
                pygame.draw.line(window, RED, self.point_coordinates[i], self.point_coordinates[i + 1], 1)
        
        radar_middle = (self.numbers[0] + 12.5, self.numbers[1] + 25)
        self.radar_coordinates = []
        for i in range(NUMBER_OF_LINES):
            x = radar_middle[0] + math.cos(math.radians(STARTING_DEGREE + i * INTERVALS)) * RADAR_LEN
            y = radar_middle[1] + math.sin(math.radians(STARTING_DEGREE + i * INTERVALS)) * RADAR_LEN
            
            temporary = []
            temporary.append(radar_middle[0])
            temporary.append(radar_middle[1])

            self.radar_coordinates.append(deepcopy(temporary))

            temporary = []
            temporary.append(x)
            temporary.append(y)
            self.radar_coordinates.append(deepcopy(temporary))

            pygame.draw.line(window, BLACK, radar_middle, (x,y), 1)
    
    def on_segment(self, p, q, r):
        if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
            return True
        
        return False 

    def orientation(self, p, q, r):
        val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))

        if val == 0 : return 0
        return 1 if val > 0 else -1

    def intersects(self, seg1, seg2):
        p1, q1 = seg1
        p2, q2 = seg2

        o1 = self.orientation(p1, q1, p2)

        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self.on_segment(p1, q1, p2) : return True
        if o2 == 0 and self.on_segment(p1, q1, q2) : return True
        if o3 == 0 and self.on_segment(p2, q2, p1) : return True
        if o4 == 0 and self.on_segment(p2, q2, q1) : return True

        return False

    def calculate_formula(self, seg):
        p1, q1 = seg
        x1, y1 = p1
        x2, y2 = q1

        if (x2 - x1) == 0:
            return -1000, -1000

        m = (y2 - y1)/(x2 - x1)
        b = y1 - m*x1

        return m, b

    def calculate_intersection(self, seg1, seg2):
        p1, q1 = seg1
        x1, y1 = p1

        m1, b1 = self.calculate_formula(seg1)
        m2, b2 = self.calculate_formula(seg2)
        
        if m1 - m2 == 0 or (m1 == -1000 and b1 == -1000) or (m2 == -1000 and b2 == -1000):
            return 1000
            
        x = (b2 - b1)/(m1 - m2)
        y = m1 * x + b1

        distance = math.sqrt((y - y1)**2 + (x - x1)**2)

        return distance
    
    def radar_distances(self):
        self.calculated_distances = []
        self.point_distances = []

        for i in range(len(self.radar_coordinates)):
            values_walls = []
            values_points = []
            if i % 2 == 0:
                radar_segment = []
                radar_segment.append(self.radar_coordinates[i]) # constant origin
                radar_segment.append(self.radar_coordinates[i + 1])

                for j in range(len(self.wall_coordinates) - 1):
                    if j % 2 == 0:
                        wall_segment = []
                        wall_segment.append(self.wall_coordinates[j])
                        wall_segment.append(self.wall_coordinates[j + 1])

                        result = self.intersects(radar_segment, wall_segment)

                        if result == True:
                            distance = self.calculate_intersection(radar_segment, wall_segment)
                        else:
                            distance = 1000

                        values_walls.append(distance)

                for k in range(len(self.point_coordinates) - 1):
                    if k % 2 == 0:
                        wall_segment = []
                        wall_segment.append(self.point_coordinates[k])
                        wall_segment.append(self.point_coordinates[k + 1])

                        result = self.intersects(radar_segment, wall_segment)

                        if result == True:
                            distance = self.calculate_intersection(radar_segment, wall_segment)
                        else:
                            distance = 1000

                        values_points.append(distance)
            
            if len(values_walls) > 0:
                self.calculated_distances.append(min(values_walls))
            if len(values_points) > 0:
                self.point_distances.append(min(values_points))
    
    def detecting_crash(self):
        self.crash = False
        self.reward_gain = False

        for i in range(len(self.calculated_distances)):
            if self.calculated_distances[i] < 20:
                #print("detected crash")
                self.crash = True

        for i in range(len(self.point_distances)):
            if self.point_distances[i] <= 20:
                #print("detected reward")
                self.reward_gain = True
                temporary_point_coords = deepcopy(self.point_coordinates)

                for i in range(len(self.radar_coordinates)):
                    if i % 2 == 0:
                        radar_segment = []
                        radar_segment.append(self.radar_coordinates[i]) # constant origin
                        radar_segment.append(self.radar_coordinates[i + 1])

                        for k in range(len(self.point_coordinates) - 1):
                            if k % 2 == 0:
                                wall_segment = []
                                wall_segment.append(self.point_coordinates[k])
                                wall_segment.append(self.point_coordinates[k + 1])

                                result = self.intersects(radar_segment, wall_segment)

                                if result == True:
                                    distance = self.calculate_intersection(radar_segment, wall_segment)
                                else:
                                    distance = 1000
                                
                                if distance <= 20:
                                        if self.point_coordinates[k] in temporary_point_coords and self.point_coordinates[k+1] in temporary_point_coords:
                                            temporary_point_coords.remove(self.point_coordinates[k])
                                            temporary_point_coords.remove(self.point_coordinates[k + 1])
                
                self.point_coordinates = deepcopy(temporary_point_coords)              



class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0

        self.states = np.zeros((MEM_SIZE, NUMBER_OF_LINES), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, NUMBER_OF_LINES),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones
    

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = NUMBER_OF_LINES # 10 radars
        self.action_space = 3 # 4 possible movements

        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)

        self.advantage = nn.Linear(FC2_DIMS, self.action_space)
        self.value = nn.Linear(FC2_DIMS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        advantage = self.advantage(x)
        value = self.value(x)

        return (value + advantage) - advantage.mean()

class Agent():
    def __init__(self):
        self.action_network = Network()
        self.target_network = Network()
        self.memory = ReplayBuffer()

        self.exploration_rate = EXPLORATION_MAX
        self.learn_step_counter = 0
        self.net_copy_interval = 10
    
    def choosing_action(self, observation):
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)

        network_inputs = torch.Tensor(observation)
        network_outputs = self.action_network.forward(network_inputs)

        index = np.argmax(network_outputs.detach())
        return index # 0 to 3
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
                
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.action_network(states)[batch_indices, actions]
        next_q_values = self.target_network(states_)
        actions_ = self.action_network(states_).max(dim=1)[1]
        actions_ = next_q_values[batch_indices, actions_]

        q_target = rewards + GAMMA * actions_ * dones
        td = q_target - q_values

        self.action_network.optimizer.zero_grad()
        loss = ((td ** 2.0)).mean()
        loss.backward()
        self.action_network.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_network.load_state_dict(self.action_network.state_dict())

        self.learn_step_counter += 1
    

car = Car()
agent = Agent()
episode = 1
standard = len(car.point_coordinates)/2
total_reward = 0
all_total_reward = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    print("Episode {} Randomizing {}".format(episode, agent.exploration_rate))
    episode += 1
    reward = 0
    done = False
    move_chosen = 0
    step = 0
    counter = 0

    if episode % 10 == 0:
        print("average barriers = ")
        all_total_reward.append(deepcopy(total_reward/episode))
        for k in range(len(all_total_reward)):
            print(all_total_reward[k])
        total_reward = 0
        episode = 1


    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        step += 1
        car.rendering()
        car.radar_distances()

        values = car.calculated_distances
        state = deepcopy(values)
        action = agent.choosing_action(values)

        car.agent_movement(action)

        car.radar_distances()
        values = car.calculated_distances
        state_ = deepcopy(values)

        car.rendering()
        car.radar_distances()
        car.detecting_crash()

        if car.reward_gain == True:
            reward = 5 * (standard - len(car.point_coordinates)/2)
        if car.crash == True:
            print("crash")
            reward = -100
            done = True
    
        agent.memory.add(state, action, reward, state_, done)
        agent.learn()

        pygame.display.flip()

        if car.point_coordinates == [] or done == True:
            temp = len(car.point_coordinates)/2
            car = Car()

            total_reward += len(car.point_coordinates)/2 - temp
            print("barriers crossed {}".format(len(car.point_coordinates)/2 - temp))
            print("--")
            done = True

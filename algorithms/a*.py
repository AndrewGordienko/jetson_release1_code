import numpy as np
import pygame
import math
from itertools import product
from operator import attrgetter

DIMENSION = 50
WIDTH = 500
HEIGHT = 500

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

COST = 1
START_NUMBER = 1
END_NUMBER = 2
WALL = 3
STARTING_COORDINATE_X, STARTING_COORDINATE_Y = 0, 0
ENDING_COORDINATE_X, ENDING_COORDINATE_Y = 0, 0

screen = pygame.display.set_mode((WIDTH, HEIGHT))
board = np.zeros((DIMENSION, DIMENSION))

placeholder = 0
enter_toggle = 0

class Node():
    def __init__(self, parent_node):
        self.g_cost = None
        self.f_cost = None
        self.h_cost = None

        self.parent_node = parent_node
        self.x_coordinate, self.y_coordinate = None, None

class Agent():
    def finding_position(self, arr, number):
        for i in range(len(arr[0])):
            for j in range(len(arr[0])):
                if arr[i][j] == number:
                    return j, i   
        return None, None
        
    def initialization(self):
        self.start_node = Node(None)
        self.start_node.g_cost = self.start_node.f_cost = self.start_node.h_cost = 0
        self.start_node.x_coordinate, self.start_node.y_coordinate = STARTING_COORDINATE_X, STARTING_COORDINATE_Y
        
        self.nodes_not_visited = []
        self.nodes_visited = []
        self.nodes_not_visited.append(self.start_node)
    
    def backtrack(self, current):
        self.path = []
        while current.parent_node != None:
            self.path.append((current.x_coordinate, current.y_coordinate))
            current = current.parent_node
        
        self.path.pop(0)
        

    def main(self):
        self.initialization()

        while True:
            current = min(self.nodes_not_visited, key=attrgetter('f_cost'))
            self.nodes_not_visited.remove(current)
            self.nodes_visited.append(current)

            self.nodes_not_visited = list(set(self.nodes_not_visited))
            self.nodes_visited = list(set(self.nodes_visited))

            if (current.x_coordinate, current.y_coordinate) == (ENDING_COORDINATE_X, ENDING_COORDINATE_Y):
                self.backtrack(current)
                break

            neighbors = []
            for vec in product([-1, 0, 1], repeat=2):
                if not any(vec):
                    continue  
                search_x, search_y = current.x_coordinate + vec[1], current.y_coordinate + vec[0]
                if search_x >= WIDTH/DIMENSION or search_x < 0 or search_y >= HEIGHT/DIMENSION or search_y < 0: 
                    continue
                if board[search_y][search_x] != 1:
                    new_node = Node(current)
                    new_node.x_coordinate, new_node.y_coordinate = search_x, search_y
                    neighbors.append((new_node, vec))
            
            for neighbor, vec in neighbors:
                if neighbor not in self.nodes_visited:
                    for open_node in range(len(self.nodes_not_visited)):
                        if neighbor == self.nodes_not_visited[open_node] and neighbor.g_cost >= self.nodes_not_visited[open_node].g_cost:
                            break
                    else:
                        self.nodes_not_visited.append(neighbor)
                        
                        if vec == (-1, -1) or vec == (-1, 1) or vec == (1, -1) or vec == (1, 1):
                            COST = 14
                        else:
                            COST = 10
                        
                        neighbor.g_cost = current.g_cost + COST
                        neighbor.h_cost = abs(neighbor.y_coordinate - ENDING_COORDINATE_Y) + abs(neighbor.x_coordinate - ENDING_COORDINATE_X)
                        neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                        temp_list = []
                        for a_node in self.nodes_not_visited:
                            value = 0
                            for i in range(len(temp_list)):
                                if temp_list[i].x_coordinate == a_node.x_coordinate and temp_list[i].y_coordinate == a_node.y_coordinate:
                                    value = 1
                            if value == 0:
                                temp_list.append(a_node)
                        
                        self.nodes_not_visited = temp_list
                

agent = Agent()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()

        if enter_toggle != 1:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT:
                    placeholder = 1
                
                if event.key == pygame.K_LCTRL:
                    placeholder = 2

                if event.key == pygame.K_SPACE:
                    enter_toggle = 1
                    
                    for row in range(DIMENSION):
                        for column in range(DIMENSION):
                            if board[row][column] == 2:
                                # this is all the set up code
                            
                                x_start, y_start = column, row
                            
                            if board[row][column] == 3:
                                # this is the end spot
                                x_end, y_end = column, row
                    
                    # set up code should be here
                    STARTING_COORDINATE_X, STARTING_COORDINATE_Y = agent.finding_position(board, 2)
                    ENDING_COORDINATE_X, ENDING_COORDINATE_Y = agent.finding_position(board, 3)

                    board[STARTING_COORDINATE_Y][STARTING_COORDINATE_X] = 0
                    board[ENDING_COORDINATE_Y][ENDING_COORDINATE_X] = 0

                    agent.main()
                    for i in range(len(agent.path)):
                        board[agent.path[i][1]][agent.path[i][0]] = 4

                    board[STARTING_COORDINATE_Y][STARTING_COORDINATE_X] = 2
                    board[ENDING_COORDINATE_Y][ENDING_COORDINATE_X] = 3
                
            if event.type == pygame.KEYUP:
                placeholder = 0

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                x, y = math.floor(x/DIMENSION), math.floor(y/DIMENSION)

                if placeholder == 1:
                    # start
                    if board[x][y] == 0:
                        board[x][y] = 2
                    else:
                        board[x][y] = 0
                
                if placeholder == 2:
                    # end
                    if board[x][y] != 3:
                        board[x][y] = 3
                    else:
                        board[x][y] = 0
                
                if placeholder == 0:
                    # wall
                    if board[x][y] == 0:
                        board[x][y] = 1
                    else:
                        board[x][y] = 0

    screen.fill(WHITE)

    for row in range(DIMENSION):
        for column in range(DIMENSION):
            if board[row][column] == 0:
                # empty spot
                pygame.draw.rect(screen, BLACK, (DIMENSION * row, DIMENSION * column, 50, 50), 1)

            if board[row][column] == 1:
                # wall
                pygame.draw.rect(screen, BLACK, (DIMENSION * row, DIMENSION * column, 50, 50))

            if board[row][column] == 2:
                # start
                pygame.draw.rect(screen, GREEN, (DIMENSION * row, DIMENSION * column, 50, 50))
                pygame.draw.rect(screen, BLACK, (DIMENSION * row, DIMENSION * column, 50, 50), 1)

            if board[row][column] == 3:
                # end
                pygame.draw.rect(screen, RED, (DIMENSION * row, DIMENSION * column, 50, 50))
                pygame.draw.rect(screen, BLACK, (DIMENSION * row, DIMENSION * column, 50, 50), 1)
            
            if board[row][column] == 4:
                # path solved
                pygame.draw.rect(screen, BLUE, (DIMENSION * row, DIMENSION * column, 50, 50))
                pygame.draw.rect(screen, BLACK, (DIMENSION * row, DIMENSION * column, 50, 50), 1)

    pygame.display.flip()

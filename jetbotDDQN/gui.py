import pygame
import math
from copy import deepcopy

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

SCREEN_DIMENSION = 800

NUMBER_OF_LINES = 5
INTERVALS = 360/NUMBER_OF_LINES
STARTING_DEGREE = INTERVALS/2

WIDTH_OF_CAR = 25
LENGTH_OF_CAR = 50
RADAR_LEN = 200

pygame.init()
window = pygame.display.set_mode((SCREEN_DIMENSION, SCREEN_DIMENSION))

class Car():
    def __init__(self):
        self.speed = 0.5
        self.wall_coordinates = []
        self.point_coordinates = []

        self.original_image = pygame.Surface((LENGTH_OF_CAR, WIDTH_OF_CAR))
        self.original_image.set_colorkey(WHITE)
        self.original_image.fill(BLACK)
        self.car_body = self.original_image

        self.position = pygame.math.Vector2((200, 100))
        self.direction = pygame.math.Vector2(self.speed, 0)
        self.numbers = 0, 0
    
    def movement(self, keys):
        if keys[pygame.K_w]:
            self.position += self.direction
        if keys[pygame.K_s]:
            self.position -= self.direction
        if keys[pygame.K_a]:
            self.direction.rotate_ip(-1)
        if keys[pygame.K_d]:
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

    def wall_placing(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                x, y = pygame.mouse.get_pos()
                temporary = []
                temporary.append(x)
                temporary.append(y)

                if temporary not in self.wall_coordinates:
                    self.wall_coordinates.append(deepcopy(temporary))
            
            if event.key == pygame.K_LALT:
                x, y = pygame.mouse.get_pos()
                temporary = []
                temporary.append(x)
                temporary.append(y)

                if temporary not in self.point_coordinates:
                    self.point_coordinates.append(deepcopy(temporary))

    def radar_distances(self):
        self.calculated_distances = []
        self.point_distances = []

        for i in range(len(self.radar_coordinates)):
            values = []
            if i % 2 == 0:
                for j in range(len(self.wall_coordinates) - 1):
                    if j % 2 == 0:
                        temporary_wall_coords = []
                        temporary_wall_coords.append(self.wall_coordinates[j])
                        temporary_wall_coords.append(self.wall_coordinates[j + 1])

                        temporary_radar_coords = []
                        temporary_radar_coords.append(self.radar_coordinates[i])
                        temporary_radar_coords.append(self.radar_coordinates[i + 1])

                        m_wall = (temporary_wall_coords[1][1] - temporary_wall_coords[0][1])/(temporary_wall_coords[1][0] - temporary_wall_coords[0][0])
                        b_wall = temporary_wall_coords[0][1] - m_wall * temporary_wall_coords[0][0]
                        m_wall, b_wall = round(m_wall, 2), round(b_wall, 2)

                        m_radar = (temporary_radar_coords[1][1] - temporary_radar_coords[0][1])/(temporary_radar_coords[1][0] - temporary_radar_coords[0][0])
                        b_radar = temporary_radar_coords[0][1] - m_radar * temporary_radar_coords[0][0]
                        m_radar, b_radar = round(m_radar, 2), round(b_radar, 2)

                        if m_wall != m_radar:
                            x = (b_radar - b_wall)/(m_wall - m_radar)
                            y = m_wall*x + b_wall

                            distance = math.sqrt((x - self.radar_coordinates[i][0])**2 + (y - self.radar_coordinates[i][1])**2)

                            if self.wall_coordinates[j + 1][0] >= x >= self.wall_coordinates[j][0]:
                                values.append(distance)
                            if self.wall_coordinates[j][0] >= x >= self.wall_coordinates[j + 1][0]:
                                values.append(distance)
            
            if len(values) > 0:
                self.calculated_distances.append(min(values))

        for i in range(len(self.radar_coordinates)):
            values = []
            if i % 2 == 0:
                for j in range(len(self.point_coordinates) - 1):
                    if j % 2 == 0:
                        temporary_wall_coords = []
                        temporary_wall_coords.append(self.point_coordinates[j])
                        temporary_wall_coords.append(self.point_coordinates[j + 1])

                        temporary_radar_coords = []
                        temporary_radar_coords.append(self.radar_coordinates[i])
                        temporary_radar_coords.append(self.radar_coordinates[i + 1])

                        m_wall = (temporary_wall_coords[1][1] - temporary_wall_coords[0][1])/(temporary_wall_coords[1][0] - temporary_wall_coords[0][0])
                        b_wall = temporary_wall_coords[0][1] - m_wall * temporary_wall_coords[0][0]
                        m_wall, b_wall = round(m_wall, 2), round(b_wall, 2)

                        m_radar = (temporary_radar_coords[1][1] - temporary_radar_coords[0][1])/(temporary_radar_coords[1][0] - temporary_radar_coords[0][0])
                        b_radar = temporary_radar_coords[0][1] - m_radar * temporary_radar_coords[0][0]
                        m_radar, b_radar = round(m_radar, 2), round(b_radar, 2)

                        if m_wall != m_radar:
                            x = (b_radar - b_wall)/(m_wall - m_radar)
                            y = m_wall*x + b_wall

                            distance = math.sqrt((x - self.radar_coordinates[i][0])**2 + (y - self.radar_coordinates[i][1])**2)

                            if self.point_coordinates[j + 1][0] >= x >= self.point_coordinates[j][0]:
                                values.append(distance)
                            if self.point_coordinates[j][0] >= x >= self.point_coordinates[j + 1][0]:
                                values.append(distance)
            if len(values) > 0:
                self.point_distances.append(min(values))

    
    def detecting_crash(self):
        for i in range(len(self.calculated_distances)):
            if self.calculated_distances[i] <= 20:
                print("--")
                print("i died")
        
        for i in range(len(self.point_distances)):
            if self.point_distances[i] <= 20:
                print("--")
                print("i got points")
    
    def saving_lists(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LCTRL:
                with open("pointcoordinates.txt", "w") as f:
                    for s in self.point_coordinates:
                        f.write(str(s) +"\n")
                
                with open("wallcoordinates.txt", "w") as f:
                    for s in self.wall_coordinates:
                        f.write(str(s) +"\n")
                
                print("saved")
        
car = Car()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        
        car.wall_placing(event)
        car.saving_lists(event)

    keys = pygame.key.get_pressed()
    car.movement(keys)
    car.rendering()
    car.radar_distances()
    car.detecting_crash()

    pygame.display.flip()

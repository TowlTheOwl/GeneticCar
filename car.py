import numpy as np
import math
import pygame
import itertools
import tools

"""
start_pos: starting position of the car
size: size of the car
color: color of the car
parent_wegiths: the neural network weights of the parent car
mr: mutation rate
keep_weight: whether the car survived or not
"""


class Car:
    def __init__(self, start_pos:tuple, size:tuple, mr:float, nn_size:tuple, start_vel:float, max_vel:float, 
                 min_vel:float, rot_vel:float, acc:float, bias_term:int,  scan_length:int, num_sensors:int=5, sensor_angle=math.pi) -> None:
        
        self.start_pos = start_pos
        self.pos = start_pos
        self.size = size
        self.center = (self.pos[0] + self.size[0]/2, self.pos[1], self.size[1]/2)
        self.corners = ()
        self.sensors_dist = np.zeros(num_sensors)
        self.sensors_pos = np.zeros((2, num_sensors))
        self.sensor_angles = ()
        for i in range(num_sensors):
            self.sensor_angles += ((i, -sensor_angle/2 + sensor_angle/(num_sensors-1)*i),)
        self.age = 0
        self.theta_diag = math.atan(size[1] / size[0])
        self.scan_length = scan_length

        # neural network
        self.command = np.zeros(4, dtype=bool)
        self.weights = None
        self.nn_size = nn_size
        self.mr = mr
        self.bias_term = bias_term
        self.actions = (self.accelerate, self.decelerate, self.turn_left, self.turn_right)

        # img
        self.image = pygame.image.load("images/car.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, self.size)
        self.sprite_image = self.image.copy()

        # values
        self.angle = 0
        self.start_vel = start_vel
        self.vel = start_vel
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.rotation_vel = rot_vel
        self.acceleration = acc
        self.one = np.array([[1]])

        # track status
        self.alive = True
        self.distance = 0
        self.time = 0
        self.finished = False

        self.init_weight()

    def return_age(self):
        return self.age
    
    def return_distance(self):
        if self.finished:
            return float('inf')
        else:
            return self.distance
    
    def return_time(self):
        return self.time

    def init_weight(self, parent_weight:np.ndarray=None, rand=False):
        """
        MODES TO BE DEVELOPED:
        0. Natural Selection: everyone is random, but the top 'n' cars survive
        1. Inherit From the Best: the new neural networks are mutations to the best car
        2. Own Family Line: 2 cars collaborate, where the best one survives and second mutates off the best
        """
        mode = 1

        # output = array with 4 elements, the max values gets executed

        # initialize/mutate neural network
        
        if rand or parent_weight is None:
            prev_weight = self.weights
            num_weights = 0
            for i in range(len(self.nn_size)-1):
                num_weights += (self.nn_size[i]+self.bias_term) * self.nn_size[i+1]
            self.weights = np.random.uniform(-1, 1, size=(num_weights,))

        else:
            parent_weight_copy = parent_weight.copy()
            if mode == 1:
                self.weights = parent_weight_copy+(np.random.uniform(-1, 1, size=parent_weight_copy.shape) * self.mr)
            
    def draw(self, win):
        rect = self.image.get_rect(center=self.sprite_image.get_rect(topleft=self.pos).center)
        win.blit(self.image, rect)

    def draw_radar(self, win):
        for i in range(self.sensors_pos.shape[1]):
            pygame.draw.line(win, (0, 200, 0), self.center, (self.sensors_pos[0][i], self.sensors_pos[1][i]), 1)
        for corner in self.corners:
            pygame.draw.circle(win, (200, 0, 0), corner, 3)
    
    def update_center_pos(self):
        self.center = (self.pos[0] + self.size[0]/2, self.pos[1] + self.size[1]/2)

    def return_weights(self):
        return self.weights
    
    def life_status(self):
        return self.alive
    
    def move(self, track, border_color, finish_color):
        vertical = math.sin(self.angle) * self.vel
        horizontal = math.cos(self.angle) * self.vel

        x, y = self.pos
        x += horizontal
        y-= vertical
        self.pos = (x, y)
        self.update_corners()
        self.check_collision(track, border_color, finish_color)

        self.time +=1
        self.distance += self.vel

    def accelerate(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        
    def decelerate(self):
        self.vel = max(self.vel - self.acceleration, self.min_vel)

    def turn_left(self):
        self.angle += self.rotation_vel

    def turn_right(self):
        self.angle -= self.rotation_vel

    def update_corners(self):
        length = self.size[0]/2
        center_x, center_y = self.center
        
        # python unit circle is a normal unit circle reflected over y axis
        front_left = (
            center_x + math.cos(2*math.pi - self.angle - self.theta_diag) * length,
            center_y + math.sin(2*math.pi - self.angle - self.theta_diag) * length
            )

        front_right = (
            center_x + math.cos(2*math.pi - self.angle + self.theta_diag) * length,
            center_y + math.sin(2*math.pi - self.angle + self.theta_diag) * length
            )

        back_left = (
            center_x + math.cos(math.pi - self.angle + self.theta_diag) * length,
            center_y + math.sin(math.pi - self.angle + self.theta_diag) * length
            )

        back_right = (
            center_x + math.cos(math.pi - self.angle - self.theta_diag) * length,
            center_y + math.sin(math.pi - self.angle - self.theta_diag) * length
            )
        
        self.corners = (front_left, front_right, back_left, back_right)

    def check_collision(self, track:pygame.SurfaceType, border_color:tuple, finish_color:tuple):
        self.alive = True
        for corner in self.corners:
            if track.get_at((int(corner[0]), int(corner[1]))) == border_color:
                self.alive = False
                break
            elif track.get_at((int(corner[0]), int(corner[1]))) == finish_color:
                self.alive = False
                self.finish = True
                break
    
    def check_sensors(self, angle:int, track:pygame.SurfaceType, border_color:tuple, index:int):
        length = 0
        x = int(self.center[0] + math.cos(math.pi*2 - (self.angle+angle)) * length)
        y = int(self.center[1] + math.sin(math.pi*2 - (self.angle+angle)) * length)

        while not track.get_at((x, y)) == border_color and length < self.scan_length:
            length += 1
            x = int(self.center[0] + math.cos(math.pi*2 - (self.angle+angle)) * length)
            y = int(self.center[1] + math.sin(math.pi*2 - (self.angle+angle)) * length)
        
        self.sensors_dist[index] = length
        self.sensors_pos[0][index] = x
        self.sensors_pos[1][index] = y

    def forward_prop(self):
        out = np.array((self.sensors_dist,))
        idx = 0

        for i in range(len(self.nn_size)-1):
            if self.bias_term:
                inp = np.concatenate((self.one, out), axis=1)
            weight_shape = ((self.nn_size[i]+self.bias_term), self.nn_size[i+1])
            num_weights = np.prod(weight_shape)
            weight = self.weights[idx:idx+num_weights]
            weight = np.reshape(weight, weight_shape)
            out = inp @ weight
            out = tools.sig(out)
            idx += num_weights
        
        result = (out > 0.5)
        self.command[:] = result[:]

    def update(self, win, track, border_color, finish_color, show_radar:bool):
        if self.alive:
            # check sensors
            for idx, angle in self.sensor_angles:
                self.check_sensors(angle, track, border_color, idx)
                
            # forward prop
            self.forward_prop()
            # perform action
            for i in range(len(self.command)):
                if self.command[i]:
                    self.actions[i]()
            
            # move
            self.move(track, border_color, finish_color)

            # update variables
            self.image = pygame.transform.rotate(self.sprite_image, math.degrees(self.angle))
            self.update_center_pos()

            # draw radar
            if show_radar:
                self.draw_radar(win)
        
        self.draw(win)

    def reset(self, eliminate=False, parent_weight=None):
        self.pos = self.start_pos
        self.update_center_pos()
        if not eliminate:
            self.age += 1
        else:
            self.age = 0
            self.init_weight(parent_weight)

        self.angle = 0
        self.vel = self.start_vel

        self.alive = True
        self.distance = 0
        self.time = 0
        self.finished = False


import random 
from enum import Enum 
from collections import namedtuple 
import math 
import numpy as np 
import cv2 


class Dir(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y')

WIDTH =  20
HEIGHT = 20
BLOCK_SIZE = 40

class SnakeGame:

    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h 
        self.reset() 


    def reset(self):
        self.frame_cnt = 0
        self.dir = Dir.RIGHT 

        self.head = Point(int(self.w/2), int(self.h/2))
        self.body = [
            self.head,
            Point(self.head.x-1, self.head.y),
            Point(self.head.x-2, self.head.y)
        ]
        self.fruit = None
        self.gen_fruit()
        self.frame_cnt = 0
        self.score = 0

        

    def gen_fruit(self):
        x = random.randint(0, self.w-1)
        y = random.randint(0, self.h-1)
        self.fruit = Point(x, y)
        
        if self.fruit in self.body:
            self.gen_fruit()

    
    #return next_state, reward, done, score
    def step(self, action):
        self.frame_cnt += 1

        reward = 0
        done = False 

        prev_dist = self.get_dist(self.head, self.fruit)

        self.update(action)
        self.body.insert(0, self.head)

        curr_dist = self.get_dist(self.head, self.fruit)

        if prev_dist - curr_dist > 0:
            reward = 0.5
        else:
            reward = -0.5

        if self.head == self.fruit:
            self.score += 1
            reward = 10
            self.gen_fruit()
        else:
            self.body.pop()

        next_state = self.get_state()

        if self.game_over() or self.frame_cnt > 100*len(self.body):
            done = True 
            reward = -10
        
        return next_state, reward, done, self.score 
        
        

    def get_state(self):
        #8 ways
        dx = [0,1,1,1,0,-1,-1,-1]
        dy = [-1,-1,0,1,1,1,0,-1]

        state = [ 0,0,    #(dir(8 ways), type(wall or tail or fruit)) * 8
                  0,0,
                  0,0,
                  0,0,
                  0,0,
                  0,0,
                  0,0,
                  0,0,    
                  0,0,0, #head
                  0,0]   #fruit
        for i in range(0, 8):
            offset = i * 2
            distance = 0
            type = 0 #None
            nx = self.head.x 
            ny = self.head.y 
            while True:
                distance += 1
                nx += dx[i] 
                ny += dy[i]
                if self.collision(Point(nx, ny)) == True:
                    type = 1 # wall or tail
                    break
                if Point(nx, ny) == self.fruit:
                    type = 2 # fruit
                    break
            state[offset] = distance 
            state[offset + 1] = type 

        state[16] = int(self.head.x)
        state[17] = int(self.head.y)
        
        if self.dir == Dir.RIGHT:
            state[18] = 1
        elif self.dir == Dir.LEFT:
            state[18] = 2
        elif self.dir == Dir.DOWN:
            state[18] = 3
        elif self.dir == Dir.UP:
            state[18] = 4
            
        state[19] = int(self.fruit.x)
        state[20] = int(self.fruit.y)

        return state 

    def game_over(self):
        return self.collision(self.head)
    
    def collision(self, pt):
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.body[1:]:
            return True 
        return False 
    
    def get_dist(self, pt1, pt2):
        X = (pt1.x - pt2.x) ** 2
        Y = (pt1.y - pt2.y) ** 2
        return math.sqrt(float(X + Y))

    def update(self, action):
        # [1, 0, 0] straight
        # [0, 1, 0] right
        # [0, 0, 1] left

        dirs = [Dir.UP, Dir.RIGHT, Dir.DOWN, Dir.LEFT]
        idx = dirs.index(self.dir)

        if action == 1:
            idx = (idx + 1) % 4
        if action == 2:
            idx = (idx - 1) % 4 

        self.dir = dirs[idx]
        x = self.head.x 
        y = self.head.y
        if self.dir == Dir.UP:
            y -= 1
        if self.dir == Dir.DOWN:
            y += 1
        if self.dir == Dir.LEFT:
            x -= 1
        if self.dir == Dir.RIGHT:
            x += 1
        self.head = Point(x, y)
    
    def get_frame(self):
        m_size = self.h * BLOCK_SIZE, self.w * BLOCK_SIZE, 3
        frame = np.zeros(m_size, dtype=np.uint8)
        #m = cv2.rectangle(m, (0,0), (100,100), (255,0,0))
        for pt in self.body:
            frame = cv2.rectangle(frame, (int(pt.x) * BLOCK_SIZE, int(pt.y) * BLOCK_SIZE), (int(pt.x + 1) * BLOCK_SIZE, int(pt.y + 1) * BLOCK_SIZE), (255,0,0), -1)
        frame = cv2.rectangle(frame, (int(self.head.x) * BLOCK_SIZE, int(self.head.y) * BLOCK_SIZE), (int(self.head.x + 1) * BLOCK_SIZE, int(self.head.y + 1) * BLOCK_SIZE), (0,255,0), -1)
        frame = cv2.rectangle(frame, (self.fruit.x * BLOCK_SIZE, self.fruit.y * BLOCK_SIZE), ((self.fruit.x + 1) * BLOCK_SIZE, (self.fruit.y + 1) * BLOCK_SIZE), (0,0,255), -1)

        return frame
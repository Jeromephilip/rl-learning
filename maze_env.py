import random

import numpy as np
import time
import sys
# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
import tkinter as tk


UNIT = 40   # pixels per cell (width and height)
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells
origin = np.array([UNIT/2, UNIT/2])


class Maze(tk.Tk, object):
    def __init__(self, agentXY=[0,0], goalXY=[9,9], walls=[],pits=[], showRender=True, name=''):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.agentXY = agentXY
        self.goalXY = goalXY

        self.reward_goal = 50
        self.reward_wall = -3
        self.reward_pit = -10
        self.reward_walk = -1


        self.wallblocks = []
        self.pitblocks=[]
        self.showRender = showRender
        self.UNIT = 40   # pixels per cell (width and height)
        self.MAZE_H = 10  # height of the entire grid in cells
        self.MAZE_W = 10  # width of the entire grid in cells
        self.title('Maze {}'.format(name))
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self.build_shape_maze(agentXY, goalXY, walls, pits)


    def build_shape_maze(self,agentXY,goalXY, walls,pits):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        for x,y in walls:
            self.add_wall(x,y)
        for x,y in pits:
            self.add_pit(x,y)
        self.add_goal(goalXY[0],goalXY[1])
        self.add_agent(agentXY[0],agentXY[1])
        self.canvas.pack()

    def add_wall(self, x, y):
        '''Add a solid wall block at coordinate for centre of bloc'''
        wall_center = origin + np.array([UNIT * x, UNIT*y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - 15, wall_center[1] - 15,
            wall_center[0] + 15, wall_center[1] + 15,
            fill='black'))

    def add_pit(self, x, y):
        '''Add a solid pit block at coordinate for centre of bloc'''
        pit_center = origin + np.array([UNIT * x, UNIT*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='blue'))

    def add_goal(self, x=4, y=4):
        '''Add a solid goal for goal at coordinate for centre of bloc'''
        goal_center = origin + np.array([UNIT * x, UNIT*y])

        self.goal = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

    def add_agent(self, x=0, y=0):
        '''Add a solid wall red block for agent at coordinate for centre of bloc'''
        agent_center = origin + np.array([UNIT * x, UNIT*y])

        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

    def reset(self, value = 1, renderNow=False, resetAgent=True):
        if renderNow:
            self.update() #just wait for other processes to finish
        time.sleep(0.2)
        if(value == 0):
            return self.canvas.coords(self.agent)
        else:
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.add_agent(self.agentXY[0],self.agentXY[1])

            return self.canvas.coords(self.agent)

    def computeReward(self, currstate, action, nextstate):
        '''computeReward - definition of reward function'''
        reverse=False
        hit_wall = False
        hit_pit = False
        if nextstate == self.canvas.coords(self.goal):
            reward = self.reward_goal #1
            done = True
            nextstate = 'terminal'
        #elif nextstate in [self.canvas.coords(self.pit1), self.canvas.coords(self.pit2)]:
        elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
            reward = self.reward_wall #-0.3
            done = False
            nextstate = currstate
            reverse=True
            hit_wall = True
            #print("Wall penalty:{}".format(reward))
        elif nextstate in [self.canvas.coords(w) for w in self.pitblocks]:
            reward = self.reward_pit #-10
            done = True
            nextstate = 'terminal'
            reverse=False
            hit_pit = True
            #print("Wall penalty:{}".format(reward))
        else:
            reward = self.reward_walk #-0.1
            done = False
        return reward,done, reverse, {'hit_wall': hit_wall, 'hit_pit': hit_pit}

    def step(self, action, renderNow=False):
        '''step - definition of one-step dynamics function'''
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state

        # call the reward function
        reward, done, reverse, info = self.computeReward(s, action, s_)
        if(reverse):
            self.canvas.move(self.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = self.canvas.coords(self.agent)

        return s_, reward, done, info

    def render(self, sim_speed=.01, renderNow=False):
        if self.showRender:
            time.sleep(sim_speed)
            self.update()


def update():
    for t in range(10):
        print("The value of t is", t)
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done, info = env.step(a)
            if done:
                break


if __name__ == '__main__':
    # env = Maze()
    # env.after(100, update)
    # env.mainloop()

    # mazet = 1 #S25 
    # agentXY=[2,7] # Agent start position
    # goalXY=[7, 8] # Target position, terminal state
    # wall_shape=np.array([[4,9], [4,8], [4,7], [4,6], [5,1],[4,1], [1,1],[1,2]])
    # pits=np.array([[0,3],[7,3], [9,9]])

    # mazet = 2 #S25 actual
    # agentXY=[0,2] # Agent start position
    # goalXY=[2,6] # Target position, terminal state
    # wall_shape=np.array([ [0,3], [0,4], [0,5], [0,6], [0,7], [0,1],[1,1],[2,1],[8,7],[8,5],[8,3],[2,7]])
    # pits=np.array([[1,3], [1,4], [1,5], [1,6], [1,7], [2,5],[8,6],[8,4],[8,2]])


    mazet = 3 #S25 
    agentXY=[0,0] # Agent start position
    goalXY=[5,5] # Target position, terminal state
    wall_shape=np.array([[1,6],[2,5],[3, 4],[4,4],[5, 4],[6, 4],[7,5],[7,6], [5,8], [4,2],[5,2], [7,2],[8,3]])
    pits=np.array([[1,2],[4,6],[6,6],[6 ,8],[6,2], [7,0]])

    env = Maze(agentXY,goalXY,wall_shape, pits, name=f'Task {mazet}')

    # env.after(100, update)
    env.mainloop()
    s = env.reset()
    env.render()

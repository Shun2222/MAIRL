import numpy as np
import random
from pprint import pprint
from MAIRL.environment import GridWorldEnv

def state_to_coordinate(s, col):
    row, col = divmod(s, col)
    return row, col

def coordinate_to_state(nrow, row, col):
    index = row * ncol + col
    return index

def get_random_states(num, state):
    random_states = random.sample(range(state[0]*state[1]), num)
    
    random_coordinate = []
    for s in random_states:
        row, col = state_to_coordinate(s, state[1])
        random_coordinate += [[row, col]]
    return random_coordinate

def create_environment(N_AGENTS, STATE_SIZE, N_OBSTACLES=0):
    start_state = []
    goal_state = []
    obstacles = []
    
    start_state = get_random_states(N_AGENTS, STATE_SIZE)
    while True:
        goal_state = get_random_states(N_AGENTS, STATE_SIZE)
        if all([x!=y for x,y in zip(start_state, goal_state)]):
            break
    # ここ後で直す（重複する可能性）
    tf = True
    while tf:
        obstacles = get_random_states(N_OBSTACLES, STATE_SIZE)
        tf = False
        for o in obstacles:
            if any([s==o for s in start_state]):
                tf = True
            if any([g==o for g in goal_state]):
                tf = True
                    
    print(start_state)
    print(goal_state)

    env = [[] for _ in range(N_AGENTS)]
    for i in range(N_AGENTS):
        e = [[0]*STATE_SIZE[1] for i in range(STATE_SIZE[0])]
        e[start_state[i][0]][start_state[i][1]] = 'S'
        e[goal_state[i][0]][goal_state[i][1]] = 'G'
        if obstacles:
            for o in obstacles:
                if o:
                    e[o[0]][o[1]] = '-1'
        env[i] = GridWorldEnv(grid=e)
    return env
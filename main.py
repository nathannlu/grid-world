import numpy as np

COLUMNS = 4
ROWS = 3
UP=0
DOWN=1
LEFT=2
RIGHT=3

LEARNING_RATE = 0.5
DECAYING_RATE = 1

q_table = np.zeros( (COLUMNS, ROWS, 4) )

agent_pos = [0,2]

def update_qvalue(state, action):
    global agent_pos

    current_qvalue = q_table[state[0], state[1], action]
    if action == UP:
        if state[1] != 0:
            future_qvalue = max(q_table[state[0], state[1] - 1])
            agent_pos = [state[0], state[1] - 1]
        else:
            future_qvalue = current_qvalue
    elif action == DOWN:
        if state[1] != 2:
            future_qvalue = max(q_table[state[0], state[1] + 1])
            agent_pos = [state[0], state[1] + 1]
        else:
            future_qvalue = current_qvalue
    elif action == LEFT:
        if state[0] != 0: 
            future_qvalue = max(q_table[state[0] - 1, state[1]])
            agent_pos = [state[0] - 1, state[1]]
        else:
            future_qvalue = current_qvalue
    elif action == RIGHT:
        if state[0] != 3:
            future_qvalue = max(q_table[state[0] + 1, state[1]])
            agent_pos = [state[0] + 1, state[1]]
        else:
            future_qvalue = current_qvalue

    if agent_pos == [3,0]:
        REWARD = 1
        agent_pos = [0,2]
    elif agent_pos == [3,1]:
        REWARD = -1
        agent_pos = [0,2]
    else:
        REWARD = -0.04
    current_qvalue += LEARNING_RATE * (REWARD + DECAYING_RATE * (future_qvalue - current_qvalue))
    q_table[state[0], state[1], action] = round(current_qvalue, 2)

def choose_move(state):
    current_state = q_table[state[0], state[1]]
    best_move = np.argmax(current_state)

    update_qvalue(state, best_move)

for _ in range(1000):
    choose_move(agent_pos)


print(agent_pos)
print(q_table)

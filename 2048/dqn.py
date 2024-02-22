import torch
import torch.nn as nn
import numpy as np
import random
import logic
from collections import namedtuple, deque
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.FT = nn.Flatten()
        self.relu = nn.ReLU()
        self.ln = nn.Linear(16,256)
        self.ln2 = nn.Linear(256,4)
    def forward(self,x):
        x = self.FT(x)
        x = self.relu(x)
        x = self.ln(x)
        x = self.relu(x)
        return self.ln2(x)

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def string_m(m):
    s = ""
    for i in range(len(m)):
        for j in range(len(m[0])):
            s+=str(m[i][j])
    return s


policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

state = (4*4)**2
actions = 4
alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([state, actions])
num_of_episodes = 50
actis = [logic.move_left,logic.move_right,logic.move_up,logic.move_down]

def select_action(state):
    sample = random.random()
    if random.uniform(0,1) < epsilon:
        with torch.no_grad():
            return policy_net(state)
    else:
        return torch.tensor([[random.randrange(4)]], device = device ,dtype = torch.long)

for episode in range(num_of_episodes):
    
    state = random.choice(range(0,256))
    board = logic.start_game()
    
    for t in count():
        action = select_action(state)
        o_s  = string_m(board)
        terminated = logic.get_current_state(board)
        if terminated:
            break
        board,_,reward = actis[action](board)
        m_s = string_m(board)
        if o_s != m_s:
            board = logic.add_new_2(board)
        next_state = random.choice(range(0,256))
        memory.push(state,action,next_state,reward)
        state = next_state
        optimize_model()
        if t%10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    if (episode + 1) % 1 == 0:
        print("Episode: {}".format(episode + 1))
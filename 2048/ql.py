import torch
import torch.nn as nn
import numpy  as np
from collections import namedtuple, deque
import random
import logic
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from itertools import count
import math
import shutil

dest = r"C:\Users\Jorge Eliecer\Desktop\server\static\logs\logs.txt"

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)
alpha = 0.1
GAMMA = 0.9
#epsilon = 0.6
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# class DQN(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.fn = nn.Flatten()
        # self.conv1 = nn.Conv2d(4,16,1,1)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(16,32,1,1)
        # self.conv3 = nn.Conv2d(32,64,1,1)
        # self.ll = nn.Linear(256,4)
        
    # def forward(self,x):

        # x = x.unsqueeze(0).reshape(12,4,4,1)
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))

        # return self.ll(x.view(x.size(0),-1))
        
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Flatten()
        self.conv1 = nn.Conv2d(4,16,kernel_size=(2,1))
        self.lin = nn.Linear(96,256)
        self.lin2 = nn.Linear(256,4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16,32,1,1)
        self.conv3 = nn.Conv2d(32,32,1,1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        
        
        
    def forward(self,x):
        # if x.size(0)>=12:
        #print(x.size())
        x = x.permute(1,2,0).unsqueeze(0)
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(x.size(3),x.size(1)*x.size(2))
        return self.lin2(self.relu(self.lin(x)))
        # else:
            # x = x.unsqueeze(0).reshape(1,4,4,1)
            # x = self.relu(self.conv1(x))
            # x = self.bn1(x)
            # x = self.relu(self.conv2(x))
            # x = self.bn2(x)
            # x = self.relu(self.conv3(x))
            # x = self.bn3(x)
            # x = self.ll(x.view(x.size(0),-1))
            # return x.reshape(1,4)
            
            #x = self.ln(x).reshape(4,4,4,4)
            
            
policy_dqn = DQN().to(device)
target_dqn = DQN().to(device)
target_dqn.load_state_dict(policy_dqn.state_dict())
target_dqn.eval()

# def change_values(X):
    # power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
    # for i in range(4):
        # for j in range(4):
            # if(X[i][j]==0):
                # power_mat[0][i][j][0] = 1.0
            # else:
                # power = int(math.log(X[i][j],2))
                # power_mat[0][i][j][power] = 1.0
    # return power_mat   
    
steps_done = 0

def string_m(m):
    #m = m.numpy
    #print(m)
    s = ""
    for i in range(len(m)):
        for j in range(len(m[0])):
            s+=str(m[i][j])
    return s

def select_action(state):
    global steps_done
    #sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.uniform(0,1) < eps_threshold:
        return torch.as_tensor([[random.randrange(4)]]).to(device)
    else:
        #state = torch.as_tensor(change_values(state)).float().to(device)
        with torch.no_grad():
            state = state.unsqueeze(0)
            return torch.argmax(policy_dqn(state)).unsqueeze(0).unsqueeze(0) #policy_dqn(state).max(1)[1].view(1, 1)#
    
def step(board,o_s):
    state = board
    actis = [logic.move_left,logic.move_right,logic.move_up,logic.move_down]
    action = select_action(state)
    next_state,_,reward = actis[action](state)
    #print(o_n)
    #print(next_state)
    o_n = string_m(next_state)
    #o_s = o_s
    #print(o_n == o_s)
    
    if o_s!= o_n:
        next_state = logic.add_new_2(next_state)
    terminated = logic.get_current_state(next_state)
    #next_state = torch.Tensor(next_state).to(device).float().requires_grad_()
    next_state = torch.as_tensor(next_state).to(device)
    #print(next_state.size())
    reward = torch.as_tensor([reward]).to(device)
    #print(reward)
    return next_state, reward, terminated
    
Transition = namedtuple('Transition',
                         ('state', 'action',  'reward','next_state'))
        
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)   
        
        
memory = ReplayMemory(6000)

optimizer = torch.optim.RMSprop(policy_dqn.parameters())
#board = torch.Tensor(logic.start_game()).unsqueeze(dim=-1).unsqueeze(dim=-1).float()
#board = torch.from_numpy(board)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "2048_m.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


N = 10000
losses = []
total_epochs = 0
total_penalties = 0
t = []
rws = []
batch_size = 12
C = 10
#state = torch.Tensor(logic.start_game()).float().to(device)
t_reward = []

# def plot_durations():
    # plt.figure(2)
    # plt.clf()
    # durations_t = torch.tensor(t_reward, dtype=torch.float)
    # plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('Total reward')
    # plt.plot(durations_t.numpy())
    # # Take 100 episode averages and plot them too
    # #plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
        # display.clear_output(wait=True)
        # display.display(plt.gcf())
    
#print(target_dqn(state).size())
with open(dest,'w',buffering=1) as f:
    for epoch in tqdm(range(N)):
        state_number = logic.start_game()
        #state_dqn = change_values(state_number)
        #state_t = torch.Tensor(state_number).float().to(device).requires_grad_()
        state_t = torch.as_tensor(state_number).float().to(device)
        
        total_reward = 0
        steps = 0
        total_loss = []
        penalties = 0
        epochs = 0
        
        for _ in count():
            #print(steps)
            action = select_action(state_t)
            steps += 1
            o_s = string_m(state_number)
            #print(o_s[0:2])
            #print(o_s)
            state_next, reward,terminated = step(state_t,o_s)
            state_number_next = state_next
            state_next_tensor = state_next.unsqueeze(0)
            #print(state_next)
            
            rd = reward[0].cpu().numpy()
            #print(rd)
            total_reward += rd
            #print(total_reward)
            #print(reward)
            #print(terminated)
            if reward == 0:
                penalties+=1
            if terminated:
                #print(state_next) 
                #t_reward.append(total_reward)
                #plot_durations()
                break
            
            #if (action)!=0:
            state_tensor = state_t.unsqueeze(0)
            memory.push(state_tensor,action,reward,state_next_tensor)
            
            if len(memory)>= batch_size:
                #print(state_next)
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                
                state_batch = torch.cat(batch.state).float()
                action_batch = torch.cat(batch.action).float()
                reward_batch = torch.cat(batch.reward).float()
                #print(state_batch)
                state_action_values = policy_dqn(state_batch).gather(1, action_batch.long())
                next_state_values = torch.zeros(batch_size).to(device)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)),  dtype=torch.bool).long().to(device)

                non_final_next_states = torch.cat([s for s in batch.next_state
                                                                if s is not None]).float().to(device)

                next_state_values[non_final_mask] = policy_dqn(non_final_next_states).max(1)[0].detach()
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                
                # for i in range(batch):
                    # #policy_dqn.zero_grad()  
                    
                    # 
                    
                    # State, Action,  Reward, Next_state = transitions[0].state,transitions[0].action,transitions[0].reward,transitions[0].next_state
                    # #print(State,Action,Reward,Next_state)
                    # if i+1==batch:
                        # target.append(torch.as_tensor(Reward).to(device))
                    # else:
                        # #Next_state = torch.as_tensor(change_values(Next_state)).float().to(device)
                        # target.append((Reward + (gamma*(target_dqn(Next_state).max()))).unsqueeze(dim=0))
                    # #State = torch.as_tensor(change_values(State)).float().to(device)
                    # state_action_values.append(policy_dqn(State).squeeze()[Action].unsqueeze(dim=0))
                
                #state_action_values = torch.Tensor(state_action_values).to(device).float().requires_grad_()
                #state_action_values = torch.as_tensor(state_action_values).to(device).requires_grad_()
                #target = torch.Tensor(target).to(device).float().requires_grad_()
                #target = torch.as_tensor(target).to(device).requires_grad_()
                #print(target)
                
                
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                #print(loss)
                for param in policy_dqn.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                total_loss.append(loss.item())

            if steps % C == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                #target_dqn = policy_dqn
            state_t = state_next_tensor.squeeze()
            state_number = state_number_next
        rws.append(total_reward)
        t.append(epoch+1)
        total_penalties += penalties
        total_epochs += steps
        #s = np.array(state_next)
         
        f.write(f"epoch: {epoch+1}  [#]steps: {steps} [#] total_reward: {total_reward} [#] penalties: {penalties} [#] avg_total_loss: {sum(total_loss)/len(total_loss)} ")
        f.write(f"\n")
        f.write(f"{state_next}")
        f.write(f"\n")
        #f.write(f"loss: {loss.item()}")
        #f.write(f"\n")
        #if steps>500:#total_reward>3000 and steps>400:
            
            #print(f"epoch: {epoch+1} [#] steps: {steps} [#] total_reward: {total_reward} [#] penalties: {penalties} [#] total_loss: {total_loss}")
            #print(state_next)
        #if (epoch+1)%100 ==0:
            #print(state_action_values.size())
            #print(f"epoch: {epoch+1} [#] board : {state_next} [#] steps: {steps} [#] total_reward: {total_reward} [#] penalties: {penalties} [#] total_loss: {total_loss}")
#print("Mean toral reward", np.mean(total_reward))
#print("Epochs per episode: {}".format(total_epochs / 100))
#print("Penalties per episode: {}".format(total_penalties / 100)) 
     
torch.save(obj=policy_dqn.state_dict(), # only saving the state_dict() only saves the models learned parameters
                   f=MODEL_SAVE_PATH) 
                   

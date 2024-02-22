import torch
from torch import nn
from collections import namedtuple, deque
import random
import numpy as np
import math
    
def change_values(X):
    power_mat = np.zeros(shape=(1,4,4,16),dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(X[i][j]==0):
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j],2))
                power_mat[0][i][j][power] = 1.0
    return power_mat    
    
# #x_in = np.array([[
  # [[2], [1], [2], [0], [1]],
  # [[1], [3], [2], [2], [3]],
  # [[1], [1], [3], [3], [0]],
  # [[2], [2], [0], [1], [1]],
  # [[0], [0], [3], [1], [2]], ]])
  
# #kernel_in = np.array([
 # [ [[2, 0.1]], [[3, 0.2]] ],
 # [ [[0, 0.3]], [[1, 0.4]] ], ])
 
#ff = torch.nn.Flatten()
ln = nn.Linear(2176,4)
cv1 = nn.Conv2d(4,128,kernel_size = (1,14), stride= 1)
cv2 = nn.Conv2d(4,128,kernel_size = (1,14), stride= 1)

cv_l11 = nn.Conv2d(4,3,kernel_size = (1,1), stride= 1)
cv_l12 = nn.Conv2d(4,4,kernel_size = (2,1), stride= 1)

cv_l21 = nn.Conv2d(3,3,kernel_size = (2,1), stride= 1)
cv_l22 = nn.Conv2d(3,2,kernel_size = (1,1), stride= 1)

f = torch.Tensor([[0,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]])
f = torch.as_tensor(change_values(f))

print(f.shape)

f1 = cv1(f).reshape(1,4,3,128)
f1_1 = cv_l11(f1)
f1_2 = cv_l12(f1)


f2 = cv2(f).reshape(1,3,4,128)
f2_1 = cv_l21(f2)
f2_2 = cv_l22(f2)
l1 = f2_1.view((-1,1)).squeeze()
l2 = f2_2.view((-1,1)).squeeze()
l3 = f1_1.view((-1,1)).squeeze()
l4 = f1_2.view((-1,1)).squeeze()

l = torch.cat((l1,l2,l3,l4))
#print(l.shape)
#print(ln(l).shape)
print(l.shape)
#print(cv_2(f2).shape)

#idx = random.choices(range(30000), k=32)
#rw = torch.zeros(30000,1)
#pt = torch.zeros(30000,16)
#print(x_in.shape)
#print(kernel_in.shape)
#print(idx)
#print(rw[idx])
#print(pt[idx])
#f1 = torch.nn.Linear(16,256)
#f_m = f1(f).squeeze().reshape(4,4,4,4)
#conv1 = torch.nn.Conv2d(4,16,2,2)
#conv2 = torch.nn.Conv2d(16,64,2,2)
#print(f1(f).size())
#print(f_m.size())
#print(conv1(f_m).size())
#print(conv2(conv1(f_m)).size())
#print(ff(conv2(conv1(f_m))).view((-1,1)).reshape(256,1).size())

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Flatten()
        #self.conv1 = nn.Conv2d(16,32,kernel_size = 5,stride = 2)
        #self.conv2 = nn.Conv2d(32,64,kernel_size = 5, stride = 2)
        #self.ln = nn.Linear(1,4)
        #self.ln = nn.Linear(1600,)
        self.conv1 = nn.Conv2d(4,16,2,2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16,64,2,2)
        self.conv3 = nn.Conv2d(8,32,2,2)
        self.ln = nn.Linear(16,256)
        self.ln1 = nn.Linear(256,4)
        self.ln2 = nn.Linear(512,1024)
        self.ln3 = nn.Linear(1024,4)
    def forward(self,x):
        #x = x.view((-1,1)).reshape(1,1600).squeeze().reshape(4,16,5,5)
        #x = self.conv1(self.relu(x))
        #x = self.conv2(self.relu(x))
        #x = self.fn(x)
        x = x.view((-1,1)).reshape(1,16)
        x = self.ln(x).squeeze().reshape(4,4,4,4)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.fn(x).view((-1,1)).reshape(256,1).squeeze()
        x = self.ln1(x)
        #x = self.relu(x)
        #
        #print(x.size())
        
        #x = self.ln2(x).squeeze()
        
        #x = self.relu(x).reshape(8,8,4,4)
        #x = self.conv3(x)
        #x = self.relu(x)
        #x = self.fn(x).view((-1,1)).squeeze()
        #x = self.ln3(x)
        return self.relu(x)
        
model  = DQN()
#print(model(f).size())
#print(model(f))
#print(model(f).size())
#print(model(f))

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
        
        
# memory = ReplayMemory(1000)
# memory.push(f,2,0,f)
# memory.push(f,0,8,f)
# memory.push(f,3,4,f)
# memory.push(f,1,2,f)

# batch = 3
# transitions = memory.sample(batch)

# batch = Transition(*zip(*transitions))
# #print(batch)
# state_batch = torch.cat(batch.state)
# action_batch = batch.action
# reward_batch = batch.reward

# #state_action_values = policy_net(state_batch)#.gather(1, action_batch)
# t = model(state_batch).gather(1,action_batch)

#print(state_batch.view((-1,1)).size())

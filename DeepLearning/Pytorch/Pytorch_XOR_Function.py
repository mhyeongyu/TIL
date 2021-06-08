#!/usr/bin/env python
# coding: utf-8

# In[174]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')


# In[175]:


X = torch.from_numpy(np.array([[0,0], [1,0], [0,1], [1,1]]))
y = torch.from_numpy(np.array([[0], [1], [1], [0]]))

X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)


# In[176]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


# In[177]:


net = Net()


# In[178]:


print(net.fc1.weight)
print(net.fc2.weight)


# In[179]:


criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# In[180]:


for epoch in range(1500):
    running_loss = 0.0
    
    for i, data in enumerate(zip(X, y)):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 4 == 3:
            print('[{0}, {1}] loss: {2:.4f}'.format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
        
print('Fnish')


# In[181]:


print(net.fc1.weight)
print(net.fc2.weight)


# In[182]:


net(torch.tensor([[0,0], [1,0], [0,1], [1,1]], dtype = torch.float32))


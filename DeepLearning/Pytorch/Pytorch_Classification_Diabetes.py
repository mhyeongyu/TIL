#!/usr/bin/env python
# coding: utf-8

# In[388]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[389]:


df = pd.read_csv('diabetes_data.csv')
df = df.drop('insulin', axis=1)

X = df.drop('result', axis=1).values
y = df['result']

X = torch.FloatTensor(X)
y = torch.FloatTensor(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[390]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


# In[391]:


model = Net()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[392]:


model.eval()
test_loss_before = criterion(model(X_test).squeeze(), y_test)
print('Before Training, test loss is {:.4f}'.format(test_loss_before.item()))


# In[393]:


epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    train_output = model(X_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    
    if epoch % 50 == 0:
        print('Epoch {} Train loss: {:.4f}'.format(epoch, train_loss.item()))
    
    train_loss.backward()
    optimizer.step()


# In[387]:


model.eval()
test_loss = criterion(torch.squeeze(model(X_test)), y_test)
print('After Training, test loss is {:.4f}'.format(test_loss.item()))


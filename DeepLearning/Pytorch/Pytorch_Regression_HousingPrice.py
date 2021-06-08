#!/usr/bin/env python
# coding: utf-8

# In[98]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[99]:


df = pd.read_csv('bostonhousing_data.csv')

features = df.drop(['MEDV'], axis=1).values
labels = df['MEDV'].values

features = torch.FloatTensor(features)
labels = torch.FloatTensor(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.3,
                                                    random_state=0)


# In[100]:


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x


# In[101]:


model = NeuralNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[102]:


model.eval()
test_loss_before = criterion(model(X_test).squeeze(), y_test)
print('Before Training, test loss {:.4f}'.format(test_loss_before.item()))


# In[103]:


epochs = 500

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    train_output = model(X_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    
    if epoch % 50 == 0:
        print('Epoch {} Train loss: {:.4f}'.format(epoch, train_loss.item()))
    
    train_loss.backward()
    optimizer.step()


# In[104]:


model.eval()
test_loss = criterion(torch.squeeze(model(X_test)), y_test)
print('After Training, test loss is {:.4f}'.format(test_loss.item()))


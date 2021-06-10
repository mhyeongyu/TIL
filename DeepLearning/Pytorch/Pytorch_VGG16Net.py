#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
import torchvision.models as models


# In[2]:


transform = transforms.Compose([
    transforms.ToTensor()
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     transforms.Resize((224, 224))
])


# In[3]:


train_dataset = datasets.CIFAR10(root='./CIFAR10',
                              train=True,
                              download=True,
                              transform=transform)

test_dataset = datasets.CIFAR10(root='./CIFAR10',
                             train=False,
                             download=True,
                             transform=transform)


# In[4]:


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=512,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=512,
                                          shuffle=False)


# In[5]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[6]:


# for X_train, y_train in train_loader:
#     break
    
# for i in range(2):
#     plt.imshow(np.transpose(X_train[i], (1,2,0)))
#     plt.axis('off')
#     plt.show()
    
# print(X_train.shape)


# In[7]:


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = models.vgg16().to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 1

for epoch in tqdm(range(epochs)):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
#         inputs, labels = data[0], data[1]
    
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[{} / {}] loss: {:.4f}'.format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
print('Finish')


# In[ ]:


dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = model(images)


# In[ ]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# In[ ]:


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[ ]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


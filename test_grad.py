import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


a = torch.tensor([2., 3.])
a = Variable(a, requires_grad=True)
l = nn.Linear(2, 3)
optimizer = optim.Adam(l.parameters(), lr=0.01)

print('0')
for i in l.parameters():
    print(i.grad)

optimizer.zero_grad()
print('1')
for i in l.parameters():
    print(i.grad)

l(a).sum().backward()
print('2')
for i in l.parameters():
    print(i.grad)

optimizer.step()
print('3')
for i in l.parameters():
    print(i.grad)


optimizer.zero_grad()
print('1')
for i in l.parameters():
    print(i.grad)

l(a).sum().backward()
print('2')
for i in l.parameters():
    print(i.grad)

optimizer.step()
print('3')
for i in l.parameters():
    print(i.grad)
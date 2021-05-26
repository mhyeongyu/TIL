import torch

#scalar, 상수

scalar1 = torch.tensor([1.])
scalar2 = torch.tensor([3.])
print(scalar1)
print(scalar2)

print('\n')

add = scalar1 + scalar2
sub = scalar1 - scalar2
mul = scalar1 * scalar2
div = scalar1 / scalar2

print(add)
print(sub)
print(mul)
print(div)

print('\n')

print(torch.add(scalar1, scalar2))
print(torch.sub(scalar1, scalar2))
print(torch.mul(scalar1, scalar2))
print(torch.div(scalar1, scalar2))

#vector, 벡터

vector1 = torch.tensor([1., 2., 3.])
vector2 = torch.tensor([4., 5., 6.])
print(vector1)
print(vector2)

print('\n')

add = vector1 + vector2
sub = vector1 - vector2
mul = vector1 * vector2
div = vector1 / vector2

print(add)
print(sub)
print(mul)
print(div)

print('\n')

print(torch.add(vector1, vector2))
print(torch.sub(vector1, vector2))
print(torch.mul(vector1, vector2))
print(torch.div(vector1, vector2))
print(torch.dot(vector1, vector2))

#matrix, 행렬

matrix1 = torch.tensor([[1., 2.], [3., 4.]])
matrix2 = torch.tensor([[5., 6.], [7., 8.]])
print(matrix1)
print(matrix2)

print('\n')

add = matrix1 + matrix2
sub = matrix1 - matrix2
mul = matrix1 * matrix2
div = matrix1 / matrix2

print(add)
print(sub)
print(mul)
print(div)

print('\n')

print(torch.add(matrix1, matrix2))
print(torch.sub(matrix1, matrix2))
print(torch.mul(matrix1, matrix2))
print(torch.div(matrix1, matrix2))
print(torch.matmul(matrix1, matrix2))

#tensor, 텐서

tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])
print(tensor1)
print(tensor2)

print('\n')

add = tensor1 + tensor2
sub = tensor1 - tensor2
mul = tensor1 * tensor2
div = tensor1 / tensor2

print(add)
print(sub)
print(mul)
print(div)

print('\n')

print(torch.add(tensor1, tensor2))
print(torch.sub(tensor1, tensor2))
print(torch.mul(tensor1, tensor2))
print(torch.div(tensor1, tensor2))
print(torch.matmul(tensor1, tensor2))


if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

BATCH_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10

x = torch.randn(BATCH_SIZE,
                INPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

y = torch.randn(BATCH_SIZE,
                OUTPUT_SIZE,
                device=DEVICE,
                dtype=torch.float,
                requires_grad=False)

w1 = torch.randn(INPUT_SIZE,
                 HIDDEN_SIZE,
                 device=DEVICE,
                 dtype=torch.float,
                 requires_grad=True)

w2 = torch.randn(HIDDEN_SIZE,
                 OUTPUT_SIZE,
                 device=DEVICE,
                 dtype=torch.float,
                 requires_grad=True)

learning_rate = 1e-6
for t in range(1, 501):
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  loss = (y_pred - y).pow(2).sum()
  
  if t % 50 == 0:
    print('Iteration:', t, '\t', 'Loss: {0:.4f}'.format(loss.item()))
  loss.backward()

  with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    w1.grad.zero_()
    w2.grad.zero_()
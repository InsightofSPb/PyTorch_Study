import torch
import pandas as pd
import numpy as np
import time

start_time = time.time()

# Introduction to Tensors

# scalar

scalar = torch.tensor(7)
# print(scalar.item())  # возвращает число

# vector
vector = torch.tensor([7,7])
# print(vector.shape)

# matrix

MATRIX = torch.tensor([[7, 8], [9 ,10]])
# print(MATRIX.shape)

# Tensor
TENSOR = torch.tensor([[[[1,2,3],[3,6,9],[2,4,5]]]])  # получаем 1 3 х 3 тензор. Первая 3 за строки, вторая за элементы
# print(TENSOR.shape)

# Random tensors
random_tensor = torch.rand(3, 4)
# print(random_tensor)
# print(random_tensor.ndim)

# size similar to image
r_image_size_t = torch.rand(size=(3,255,255)) # h, w, layer
# print(r_image_size_t.shape, r_image_size_t.ndim)

# Zeros and ones

zero = torch.zeros(size=(3,4))

# Use torch.range(), arange()
T = torch.arange(start=1, end=11, step=1)
# print(T)
# Creating tensors like - when we dont want to define the shape

ten_zeroes = torch.ones_like(input=T)
# print(ten_zeroes)

# Float 32 tensor
fl_32_tensor = torch.tensor([3.0,6.0,9.0], dtype=None, device='cuda', requires_grad=False)  # datatype - 64,32,16 float + documentation
# device - cpu, cuda; requires_grad - for gradient tracking
# print(fl_32_tensor.type)
fl_16 = fl_32_tensor.type(torch.float16)
# print(fl_16 * fl_32_tensor)  # всё считает, не смотря на разные типы


# Manipulating tensors

tensor = torch.tensor([1, 2, 3])

# print(tensor * tensor)

# Uses operation of vectorization: параллелизм на уровне данных
M = torch.matmul(tensor,tensor)  # .mm
# print(M)

A = torch.tensor([[1, 2], [3, 4], [5, 6]])
B = torch.tensor([[7, 10], [8, 11], [9, 12]]).T

C = torch.mm(A, B)
# print(C)

x = torch.arange(0, 100, 10)
# print(torch.mean(x.type(torch.float32)))  # require float or int, not long
# argmax(), argming() for finding position


# Reshaping, stacking, squeezing, unsquezing
# Reshaping - reshape an input tensor to a defined shape
# View - return a view of an input tensor of certain shape but keep the same memory as the original tensor
# Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# Squeeze - removes all 1 dimensions from a tensor
# Unsqueeze - add 1
# Permute - return a view of the input with dimensions permuted (swapped) in a certain way

x = torch.arange(1., 10.)

x_reshaped = x.reshape(1, 9)
# print(x_reshaped)

## View

z = x.view(1, 9)  # changing z changes x (sharing memory)
z[:, 0] = 5
# print(z, x)

# Stack
x_stacked = torch.stack([x, x, x, x], dim=-2)
# print(x_stacked)

# print(x_reshaped.squeeze().shape)

# Permute
x_original = torch.rand(size=(255, 255, 3))

x_permuted = x_original.permute(2, 0, 1)
# print(x_permuted, f'New shape: {x_original}', sep='\n')

# Indexing

x = torch.arange(1, 10).reshape(1, 3, 3)
# print(x[:, :, 2])


# Numpy and pytorch

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)  # type is float64, not default float32
# print(array, tensor, sep='\n')

array = array + 1
# print(array, tensor)  # new tensor in memory is created, not view or lists

tensor = torch.ones(7)
np_ten = tensor.numpy()

tensor = tensor + 1  # do not share memory too


# Reproducibility

r_t_A = torch.rand(3, 4)
r_t_B = torch.rand(3, 4)
# print(r_t_A, r_t_B, r_t_A == r_t_B, sep='\n')

random_seed = 42
torch.manual_seed(random_seed)

r_t_C = torch.rand(3, 4)

torch.manual_seed(random_seed)  # подходит для 1 только
r_t_D = torch.rand(3, 4)

# print(r_t_C, r_t_D, r_t_C == r_t_D, sep='\n')

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(device)

# Putting tensors and models on the GPU

tensor = torch.tensor([1, 2, 3])

# print(tensor, tensor.device)

tensor_gpu = tensor.to(device)
# print(tensor_gpu)


# Back to cpu (numpy on cpu only)

t_cpu = tensor_gpu.cpu().numpy()
print(t_cpu)

print()
print(f'Calculation time: {time.time() - start_time}')
import torch
import numpy as np

# Criando os dados e passando pro tensor
data = [1,2], [3,4]
x_data = torch.tensor(data)

# transformando os dados em um array e trazendo pro tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# O novo tensor retém as propriedades do argumento do tensor 
# (Shape,datatype) 

x_ones = torch.ones_like(x_data) #Retem as propriedades de x_data
print(f'Ones Tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype=torch.float) #Sobrescreve o datatype de x_data
print(f'Random Tensor: \n {x_rand} \n')

# Com valores de constantes randomicas

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Os atributos dos tensores descrevem o shape, datatype e espaço de armazenamento
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Comando para mover o tensor para ser preocessado na GPU
# # We move our tensor to the GPU if available
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')


tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

#torch.cat concatena a sequencia dos tensores dentre as dimensões recebidas
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


#Concatena os tensores em forma de stack
t1 = torch.stack([tensor, tensor, tensor], dim=1)
print(t1)


#Algumas operações Aritmeticas com tensores

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Se tivermos um tensor com somente 1 elemento, feito por agregação de todos os valores do tensor
# em um único valor, podemos convertê-lo em valor numerico em Python usando item():

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# Operações que armazenam o resultado em um operador chamado in-place. Estes, tem a notação _ em
# seu sufixo. EX: x.copy_(y), x.t_() altera o x.

print(tensor, "\n")
tensor.add_(5)
print(tensor)

##### Operações com IN-PLACE são desencorajadas, mesmo salvando algum espaço na memoria, estas 
# podem se tornar problematicas quando utilizam derivativas por sua perda imediata de memoria



# Bridge com Numpy

# Tensores na CPU e Numpy arrays podem compartilhar seu espaço alocado na memoria, logo, alterando
# um o outro é alterado.

t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

t.add_(1)
print(f't: {t}')
print(f'n: {n}')

# Numpy array para Tensor

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)
print(f't: {t}')
print(f':n: {n}')
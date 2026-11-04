import tensorflow as tf
import numpy as np

a = tf.constant(10)
b = tf.constant(20)
print("Addition:", (a + b).numpy())

x = tf.constant([[1, 2], [3, 4]])
print("2D Tensor:\n",x.numpy())

print("\nAddition:\n", (x + 2))
print("Subtraction:\n", (x - 2))
print("Multiplication:\n", (x * 2))
print("Division:\n", (x / 2))
print("Modulo:\n", (x % 2))

print("\nZeros:\n", tf.zeros((2, 4)))
print("Ones:\n", tf.ones((3, 4)))
print("Random tensor:\n", tf.random.uniform((2, 2)))

print("\nShape:", x.shape)
print("Rank:", tf.rank(x).numpy())
print("Data type:", x.dtype)

y = tf.reshape(x, (4,))
print("\nReshaped tensor:", y.numpy())

m = tf.constant([[1, 2], [3, 4]])
n = tf.constant([[5, 6], [7, 8]])
print("\n2D Matrix multiplication:\n", tf.matmul(m, n))

p = tf.constant([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
q = tf.constant([
    [[1, 0], [0, 1]],
    [[2, 0], [0, 2]]
])
print("\n3D Tensor shape:", p.shape)
print("3D Matrix multiplication:\n", tf.matmul(p, q).numpy())

print("\nFirst row:", x[0].numpy())
print("Element:", x[0, 1].numpy())
print("First column:", x[:, 0].numpy())
print("\nSum:", tf.reduce_sum(x).numpy())

print("\nTranspose:\n", tf.transpose(x).numpy())

arr = x.numpy()
print("\nNumPy array:\n", arr)
print("Array type:", type(arr))

arr2 = np.array([10, 20, 30])
t = tf.convert_to_tensor(arr2)
print("\nNumPy to TensorFlow:", t.numpy())

print("\nIs each element greater than 2?\n", (x > 2).numpy())

import torch

x = torch.tensor([[1, 2], [3, 4]])
print("Tensor:\n", x)

print("Ones:\n", torch.ones(2, 2))
print("Zeros:\n", torch.zeros(2, 2))
print("Random:\n", torch.rand(2, 2))

print("Addition:\n", x + 2)
print("Subtraction:\n", x - 2)
print("Multiplication:\n", x * 2)
print("Division:\n", x / 2)
print("Modulo:\n", x % 2)

print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data Type:", x.dtype)

print("Reshape:\n", x.reshape(4))
print("First Row:", x[0])
print("First Column:", x[:, 0])

print("Sum:", x.sum())
print("Mean:", x.float().mean())
print("Maximum:", x.max())

print("Transpose:\n", x.T)

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print("2D Multiplication:\n", torch.matmul(a, b))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("GPU Available:", torch.cuda.is_available())
print("Device:", x.to(device).device)

arr = np.array([10, 20, 30])
tensor = torch.from_numpy(arr)
print("NumPy to PyTorch:\n", tensor)

print("PyTorch to NumPy:\n", tensor.numpy())

############################
# Author: Malik Al Ashter Ghansletwala
# Date added: 8.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\report\Code\NumPy\NumPy.py
# Version: 2
# Reviewed by: Malik Al Ashter Ghansletwala
# Review Date: 9.02.2024
############################

# Use an environment rather than install in the base env
conda create -n my-env
conda activate my-env
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install numpy

pip install numpy


import numpy as np


# NumPy Version Check
import numpy 
print(numpy.__version__)

# NumPy Example Files
import numpy as np
arr = np.array([1, 2, 3, 4]) np.save('my_array', arr)
loaded_array = np.load('my_array.npy')

#Creating Array

arr_1d = np.array([1, 2, 3, 4, 5])
arr_zeros = np.zeros((2, 3))
arr_identity = np.eye(3)
arr_range = np.arange(1, 10, 2)
arr_linspace = np.linspace(0, 1, 5)

#Array Operations
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
result_add = arr1 + arr2
result_mul = arr1 * arr2
result_dot = np.dot(arr1, arr2)
sin_arr = np.sin(arr1)

#Indexing and Slicing in Arrays
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
element = arr[1, 2]
row_slice = arr[1]
col_slice = arr[:, 1]
bool_index = arr[arr > 5] 

#Shape Manipulation in Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])
reshaped_arr = arr.reshape(3, 2)
transposed_arr = arr.T
flattened_arr = arr.flatten()

#Linear Algebra Operations

arr = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(arr)
inverse = np.linalg.inv(arr)
eigenvalues, eigenvectors = np.linalg.eig(arr)

# Solving linear equations

A = np.array([[2, 3], [1, -1]])
b = np.array([8, 1])
solution = np.linalg.solve(A, b)


# Importing Data

# Load data from a text file
data = np.loadtxt('data.txt', delimiter=',')
# Load data from a CSV file
data = np.genfromtxt('data.csv', delimiter=',')
# Load data from a NumPy binary file
data = np.load('data.npy')

# Exporting Data

# Exporting data to a text file
# Assuming 'data' is your NumPy array
np.savetxt('data.txt', data, delimiter=',')
# Exporting data to a CSV file
np.savetxt('data.csv', data, delimiter=',')
# Exporting data to a NumPy Binary file
np.save('data.npy', data)

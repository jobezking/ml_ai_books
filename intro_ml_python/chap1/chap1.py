import numpy as np 
x = np.array([[1, 2, 3], [4, 5, 6]]) 
print("x:\n{}".format(x))

from scipy import sparse
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("\nCOO representation:\n{}".format(eye_coo))

%matplotlib inline
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
y = np.cos(x)
plt.plot(x, y, marker='x')
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.title("A simple plot")
plt.show()

import pandas as pd
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Location': ['New York', 'Paris', 'Berlin', 'London'],
        'Age': [24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
print(data_pandas)
print("\nAge column:\n{}".format(data_pandas['Age']))
print("\nDescribe:\n{}".format(data_pandas.describe()))

import sys
print("\nPython version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))
print("SciPy version: {}".format(sparse.__version__))
print("Matplotlib version: {}".format(plt.__version__)) 

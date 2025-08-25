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

from sklearn.datasets import load_iris
iris_data = load_iris()
print("Keys of iris_data:\n{}".format(iris_data.keys()))
print(iris_data['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_data['target_names']))
print("Feature names:\n{}".format(iris_data['feature_names']))
print("Type of data: {}".format(type(iris_data['data'])))
print("Shape of data: {}".format(iris_data['data'].shape))
print("First five columns of data:\n{}".format(iris_data['data'][:5]))
print("Type of target: {}".format(type(iris_data['target'])))
print("Shape of target: {}".format(iris_data['target'].shape))
print("Target:\n{}".format(iris_data['target']))
print("First five targets:\n{}".format(iris_data['target'][:5]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_data['data'], iris_data['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))  

#create dataframe from data in X_train
#label the columns using the strings in iris_data.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_data.feature_names)
#create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
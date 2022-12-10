from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

# X, y = datasets.load_digits(return_X_y = True, as_frame = True)
# print(X.dtypes)

# print(data)

data = datasets.load_wine()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf = DecisionTree(max_depth = 10)
clf.fit(X_train, y_train)
predicitons = clf.predict(X_test)

def accuracy(y_test, y_pred):
	return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predicitons)
print(acc)

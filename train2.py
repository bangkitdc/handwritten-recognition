from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = datasets.load_wine()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

def accuracy(y_test, y_pred):
	return np.sum(y_test == y_pred) / len(y_test)

clf = RandomForest(n_trees = 20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# predictions2 = clf.predict(X_train)

acc = accuracy(y_test, predictions)
# acc2 = accuracy(y_train, predictions2)
print(acc)
# print(acc2)
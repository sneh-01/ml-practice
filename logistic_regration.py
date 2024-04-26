
# train a logistic regression classifier to predict whether a flower is iris viriginica or not
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()

X = iris["data"][:, 3:]
Y = (iris["target"]==2)
# Y = (iris["target"]==2).astype(np.int)


print(Y)
# print(X)
# x is feature and y is a label

# train a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,Y)
example = clf.predict(([[1.6]]))
print(example)

# using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1, 1)
print(X_new)
Y_prob = clf.predict_proba(X_new)
plt.plot(X_new , Y_prob [:,1], "g-" , label="virginica")
plt.show()

# a 0 thi 3 vachhe na 1000 points apse.


# print(list(iris.keys()))

# print(iris['data'])
# print(iris['DESCR'])

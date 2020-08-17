
## STUDENT ID: FILL IN YOUR ID
## STUDENT NAME: FILL IN YOUR NAME


## Question 2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)       # make sure you run this line for consistency 
x = np.random.uniform(1, 2, 100)
y = 1.2 + 2.9 * x + 1.8 * x**2 + np.random.normal(0, 0.9, 100)
# plt.scatter(x,y)
# plt.show()

## (c)

# YOUR CODE HERE 
losses_2c = np.zeros((9,100))
# moving provided code
alphas = [10e-1, 10e-2, 10e-3,10e-4,10e-5,10e-6,10e-7, 10e-8, 10e-9]

def question_2c(alphas):
    c_value_2c = 2
    for i in range(9):
        weights_2c = [1, 1]
        for j in range(100):
            loss = loss_total(c_value_2c, weights_2c, x, y)
            weights_2c[0] = weights_2c[0] - alphas[i]*descent_w0(c_value_2c, weights_2c,x, y)
            weights_2c[1] = weights_2c[1] - alphas[i]*descent_w1(c_value_2c, weights_2c,x, y)
            losses_2c[i][j] = loss

# assuming |x| = |y|
def loss_total(c, weights, x, y):
    sum_value = 0
    for i in range(len(x)):
        c_squared = (1)/(c**2)
        inside_bracket = (y[i] - weights[0] - weights[1]*x[i])
        sum_value += np.sqrt(c_squared*(inside_bracket**2)+ 1) - 1
    return sum_value

def descent_w0(c, w, x, y):
    sum_value = 0
    for i in range(len(x)):
        c_squared = (1)/(c**2)
        inside_bracket = (y[i] - w[0] - w[1]*x[i])
        sum_value += (-inside_bracket)/((c**2)*np.sqrt(c_squared*(inside_bracket**2)+1))
    return sum_value

def descent_w1(c, w, x, y):
    sum_value = 0
    for i in range(len(x)):
        c_squared = (1)/(c**2)
        inside_bracket = (y[i] - w[0] - w[1]*x[i])
        sum_value +=(-x[i]*(inside_bracket)/((c**2)*np.sqrt((c_squared*(inside_bracket**2) + 1))))
    return sum_value


# # uncomment for 2c, comment for 2e
# # plotting help
# fig, ax = plt.subplots(3,3, figsize=(10,10))
# question_2c(alphas)
# for i, ax in enumerate(ax.flat):
#     # losses is a list of 9 elements. Each element is an array of length 100 storing the loss at each iteration for that particular step size
#     ax.plot(losses_2c[i])         
#     ax.set_title(f"step size: {alphas[i]}")	 # plot titles	
# plt.tight_layout()      # plot formatting
# plt.savefig("grid_plot.png")
# plt.show()
#################################################################################################################################################

weight_plot = np.zeros((2,100))
losses_2e = np.zeros((1,100))
def quesiton_2e():
    c_value_2e = 2
    weights_2e = [1, 1]
    for i in range(100):
        weight_plot[0][i] = weights_2e[0]
        weight_plot[1][i] = weights_2e[1]
        loss = loss_total(c_value_2e, weights_2e, x, y)
        weights_2e[0] = weights_2e[0] - (0.001)*descent_w0(c_value_2e, weights_2e, x, y)
        weights_2e[1] = weights_2e[1] - (0.001)*descent_w1(c_value_2e, weights_2e, x, y)
        losses_2e[0][i] = loss
        
# uncomment for 2c, comment for 2e
# plotting help
fig, ax = plt.subplots(2,1, figsize=(10,10))
quesiton_2e()
print(weight_plot)
print("______")
print(losses_2e)
# plt.scatter(weight_plot[0],weight_plot[1])
# plt.set_title("w0 vs w1")
# plt.show()

for i, ax in enumerate(ax.flat):
    # losses is a list of 9 elements. Each element is an array of length 100 storing the loss at each iteration for that particular step size
    ax.plot(weight_plot[i])         
    ax.set_title(f"w: {i}")	 # plot titles	

# plt.tight_layout()      # plot formatting
# plt.savefig("grid_plot.png")
# plt.show()



## Question 3

# (c)
# YOUR CODE HERE





# Question 5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def create_dataset():
    X, y = make_classification( n_samples=1250,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2,
                                random_state=5,
                                n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size = X.shape)
    linearly_separable = (X, y)
    X = StandardScaler().fit_transform(X)
    return X, y


# (a)
# YOUR CODE HERE

def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                            np.arange(y_min, y_max, plot_step)) 
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)


# (b)
# YOUR CODE HERE

# (c)
# YOUR CODE HERE
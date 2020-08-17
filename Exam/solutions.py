
## STUDENT ID: z5075490
## STUDENT NAME: Abanob Tawfik


## Question 2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)       # make sure you run this line for consistency 
x = np.random.uniform(1, 2, 100)
y = 1.2 + 2.9 * x + 1.8 * x**2 + np.random.normal(0, 0.9, 100)
plt.scatter(x,y)
plt.show()

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


# uncomment for 2c, comment for 2e
# plotting help
fig, ax = plt.subplots(3,3, figsize=(10,10))
question_2c(alphas)
for i, ax in enumerate(ax.flat):
    # losses is a list of 9 elements. Each element is an array of length 100 storing the loss at each iteration for that particular step size
    ax.plot(losses_2c[i])         
    ax.set_title(f"step size: {alphas[i]}")	 # plot titles	
plt.tight_layout()      # plot formatting
plt.savefig("grid_plot.png")
plt.show()
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
        weights_2e[0] = weights_2e[0] - (0.065)*descent_w0(c_value_2e, weights_2e, x, y)
        weights_2e[1] = weights_2e[1] - (0.065)*descent_w1(c_value_2e, weights_2e, x, y)
        losses_2e[0][i] = loss
        
# uncomment for 2c, comment for 2e
# plotting help
fig, ax = plt.subplots(2,1, figsize=(10,10))
quesiton_2e()
for i, ax in enumerate(ax.flat):
    if(i == 0):
        # losses is a list of 9 elements. Each element is an array of length 100 storing the loss at each iteration for that particular step size
        ax.plot(weight_plot[0], label = 'w0')         
        ax.plot(weight_plot[1], label = 'w1')
        ax.legend()
        ax.set_title(f"w0 vs w1 over 100 iterations")	 # plot titles	
    else:
        ax.scatter(x,y, label = 'data')
        model = weight_plot[0][99] + weight_plot[1][99]*x
        ax.plot(x, model, label = 'model')
        ax.legend()
        ax.set_title(f"model")	 # plot titles	


plt.tight_layout()      # plot formatting
plt.savefig("grid_plot.png")
plt.show()



## Question 3

# (c)
# YOUR CODE HERE
# note this is on the original feature space as i could not get (a) to work or figure out the numpy code for it to work
# this will not converge as the space is not linearly seperable, however the general algorithm is still there, and how i print the table/ compute weights
# is reflective of how i would of done it if i were able to figure out (a)
def train_perceptron():
    weights_3c = [1, 1]
    learning_rate_3c = 0.2
    epochs = 0
    converged = False
    x1s_3c = [-0.8, 3.9, 1.4, 0.1, 1.2, -2.45, -1.5, 1.2]
    x2s_3c = [1, 0.4, 1, -3.3, 2.7, 0.1, -0.5, -1.5]
    ys_3c = [1, -1, 1, -1, -1, -1, 1,1]

    print("Iteration No. | ", " w0                          |", " w1                          | ")
    print("---------------------------------------------------------------------------------------")
    change_in_weight_0 = 0
    change_in_weight_1 = 0
    while(converged == False and epochs < 100):
        print(epochs, " | ", weights_3c[0], " + ",change_in_weight_0, "                         | " ,weights_3c[1], " + ",change_in_weight_1)
        change_in_weight_0 = 0
        change_in_weight_1 = 0
        converged = True
        for i in range(len(x1s_3c)):
            if ((weights_3c[0]*x1s_3c[i] + weights_3c[1]*x2s_3c[i]) * ys_3c[i]) < 0:

                change_in_weight_0 += learning_rate_3c*ys_3c[i]*(x1s_3c[i]) 
                change_in_weight_1 += learning_rate_3c*ys_3c[i]*(x2s_3c[i]) 

                weights_3c[0] = weights_3c[0] + learning_rate_3c*ys_3c[i]*(x1s_3c[i])
                weights_3c[1] = weights_3c[1] + learning_rate_3c*ys_3c[i]*(x2s_3c[i])
                converged = False
        epochs += 1

    print("---------------------------------------------------------------------------------------")
train_perceptron()


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
#imported to compute accuracy (was okay'd on the forum to use this)
from sklearn.metrics import accuracy_score

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

x, y = create_dataset()
# default of train_test_split will do randomly
# instantiate our classifiers, put them in a list, and for each plot em ez pz
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

SVCModel = SVC()
SVCModel.fit(x_train, y_train)
LogisticModel = LogisticRegression()
LogisticModel.fit(x_train, y_train)
AdaBoostModel = AdaBoostClassifier()
AdaBoostModel.fit(x_train, y_train)
RandomForestModel = RandomForestClassifier()
RandomForestModel.fit(x_train, y_train)
DecisionTreeModel = DecisionTreeClassifier()
DecisionTreeModel.fit(x_train, y_train)
MLPCModel = MLPClassifier()
MLPCModel.fit(x_train, y_train)

all_models = [SVCModel, LogisticModel, AdaBoostModel, RandomForestModel, DecisionTreeModel, MLPCModel]


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
        ax.show()
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)
        plt.show()


for model in (all_models):
    plotter(model, x, x_test, y_test, type(model).__name__)

# (b)
# YOUR CODE HERE
test_sample_size = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
models_accuracy = []
models_timing = []

def get_color_of_model(name):
    if(name == "SVC"):
        return "brown"
    elif(name == "LogisticRegression"):
        return "red"
    elif(name == "AdaBoostClassifier"):
        return "green"
    elif(name == "RandomForestClassifier"):
        return "orange"
    elif(name == "DecisionTreeClassifier"):
        return "blue"
    elif(name == "MLPClassifier"):
        return "palevioletred"
    else:
        return "default"

# 10 iterations
for sample_size in (test_sample_size):
    for model in (all_models):
        average = 0
        time_elapsed_avg = 0
        for i in range(10):
            start = time.time()
            # split our sample size based on number of values
            # by default split data shuffle so randomised
            x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=sample_size)
            model.fit(x_train1, y_train1)
            results = model.predict(x_test1)
            # im using sklearn to find accuracy, approved by anant mathur on forums
            # will give percentage of our accuracy not float! nicer for plotting
            accuracy = accuracy_score(results, y_test1)*100
            end = time.time()
            average += accuracy/(10)
            time_elapsed_avg +=  abs((end-start)/(10))
        models_accuracy.append((type(model).__name__, average, sample_size))
        models_timing.append((type(model).__name__, time_elapsed_avg, sample_size))

for model in (all_models):    
    x = []
    y = []
    filtered_model_tuple = [tupl for tupl in models_accuracy if (tupl[0] == (type(model).__name__))]
    for value in (filtered_model_tuple):
        x.append(value[2])
        y.append(value[1])
    plt.plot(x, y, label = (type(model).__name__), color = get_color_of_model(type(model).__name__))

plt.legend()
plt.show()

# (c)
# YOUR CODE HERE

# see additions to (b) for time code, added avg time in
for model in (all_models):    
    x = []
    y = []
    filtered_model_tuple = [tupl for tupl in models_timing if (tupl[0] == (type(model).__name__))]
    for value in (filtered_model_tuple):
        x.append(value[2])
        y.append(value[1])
    plt.plot(x, y, label = (type(model).__name__), color = get_color_of_model(type(model).__name__))

plt.legend()
plt.show()
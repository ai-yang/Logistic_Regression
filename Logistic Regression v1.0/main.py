import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset


# load data from datasets
train_set_x_orig, train_y, test_set_x_orig, test_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
train_x = train_set_x_orig.reshape(m_train, -1).T
test_x = test_set_x_orig.reshape(m_test, -1).T
train_x = train_x / 255
test_x = test_x / 255


def sigmod(x):
    k = 1.0/(1.0 + np.exp(-1.0 * x))
    return k


def initial(dim):     # initialize variable w and b
    w = np.random.rand(dim, 1)
    b = np.random.rand(1)
    return w, b


def propagate(w, b, X, Y):   # forward propagation and compute dw, db
    m = X.shape[1]
    A = sigmod(np.dot(w.T, X) + b)
    cost = -(1.0/m) * np.sum(Y * np.log(A) + (1 - Y)*np.log(1 - A))
    dw = (1.0/m) * np.dot(X, (A - Y).T)
    db = (1.0/m) * np.sum(A - Y)
    return dw, db, cost


def optimize(w, b, X, Y, iteration, learning_rate):    # gradient descent
    cost1 = []
    for i in range(iteration):
        dw, db, cost = propagate(w, b, X, Y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 20 == 0:
            cost1.append(cost)
            print(cost)
    params = {"w": w, "b": b}
    return params, cost1


def predict(w, b, X):     # do test for parameters after optimization
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmod(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def make(s, learning_rate):     # print plot
    cost = np.squeeze(s)
    plt.plot(cost)
    plt.ylabel('cost')
    plt.xlabel('iterations(per 20)')
    plt.title("Learning rate="+str(learning_rate))
    plt.show()


# the whole model of logistic regression
def model(x_train, y_train, x_test, y_test, iteration=2000, learning_rate=0.5, makeplt=1):
    w, b = initial(x_train.shape[0])
    parameters, cost = optimize(w, b, x_train, y_train, iteration, learning_rate)
    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, x_test)
    Y_prediction_train = predict(w, b, x_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))

    d = {"Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": iteration}
    if makeplt:
        make(cost, learning_rate)
    return d


d = model(train_x, train_y, test_x, test_y, iteration=3000, learning_rate=0.01, makeplt=1)
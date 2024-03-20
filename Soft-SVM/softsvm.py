import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m = trainX.shape[0]
    d = trainX.shape[1]

    H = matrix(
        np.vstack((np.hstack((2 * l * np.eye(d), np.zeros((d, m)))), np.hstack((np.zeros((m, d)), np.zeros((m, m)))))))
    u = matrix(np.vstack((np.zeros((d, 1)), (1 / m) * np.ones((m, 1)))))
    A = matrix(np.vstack((np.hstack((np.diag(trainy) @ trainX, np.eye(m))), np.hstack((np.zeros((m, d)), np.eye(m))))))
    v = matrix(np.vstack((np.ones((m, 1)), np.zeros((m, 1)))))

    sol = solvers.qp(H, u, -A, -v)
    w = np.array(sol['x'][:d])
    return w


def load_data():
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    return trainX, testX, trainy, testy


def simple_test():
    # load question 2 data
    trainX, testX, trainy, testy = load_data()

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def run_softsvm(trainX, testX, trainy, testy, m, d, lambdas, repeats):
    train_errors = {l: [] for l in lambdas}
    min_max_avg_train_errors = {}

    test_errors = {l: [] for l in lambdas}
    min_max_avg_test_errors = {}

    for l in lambdas:
        for _ in range(repeats):
            indices = np.random.permutation(trainX.shape[0])
            _trainX = trainX[indices[:m]]
            _trainy = trainy[indices[:m]]

            w = softsvm(l, _trainX, _trainy)

            assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
            assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

            train_errors[l].append(
                np.sum(np.sign(_trainX @ w).astype(int) != _trainy.reshape(-1, 1)) / _trainy.shape[0])
            test_errors[l].append(np.sum(np.sign(testX @ w).astype(int) != testy.reshape(-1, 1)) / testy.shape[0])

        min_max_avg_train_errors[l] = (np.min(train_errors[l]), np.max(train_errors[l]), np.mean(train_errors[l]))
        min_max_avg_test_errors[l] = (np.min(test_errors[l]), np.max(test_errors[l]), np.mean(test_errors[l]))

    if m == 100:
        plot_small_sample(lambdas, train_errors, min_max_avg_train_errors, "Train error: m = 100", "blue")
        plot_small_sample(lambdas, test_errors, min_max_avg_test_errors, "Test error: m = 100", "red")
    else:
        plot_large_sample(lambdas, train_errors, test_errors)


def mnist_softsvm():
    # load question 2 data
    trainX, testX, trainy, testy = load_data()

    d = trainX.shape[1]

    # small sample
    # lambdas = [10 ** n for n in range(1, 11)]
    # run_softsvm(trainX, testX, trainy, testy, 100, d, lambdas, 10)

    # large sample
    lambdas = [10 ** n for n in [1, 3, 5, 8]]
    run_softsvm(trainX, testX, trainy, testy, 1000, d, lambdas, 1)

    plt.title("Soft SVM")
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def plot_small_sample(lambdas, errors, min_max_avg, label, line_color):
    min_errors = [l_err[0] for l_err in min_max_avg.values()]
    max_errors = [l_err[1] for l_err in min_max_avg.values()]
    avg_errors = [l_err[2] for l_err in min_max_avg.values()]
    min_distance = [a - b for a, b in zip(avg_errors, min_errors)]
    max_distance = [b - a for a, b in zip(avg_errors, max_errors)]
    error_bar = np.array([min_distance, max_distance])
    plt.plot(lambdas, avg_errors, color=line_color, marker="o", label=label)
    plt.errorbar(lambdas, avg_errors, yerr=error_bar, fmt="none")


def plot_large_sample(lambdas, train_errors, test_errors):
    plt.scatter(lambdas, train_errors.values(), color='orange', label="Train error: m = 1000")
    plt.scatter(lambdas, test_errors.values(), color='purple', label="Test error: m = 1000")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    mnist_softsvm()
    # here you may add any code that uses the above functions to solve question 2

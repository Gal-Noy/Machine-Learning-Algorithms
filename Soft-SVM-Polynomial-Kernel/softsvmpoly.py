import matplotlib
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def fix_negative_eigenvalues(H):
    """
    This function takes a matrix H and makes sure that it is positive semi-definite
    """
    epsilon = 1e-5
    while min(np.linalg.eigvals(H)) <= 0:
        H = H + epsilon * np.eye(H.shape[0])
    return H


def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = trainX.shape[0]

    G = np.fromfunction(np.vectorize(lambda i, j: (1 + trainX[i] @ trainX[j]) ** k), (m, m), dtype=int)
    H = matrix(np.vstack((np.hstack((2 * l * G, np.zeros((m, m)))), np.hstack((np.zeros((m, m)), np.zeros((m, m))))))
               + 1e-5 * np.eye(2 * m))  # make sure H is positive semi-definite
    u = matrix(np.vstack((np.zeros((m, 1)), (1 / m) * np.ones((m, 1)))))
    A = matrix(np.vstack((np.hstack((np.diag(trainy) @ G, np.eye(m))), np.hstack((np.zeros((m, m)), np.eye(m))))))
    v = matrix(np.vstack((np.ones((m, 1)), np.zeros((m, 1)))))

    sol = solvers.qp(H, u, -A, -v)
    alpha = np.array(sol['x'][:m])

    return alpha


def load_data():
    data = np.load('EX3q2_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    return trainX, testX, trainy, testy


def simple_test():
    # load question 2 data
    trainX, testX, trainy, testy = load_data()

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def plot_kernel_soft_svm():
    trainX, testX, trainy, testy = load_data()

    m = 100

    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    w = softsvmpoly(10, 5, _trainX, _trainy)

    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

    # plot the decision boundary
    positive_points = trainX[trainy.squeeze() == 1]
    negative_points = trainX[trainy.squeeze() == -1]

    plt.scatter(positive_points[:, 0], positive_points[:, 1], c='b', label='positive')
    plt.scatter(negative_points[:, 0], negative_points[:, 1], c='r', label='negative')

    plt.title('Polynomial Kernel Soft SVM')
    plt.legend()
    plt.show()


def calculate_error(alpha, k, trainX, testX, testy):
    y_preds = []
    for x in testX:
        y_pred = 0
        for i in range(trainX.shape[0]):
            y_pred += alpha[i] * (1 + x @ trainX[i]) ** k
        y_preds.append(np.sign(y_pred)[0])
    return np.mean(y_preds != testy)


def cross_validation(folds):
    trainX, testX, trainy, testy = load_data()

    m = 100

    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    trainX_sets = np.array_split(_trainX, folds)
    trainy_sets = np.array_split(_trainy, folds)

    l_k_options = [(l, k) for l in [1, 10, 100] for k in [2, 5, 8]]
    avg_errors = {(l, k): 0 for l, k in l_k_options}

    for l, k in l_k_options:
        errors = []
        for i in range(folds):
            trainX = np.concatenate([trainX_sets[j] for j in range(folds) if j != i])
            trainy = np.concatenate([trainy_sets[j] for j in range(folds) if j != i])
            testX = trainX_sets[i]
            testy = trainy_sets[i]

            alpha = softsvmpoly(l, k, trainX, trainy)

            errors.append(calculate_error(alpha, k, trainX, testX, testy))
        avg_errors[(l, k)] = np.mean(errors)

    for l, k in l_k_options:
        print(f'lambda: {l}, k: {k}, avg error: {avg_errors[(l, k)]}')
    best_params = min(avg_errors, key=avg_errors.get)
    print(f'Best parameters: {best_params} with average error: {avg_errors[best_params]}')


def plot_grid():
    trainX, testX, trainy, testy = load_data()

    m = 100

    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    l = 100
    for k in [3, 5, 8]:
        alpha = softsvmpoly(l, k, _trainX, _trainy)

        x_min, x_max = np.min(_trainX[:, 0]), np.max(_trainX[:, 0])
        y_min, y_max = np.min(_trainX[:, 1]), np.max(_trainX[:, 1])
        print(x_min, x_max, y_min, y_max)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, m), np.linspace(y_min, y_max, m))

        # color the grid points red or blue depending on the label predicted by the classifier for each point
        Z = np.zeros(xx.shape)
        print(xx.shape, yy.shape, Z.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = np.array([xx[i, j], yy[i, j]])
                y_pred = calculate_error(alpha, k, _trainX, np.array([point]), np.array([1]))
                Z[i, j] = y_pred

        plt.imshow(Z, cmap=ListedColormap(['blue', 'red']), extent=[x_min, x_max, y_min, y_max])
        plt.title(f'Polynomial Kernel Soft SVM with lambda={l} and k={k}')
        plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # plot_kernel_soft_svm()
    # cross_validation(5)
    plot_grid()

    # here you may add any code that uses the above functions to solve question 4

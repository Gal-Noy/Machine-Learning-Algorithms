import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


class KNNClassifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        distances = np.array([[distance.euclidean(x1, x2) for x1 in self.x_train] for x2 in x_test])
        knn_indices = np.argsort(distances)[:, :self.k]
        knn_labels = np.array([self.y_train[knn_index] for knn_index in knn_indices])
        y_test_prediction = np.array([np.bincount(label.astype(int)).argmax() for label in knn_labels])
        return y_test_prediction.reshape(-1, 1)


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = KNNClassifier(k, x_train, y_train)
    return classifier


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    return classifier.predict(x_test)


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def load_data():
    data = np.load('mnist_all.npz')

    train2 = data['train2']
    train3 = data['train3']
    train5 = data['train5']
    train6 = data['train6']

    test2 = data['test2']
    test3 = data['test3']
    test5 = data['test5']
    test6 = data['test6']

    return train2, train3, train5, train6, test2, test3, test5, test6


def mnist_1nn():
    train2, train3, train5, train6, test2, test3, test5, test6 = load_data()

    sample_sizes = [_ for _ in range(10, 101, 10)]

    errors = {size: [] for size in sample_sizes}
    max_errors = {size: 0 for size in sample_sizes}
    min_errors = {size: 1 for size in sample_sizes}
    avg_errors = {size: 0 for size in sample_sizes}

    for size in sample_sizes:
        for _ in range(10):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], size)
            x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], 50)

            classifer = learnknn(1, x_train, y_train)
            preds = predictknn(classifer, x_test)

            errors[size].append(np.mean(np.vstack(y_test) != np.vstack(preds)))

        max_errors[size] = max(errors[size])
        min_errors[size] = min(errors[size])
        avg_errors[size] = np.mean(errors[size])

        print(
            f"Sample size: {size}, avg error: {avg_errors[size]}, max error: {max_errors[size]}, min error: {min_errors[size]}")

    display_plot(sample_sizes, avg_errors.values(), max_errors.values(), min_errors.values(),
                 'Average error of 1-NN on MNIST dataset', 'Sample size', 'Error')


def mnist_knn():
    train2, train3, train5, train6, test2, test3, test5, test6 = load_data()

    k_values = [_ for _ in range(1, 11)]

    errors = {k: [] for k in k_values}
    max_errors = {k: 0 for k in k_values}
    min_errors = {k: 1 for k in k_values}
    avg_errors = {k: 0 for k in k_values}

    for k in k_values:
        for _ in range(10):
            x_train, y_train = gensmallm([train2, train3, train5, train6], [2, 3, 5, 6], 200)
            x_test, y_test = gensmallm([test2, test3, test5, test6], [2, 3, 5, 6], 50)

            classifer = learnknn(k, x_train, y_train)
            preds = predictknn(classifer, x_test)

            errors[k].append(np.mean(np.vstack(y_test) != np.vstack(preds)))

        max_errors[k] = max(errors[k])
        min_errors[k] = min(errors[k])
        avg_errors[k] = np.mean(errors[k])

        print(f"k: {k}, avg error: {avg_errors[k]}, max error: {max_errors[k]}, min error: {min_errors[k]}")

    display_plot(k_values, avg_errors.values(), max_errors.values(), min_errors.values(),
                 'Average error of k-NN on MNIST dataset', 'k', 'Error')


def display_plot(x, y, max_errors, min_errors, title, xlabel, ylabel):
    plt.plot(x, y, marker='o', color='b')
    plt.plot(x, max_errors, marker='o', color='r', linestyle='None')
    plt.plot(x, min_errors, marker='o', color='g', linestyle='None')
    plt.xticks(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # simple_test()
    # mnist_1nn()
    mnist_knn()

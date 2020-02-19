import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



figure_num = 100


def normal_distribution():
    mean1 = [-2, -2, -2]
    cov1 = [[2, 1, 0.2], [1, 4, 1], [0.2, 1, 2]]
    n1 = 100
    mean2 = [4, 4, 4]
    cov2 = [[4, 1, 0.2], [1, 2, 0.2], [0.2, 0.2, 2]]
    n2 = 200
    x1, y1, z1 = np.random.multivariate_normal(mean1, cov1, n1).T
    x2, y2, z2 = np.random.multivariate_normal(mean2, cov2, n2).T
    class1 = np.array(list(zip(x1, y1, z1)))
    class2 = np.array(list(zip(x2, y2, z2)))
    labels1 = [0]*n1
    labels2 = [1]*n2
    labels = np.concatenate((labels1, labels2))
    X = np.concatenate((class1, class2))
    Xtrain, Xtest, Ytrain, Ytest = shuffle_and_split_data(X, labels)
    predicted_labels = perform_knn_and_calculate_score(Xtrain, Ytrain, Xtest, Ytest)
    plot_test_and_predicted(Xtest, Ytest, predicted_labels)


def test_data():
    x, y, z, classes = read_data_from_file()
    num_classes = string_labels_to_num(classes)
    axis_labels = ['Protein', 'Oil', 'Size']
    X = np.array(list(zip(x, y, z)))
    Xtrain, Xtest, Ytrain, Ytest = shuffle_and_split_data(X, num_classes)
    predicted_labels = perform_knn_and_calculate_score(Xtrain, Ytrain, Xtest, Ytest)
    plot_test_and_predicted(Xtest, Ytest, predicted_labels, axis_labels)


def shuffle_and_split_data(X, labels):
    data, annotations = shuffle(X, labels)
    return train_test_split(data, annotations, test_size=0.5)


def string_labels_to_num(classes):
    unique_labels = list(set(classes))
    num_labels = []
    for i in range(len(classes)):
        num_labels.append(unique_labels.index(classes[i]))
    return np.array(num_labels)


def perform_knn_and_calculate_score(x_train, y_train, x_test, y_test):
    nbrs = KNeighborsClassifier(n_neighbors=1)
    nbrs.fit(x_train, y_train)
    predicted_labels = nbrs.predict(x_test)
    print(nbrs.score(x_test, y_test))
    return predicted_labels


def plot_test_and_predicted(x_test, y_test, predicted_labels, axis_labels=None):
    axis_labels = ['X', 'Y', 'Z'] if axis_labels is None else axis_labels
    plot_single_figure(x_test, y_test, axis_labels, 'Real Classes')
    plot_single_figure(x_test, predicted_labels, axis_labels, 'Prediction')
    plt.show()


def plot_single_figure(x, y, labels, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)


def read_data_from_file():
    file = open("01-soybean-data.txt")
    protein = []
    oil = []
    loc = []
    size = []
    for line in file:
        splitted_line = line.split()
        protein.append(float(splitted_line[9]))
        oil.append(float(splitted_line[10]))
        loc.append(splitted_line[2])
        size.append(float(splitted_line[8]))
    file.close()
    return np.array(protein), np.array(oil), np.array(size), np.array(loc)


if __name__ == '__main__':
    # normal_distribution()
    test_data()

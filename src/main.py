import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist

data = loadmat('../ORL_32x32.mat')
train = loadmat('../train_data/7.mat')

people_index = data['gnd']
faces_db = data['fea'] / 255


class NN:
    def __init__(self):
        self.mean_face, self.eigenfaces, self.face_space_coord_train = None, None, None

    def train(self, train_faces, train_labels, k_componenents):
        self.train_labels = train_labels
        self.mean_face = np.mean(train_faces, axis=0)
        covariance_matrix = np.cov(train_faces.T)

        # eigenfaces = k principal components (eigenvectors of the covariance matrix)
        _, eigenvalues, eigenvectors = np.linalg.svd(covariance_matrix, full_matrices=False)
        self.eigenfaces = eigenvectors[:k_componenents]

        self.face_space_coord_train = project_onto_eigenface_subspace(self.eigenfaces, train_faces, self.mean_face)

        # for eigenface in self.eigenfaces:
        #     plt.imshow((eigenface.reshape(32, 32) + np.min(eigenface)) / np.max(eigenface), cmap="gray")
        #     plt.show()

    def test(self, test_faces):
        face_space_coord_test = project_onto_eigenface_subspace(self.eigenfaces, test_faces, self.mean_face)

        recognized_labels = []
        for face in face_space_coord_test:
            distances = cdist(face.reshape(1, -1), self.face_space_coord_train)[0, :]
            original_label = self.train_labels[np.argsort(distances)[0]]
            recognized_labels.append(original_label)

        return np.array(recognized_labels)


def project_onto_eigenface_subspace(eigenfaces, faces, mean_face):
    """ Project each training image onto the subspace spanned by principal components """
    return np.dot(eigenfaces, np.array(faces - mean_face).T).T


def load_images(idx):
    indexes = loadmat('../train_data/' + str(idx) + '.mat')
    data = loadmat('../ORL_32x32.mat')
    faces_db = data['fea'] / 255

    train_images = faces_db[indexes['trainIdx'] - 1]
    train_images = train_images.reshape(train_images.shape[0], train_images.shape[2])
    train_labels = data['gnd'][indexes['trainIdx'] - 1][:, 0, 0]
    test_images = faces_db[indexes['testIdx'] - 1]
    test_images = test_images.reshape(test_images.shape[0], test_images.shape[2])
    test_labels = data['gnd'][indexes['testIdx'] - 1][:, 0, 0]

    return train_images, train_labels, test_images, test_labels


def calc_accuracy(test_labels, nn_labels):
    """ Calculate the accuracy of the network """
    return 1 - sum(test_labels != nn_labels) / len(test_labels)


def study_network():
    n_train_images = [3, 5, 7]
    k_ppal_components = [i * 10 + 10 for i in range(15)]
    nearest_neighbor = NN()

    for n_train in n_train_images:
        train_img, train_labels, test_img, test_labels = load_images(n_train)
        accuracies = []
        for k_components in k_ppal_components:
            nearest_neighbor.train(train_img, train_labels, k_components)
            recognized_labels = nearest_neighbor.test(test_img)
            accuracies.append(calc_accuracy(test_labels, recognized_labels) * 100)

        plt.plot(k_ppal_components, accuracies, label=str(n_train) + " training imgs")

    plt.title("Accuracy comparison")
    plt.xlabel("Number of eigenfaces")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


def study_eigenfaces():
    train_img, train_labels, test_img, test_labels = load_images(3)
    k_ppal_components = 50
    nearest_neighbor = NN()
    nearest_neighbor.train(train_img, train_labels, k_ppal_components)

    sqrt = math.sqrt(k_ppal_components)
    rows = sqrt if sqrt == int(sqrt) else int(sqrt) + 1
    i = 0
    for eigenface in nearest_neighbor.eigenfaces:
        i += 1
        if i > sqrt * int(sqrt):
            break
        plt.subplot(int(sqrt), rows, i)
        plt.imshow((eigenface.reshape(32, 32).T + np.min(eigenface)) / np.max(eigenface), cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.title("Eigenfaces used")
    plt.show()


study_eigenfaces()
# study_network()

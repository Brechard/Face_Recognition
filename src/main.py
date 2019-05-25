import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from nn import NN

data = loadmat('../ORL_32x32.mat')
train = loadmat('../train_data/7.mat')

people_index = data['gnd']
faces_db = data['fea'] / 255


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


def study_accuracy():
    n_train_images = [3, 5, 7]
    k_ppal_components = [i * 10 + 10 for i in range(15)]
    nearest_neighbor = NN()

    fig, axarr = plt.subplots(1, 2)

    for n_train in n_train_images:
        train_img, train_labels, test_img, test_labels = load_images(n_train)
        accuracies = []
        reconstruction_errors = []
        for k_components in k_ppal_components:
            nearest_neighbor.train(train_img, train_labels, k_components)
            recognized_labels = nearest_neighbor.test(test_img)
            reconstruction_error = nearest_neighbor.get_reconstruction_error(test_img)

            accuracies.append(calc_accuracy(test_labels, recognized_labels) * 100)
            reconstruction_errors.append(reconstruction_error)

        axarr[0].plot(k_ppal_components, accuracies, label=str(n_train) + " training images")
        axarr[1].plot(k_ppal_components, reconstruction_errors, label=str(n_train) + " training images")

    axarr[0].set_title("Accuracy comparison")
    axarr[0].set(xlabel='Number of eigenfaces', ylabel='Accuracy (%)')
    axarr[0].legend()
    axarr[1].set_title("Reconstruction error comparison")
    axarr[1].set(xlabel='Number of eigenfaces', ylabel='Reconstruction error')
    axarr[1].legend()
    plt.show()


def study_eigenfaces(n_training_img, k_ppal_components):
    train_img, train_labels, test_img, test_labels = load_images(n_training_img)
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
        plt.imshow(shape_image(eigenface), cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(f'Eigenvectors. {n_training_img} training images, {k_ppal_components} eigenfaces')

    # plt.title("Eigenfaces used")
    plt.show()


def shape_image(eigenface):
    return (eigenface.reshape(32, 32).T + np.min(eigenface)) / np.max(eigenface)


def show_reconstructed(n_training_img, k_ppal_components, n_images):
    """ Show side by side the original image and the reconstruction """
    train_img, train_labels, test_img, test_labels = load_images(n_training_img)
    nearest_neighbor = NN()
    nearest_neighbor.train(train_img, train_labels, k_ppal_components)

    reconstructed = nearest_neighbor.get_reconstructed_faces(train_img)

    fig, axarr = plt.subplots(n_images, 2)
    axarr[0, 0].set_title("Original")
    axarr[0, 1].set_title("Reconstructed")
    for i in range(n_images):
        axarr[i, 0].imshow(shape_image(train_img[i]), cmap="gray")
        axarr[i, 1].imshow(shape_image(reconstructed[i]), cmap="gray")

    # Remove the labels in the plots
    plt.setp([a.get_xticklabels() for a in axarr[:, 0]], visible=False)
    plt.setp([a.get_xticklabels() for a in axarr[:, 1]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    fig.suptitle(f'Reconstruction analysis\n{n_training_img} training images, {k_ppal_components} eigenfaces')
    plt.show()


def show_reconstruction(n_training_img=5):
    train_img, train_labels, test_img, test_labels = load_images(n_training_img)
    nearest_neighbor = NN()
    nearest_neighbor.train(train_img, train_labels, 10)
    eigenfaces = np.zeros(nearest_neighbor.eigenfaces.shape)
    for i in range(nearest_neighbor.eigenfaces.shape[1]):
        eigenfaces[:, i] = nearest_neighbor.face_space_coord_train[0] * nearest_neighbor.eigenfaces[:, i]

    plt.imshow(shape_image(nearest_neighbor.mean_face), cmap="gray")
    plt.title("Mean face")
    plt.figure()
    for i, eigenface in enumerate(eigenfaces):
        plt.subplot(1, eigenfaces.shape[0], i + 1)
        plt.imshow(shape_image(eigenface), cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle("Eigenfaces")
    plt.show()

# show_reconstruction()
# show_reconstructed(3, 40, 5)
# study_eigenfaces(3, 50)
study_accuracy()

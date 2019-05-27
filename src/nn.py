import numpy as np
from scipy.spatial.distance import cdist


class NN:
    def __init__(self):
        self.mean_face, self.eigenfaces, self.face_space_coord_train, self.train_labels = None, None, None, None

    def train(self, train_faces, train_labels, k_componenents):
        """ Train the NN, creating the face space coordinates """
        self.train_labels = train_labels
        self.mean_face = np.mean(train_faces, axis=0)
        covariance_matrix = np.cov(train_faces.T)

        # eigenfaces = k principal components (eigenvectors of the covariance matrix)
        _, eigenvalues, eigenvectors = np.linalg.svd(covariance_matrix, full_matrices=False)
        self.eigenfaces = eigenvectors[:k_componenents]

        self.face_space_coord_train = self.project_onto_eigenface_subspace(train_faces)

    def test(self, test_faces):
        """ Test the NN returning the labels of the input faces """
        face_space_coord_test = self.project_onto_eigenface_subspace(test_faces)

        recognized_labels = []
        for face in face_space_coord_test:
            distances = cdist(face.reshape(1, -1), self.face_space_coord_train)[0, :]
            original_label = self.train_labels[np.argsort(distances)[0]]
            recognized_labels.append(original_label)

        return np.array(recognized_labels)

    def get_reconstruction_error(self, faces):
        """ Return the reconstruction error as the sum of errors of all the pixels of the faces sent """
        return np.sum(abs(self.get_reconstructed_faces(faces) - faces))

    def get_reconstructed_faces(self, faces):
        """ Return the reconstruction of the faces sent as input """
        return self.mean_face + np.dot(self.project_onto_eigenface_subspace(faces), self.eigenfaces)

    def project_onto_eigenface_subspace(self, faces):
        """ Project each training image onto the subspace spanned by principal components """
        return np.dot(self.eigenfaces, np.array(faces - self.mean_face).T).T

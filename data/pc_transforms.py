import argparse
import numpy as np
import trimesh.transformations as tf

from typing import List, Optional


class PointcloudTransforms(object):
    def __init__(self, opts: argparse.Namespace, seed: Optional[int] = None):
        self.rotate_opts = [opts.rotate_x, opts.rotate_y, opts.rotate_z]
        self.translate_opts = [opts.translate_x, opts.translate_y, opts.translate_z]
        self.rng = np.random if seed is None else np.random.RandomState(seed)

    def get_rotation_transform(self):
        rand_rotation = 2 * np.pi * self.rng.rand(3)
        alpha, beta, gamma = [x if y is None else 0.0 for x, y in zip(rand_rotation, self.rotate_opts)]

        origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        homogeneous_rotation_transform = tf.concatenate_matrices(
            tf.rotation_matrix(alpha, xaxis), tf.rotation_matrix(beta, yaxis), tf.rotation_matrix(gamma, zaxis)
        )  # in homogeneous coordinates
        cartesian_rotation_transform = homogeneous_rotation_transform[:3, :3]

        return cartesian_rotation_transform

    def apply_rotation(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies random rotation over the arrays
        where each array undergoes the same rotation!

        :param arrays: List of arrays to rotate. arrays are arranged in shape (n_points,3).
        :return: List of rotated arrays of the same shape as input arrays.
        """
        cartesian_rotation_transform = self.get_rotation_transform()

        rotated_data = []
        for array in arrays:
            rotated_data.append(np.matmul(cartesian_rotation_transform, array.T).T)

        return rotated_data

    def get_translation_vector(self, step_size: Optional[float] = 1.0):
        rand_rotation = self.rng.rand(3)
        alpha, beta, gamma = [x if y is None else 0.0 for x, y in zip(rand_rotation, self.translate_opts)]

        vector = np.asarray([alpha, beta, gamma]).reshape((1, 3))
        unit_vector = vector / np.linalg.norm(vector)
        translation_vector = step_size * unit_vector

        return translation_vector

    def apply_translation(self, arrays: List[np.ndarray], step_size: Optional[float] = 1.0) -> List[np.ndarray]:
        """
        Applies random translation over the arrays
        where each array undergoes the same translation!

        :param arrays: List of arrays to translate. arrays are arranged in shape (n_points,3).
        :param step_size: Translation step size. Translation = step_size * random_translation.
        :return: List of translated arrays of the same shape as input arrays.
        """
        translation_vector = self.get_translation_vector(step_size)

        translated_data = []
        for array in arrays:
            translated_data.append(array + translation_vector)

        return translated_data

# -*- coding: utf-8 -*-

import unittest

import numpy as np
from rirpy.model import propagate_signal, validate_geometry


class TestModel(unittest.TestCase):
    def test_propagate_signal(self):
        # Define test parameters
        source_signal = np.array([1.0, 0.5, 0.25, 0.0, 0.0], dtype=np.float64)
        sampling_rate = 1000.0  # Hz
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0  # Tank dimensions in meters
        sound_speed = 1500.0  # m/s
        refl_coeff_wall = 0.8
        refl_coeff_ceil = 0.9
        cutoff_time = 0.01  # seconds
        num_threads = 2

        # Call the function
        result = propagate_signal(
            source_signal,
            sampling_rate,
            source_position,
            receiver_position,
            length_x,
            length_y,
            length_z,
            sound_speed,
            refl_coeff_wall,
            refl_coeff_ceil,
            cutoff_time,
            num_threads,
        )

        # Check the output shape
        self.assertEqual(result.shape, source_signal.shape)

        # Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

    def test_validate_geometry_valid(self):
        # Valid geometry
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0

        # Should not raise an exception
        try:
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )
        except ValueError:
            self.fail("validate_geometry raised ValueError unexpectedly!")

    def test_validate_geometry_invalid_source(self):
        # Invalid source position
        source_position = np.array([6.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0

        with self.assertRaises(ValueError):
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )

    def test_validate_geometry_invalid_receiver(self):
        # Invalid receiver position
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 6.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0

        with self.assertRaises(ValueError):
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )

    def test_validate_geometry_invalid_dimensions(self):
        # Invalid tank dimensions
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = -5.0, 5.0, 5.0

        with self.assertRaises(ValueError):
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )


if __name__ == "__main__":
    unittest.main()

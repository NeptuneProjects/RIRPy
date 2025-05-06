# -*- coding: utf-8 -*-

import unittest

import numpy as np
from rirpy.model import propagate_signal, validate_geometry, apply_time_delay


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

    def test_1d_with_positive_shift(self):
        """Test 1D signal with a positive shift."""
        signal = np.array([1, 2, 3, 4, 5])
        shift = 2
        num_samples = len(signal)
        result = apply_time_delay(signal, shift, num_samples)
        expected = np.array([0, 0, 1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_1d_with_zero_shift(self):
        """Test 1D signal with no shift."""
        signal = np.array([1, 2, 3, 4, 5])
        shift = 0
        num_samples = len(signal)
        result = apply_time_delay(signal, shift, num_samples)
        expected = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result, expected)

    def test_2d_with_positive_shift(self):
        """Test 2D signal with a positive shift."""
        signal1 = np.array([1, 2, 3, 4, 5])
        signal2 = np.array([10, 20, 30, 40, 50])
        signal = np.column_stack((signal1, signal2))
        num_samples = signal.shape[0]
        shift = 3
        result = apply_time_delay(signal, shift, num_samples)

        expected = np.zeros_like(signal)
        expected[:, 0] = [0, 0, 0, 1, 2]
        expected[:, 1] = [0, 0, 0, 10, 20]
        np.testing.assert_array_equal(result, expected)

    def test_float_data(self):
        """Test with floating point data."""
        signal = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        shift = 2
        num_samples = len(signal)
        result = apply_time_delay(signal, shift, num_samples)
        expected = np.array([0.0, 0.0, 1.1, 2.2, 3.3])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sine_wave(self):
        """Test with a sine wave."""
        t = np.linspace(0, 2 * np.pi, 100)
        signal = np.sin(t)
        shift = 25
        num_samples = len(signal)
        result = apply_time_delay(signal, shift, num_samples)

        # Expected: zeros for the first 25 samples, then sine wave
        expected = np.zeros(100)
        expected[shift:] = signal[: 100 - shift]
        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_channel_signal(self):
        """Test with multi-channel signal (more than 2 columns)."""
        signal1 = np.array([1, 2, 3, 4, 5])
        signal2 = np.array([10, 20, 30, 40, 50])
        signal3 = np.array([100, 200, 300, 400, 500])
        signal = np.column_stack((signal1, signal2, signal3))
        shift = 2
        num_samples = signal.shape[0]
        result = apply_time_delay(signal, shift, num_samples)

        expected = np.zeros_like(signal).T
        expected[0] = [0, 0, 1, 2, 3]
        expected[1] = [0, 0, 10, 20, 30]
        expected[2] = [0, 0, 100, 200, 300]
        np.testing.assert_array_equal(result, expected.T)

    def test_edge_case_shift_equals_num_samples(self):
        """Test edge case where shift equals num_samples."""
        signal = np.array([1, 2, 3, 4, 5])
        shift = 5
        num_samples = 5
        result = apply_time_delay(signal, shift, num_samples)
        expected = np.zeros(5)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

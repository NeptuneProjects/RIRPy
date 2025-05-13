# -*- coding: utf-8 -*-

import unittest

import numpy as np

from rirpy.model import (
    # apply_time_delay,
    # create_image_signal,
    # propagate_signal,
    validate_geometry,
    # validate_source_signal,
)


class TestModel(unittest.TestCase):

    # def test_apply_time_delay_1d_with_positive_shift(self):
    #     """Test 1D signal with a positive shift."""
    #     signal = np.array([1, 2, 3, 4, 5])
    #     shift = 2
    #     num_samples = len(signal)
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.array([0, 0, 1, 2, 3])
    #     np.testing.assert_array_equal(result, expected)

    # def test_apply_time_delay_1d_with_zero_shift(self):
    #     """Test 1D signal with no shift."""
    #     signal = np.array([1, 2, 3, 4, 5])
    #     shift = 0
    #     num_samples = len(signal)
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.array([1, 2, 3, 4, 5])
    #     np.testing.assert_array_equal(result, expected)

    # def test_apply_time_delay_2d_with_positive_shift(self):
    #     """Test 2D signal with a positive shift."""
    #     signal1 = np.array([1, 2, 3, 4, 5])
    #     signal2 = np.array([10, 20, 30, 40, 50])
    #     signal = np.column_stack((signal1, signal2))
    #     num_samples = signal.shape[0]
    #     shift = 3
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.zeros_like(signal)
    #     expected[:, 0] = [0, 0, 0, 1, 2]
    #     expected[:, 1] = [0, 0, 0, 10, 20]
    #     np.testing.assert_array_equal(result, expected)

    # def test_apply_time_delay_float_data(self):
    #     """Test with floating point data."""
    #     signal = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    #     shift = 2
    #     num_samples = len(signal)
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.array([0.0, 0.0, 1.1, 2.2, 3.3])
    #     np.testing.assert_array_almost_equal(result, expected)

    # def test_apply_time_delay_sine_wave(self):
    #     """Test with a sine wave."""
    #     t = np.linspace(0, 2 * np.pi, 100)
    #     signal = np.sin(t)
    #     shift = 25
    #     num_samples = len(signal)
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.zeros(100)
    #     expected[shift:] = signal[: 100 - shift]
    #     np.testing.assert_array_almost_equal(result, expected)

    # def test_apply_time_delay_multi_channel_signal(self):
    #     """Test with multi-channel signal (more than 2 columns)."""
    #     signal1 = np.array([1, 2, 3, 4, 5])
    #     signal2 = np.array([10, 20, 30, 40, 50])
    #     signal3 = np.array([100, 200, 300, 400, 500])
    #     signal = np.column_stack((signal1, signal2, signal3))
    #     shift = 2
    #     num_samples = signal.shape[0]
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.zeros_like(signal).T
    #     expected[0] = [0, 0, 1, 2, 3]
    #     expected[1] = [0, 0, 10, 20, 30]
    #     expected[2] = [0, 0, 100, 200, 300]
    #     np.testing.assert_array_equal(result, expected.T)

    # def test_apply_time_delay_edge_case_shift_equals_num_samples(self):
    #     """Test edge case where shift equals num_samples."""
    #     signal = np.array([1, 2, 3, 4, 5])
    #     shift = 5
    #     num_samples = 5
    #     result = apply_time_delay(signal, shift, num_samples)
    #     expected = np.zeros(5)
    #     np.testing.assert_array_equal(result, expected)

    # def test_create_image_signal_1d(self):
    #     """Test that a 1D input signal produces a 1D output with the same dimensions."""
    #     num_samples = 100
    #     shift_amount = 10
    #     scale = 0.5
    #     source_signal = np.ones(num_samples)
    #     result = create_image_signal(source_signal, shift_amount, scale)
    #     self.assertEqual(
    #         source_signal.ndim, result.ndim, "Output ndim should match input ndim"
    #     )
    #     self.assertEqual(
    #         source_signal.shape, result.shape, "Output shape should match input shape"
    #     )

    # def test_create_image_signal_2d(self):
    #     """Test that a 2D input signal produces a 2D output with the same dimensions."""
    #     num_dim = 3
    #     num_samples = 100
    #     shift_amount = 10
    #     scale = 0.5
    #     source_signal = np.ones((num_dim, num_samples))
    #     result = create_image_signal(source_signal, shift_amount, scale)
    #     self.assertEqual(
    #         source_signal.ndim, result.ndim, "Output ndim should match input ndim"
    #     )
    #     self.assertEqual(
    #         source_signal.shape, result.shape, "Output shape should match input shape"
    #     )

    # def test_propagate_signal(self):
    #     """Test the propagate_signal function."""
    #     source_signal = np.array([1.0, 0.5, 0.25, 0.0, 0.0], dtype=np.float64)
    #     sampling_rate = 1000.0
    #     source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    #     receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    #     length_x, length_y, length_z = 5.0, 5.0, 5.0
    #     sound_speed = 1500.0
    #     refl_coeff_wall = 0.8
    #     refl_coeff_ceil = 0.9
    #     cutoff_time = 0.01
    #     num_threads = 2
    #     result = propagate_signal(
    #         source_signal,
    #         sampling_rate,
    #         source_position,
    #         receiver_position,
    #         length_x,
    #         length_y,
    #         length_z,
    #         sound_speed,
    #         refl_coeff_wall,
    #         refl_coeff_ceil,
    #         cutoff_time,
    #         num_threads,
    #     )
    #     self.assertEqual(result.shape, source_signal.shape)
    #     self.assertIsInstance(result, np.ndarray)

    def test_validate_geometry_valid(self):
        """Test that valid geometry does not raise exceptions."""
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0
        try:
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )
        except ValueError:
            self.fail("validate_geometry raised ValueError unexpectedly!")

    def test_validate_geometry_invalid_source(self):
        """Test that invalid source position raises exceptions."""
        source_position = np.array([6.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0
        with self.assertRaises(ValueError):
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )

    def test_validate_geometry_invalid_receiver(self):
        """Test that invalid receiver position raises exceptions."""
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 6.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = 5.0, 5.0, 5.0
        with self.assertRaises(ValueError):
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )

    def test_validate_geometry_invalid_dimensions(self):
        """Test that invalid tank dimensions raise exceptions."""
        source_position = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        receiver_position = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        length_x, length_y, length_z = -5.0, 5.0, 5.0
        with self.assertRaises(ValueError):
            validate_geometry(
                source_position, receiver_position, length_x, length_y, length_z
            )

    # def test_validate_source_signal_valid_1d_array(self):
    #     """Test that a valid 1D array doesn't raise exceptions."""
    #     signal = np.array([1.0, 2.0, 3.0])
    #     try:
    #         validate_source_signal(signal)
    #     except Exception as e:
    #         self.fail(f"validate_source_signal raised an unexpected exception: {e}")

    # def test_validate_source_signal_valid_2d_array(self):
    #     """Test that a valid 2D array doesn't raise exceptions."""
    #     signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #     try:
    #         validate_source_signal(signal)
    #     except Exception as e:
    #         self.fail(f"validate_source_signal raised an unexpected exception: {e}")

    # def test_validate_source_signal_3d_array(self):
    #     """Test that a 3D array raises the correct exception."""
    #     signal = np.ones((2, 3, 4))
    #     with self.assertRaises(ValueError) as context:
    #         validate_source_signal(signal)
    #     self.assertEqual(str(context.exception), "Expected 1D or 2D array, got 3D")

    # def test_validate_source_signal_empty_array(self):
    #     """Test that an empty array raises the correct exception."""
    #     signal = np.array([])
    #     with self.assertRaises(ValueError) as context:
    #         validate_source_signal(signal)
    #     self.assertEqual(str(context.exception), "Input signal cannot be empty")

    # def test_validate_source_signal_scalar(self):
    #     """Test that a scalar raises the correct exception."""
    #     signal = np.array(5.0)
    #     with self.assertRaises(ValueError) as context:
    #         validate_source_signal(signal)
    #     self.assertEqual(str(context.exception), "Input signal cannot be a scalar")


if __name__ == "__main__":
    unittest.main()

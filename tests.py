import numpy as np
import ps_calculation
import unittest

# Run all tests with pytest tests.py
# Run one test with pytest tests.py::TestStringMethods::test_name


class TestStringMethods(unittest.TestCase):

    def test_compare_weighted_and_unweighted_ft(self):

        freq_array = np.arange(40960937.5, 45530761.71875 + 23925.78125, 23925.78125)
        delay_axis = np.fft.fftshift(
            np.fft.fftfreq(len(freq_array), d=freq_array[1] - freq_array[0])
        )

        n_u_points = 3
        n_v_points = 4
        n_freqs = len(freq_array)
        n_pols = 2

        # Create a random array of complex visibilities
        vis_array = np.random.normal(
            0, 5, size=(n_u_points, n_v_points, n_freqs, n_pols)
        ) + 1j * np.random.normal(0, 5, size=(n_u_points, n_v_points, n_freqs, n_pols))
        weights = np.full_like(
            vis_array, 352, dtype=int
        )  # Set all weights to an arbitrary constant

        visibilities_ft_1 = ps_calculation.frequency_ft_no_weighting(
            vis_array, delay_axis, freq_array[0]
        )
        visibilities_ft_2 = ps_calculation.frequency_ft_weighting(
            vis_array,  # Shape (Nu, Nv, Nfreqs, Npols)
            weights,  # Shape (Nu, Nv, Nfreqs, Npols)
            freq_array,  # Shape (Nfreqs)
            delay_axis,
        )
        np.testing.assert_allclose(visibilities_ft_1, visibilities_ft_2)

    def test_compare_memory_save(self):

        freq_array = np.arange(40960937.5, 45530761.71875 + 23925.78125, 23925.78125)
        delay_axis = np.fft.fftshift(
            np.fft.fftfreq(len(freq_array), d=freq_array[1] - freq_array[0])
        )

        n_u_points = 3
        n_v_points = 4
        n_freqs = len(freq_array)
        n_pols = 2

        # Create a random array of complex visibilities
        vis_array = np.random.normal(
            0, 5, size=(n_u_points, n_v_points, n_freqs, n_pols)
        ) + 1j * np.random.normal(0, 5, size=(n_u_points, n_v_points, n_freqs, n_pols))
        weights = np.random.randint(400, size=(n_u_points, n_v_points, n_freqs, n_pols))
        # Make at least one sample have uniform weights across frequency:
        weights[0, 0, :, 0] = 50

        visibilities_ft_1 = ps_calculation.frequency_ft_weighting(
            vis_array, weights, freq_array, delay_axis, memory_save=False
        )
        visibilities_ft_2 = ps_calculation.frequency_ft_weighting(
            vis_array, weights, freq_array, delay_axis, memory_save=True
        )
        np.testing.assert_allclose(visibilities_ft_1, visibilities_ft_2)


if __name__ == "__main__":
    unittest.main()

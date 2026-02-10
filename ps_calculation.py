import pyuvdata
import numpy as np
import sys


def bin_visibilities(uv, uv_resolution_wl=0.5, c=3e8):

    avg_wl = c / np.mean(uv.freq_array)
    uv_array_wls = uv.uvw_array[:, :2] / avg_wl

    # Bin to half wavelength spacing
    u_bin_edges = np.arange(
        0, np.max(uv_array_wls[:, 0]) + uv_resolution_wl, uv_resolution_wl
    )
    v_bin_edges = np.arange(
        0,
        np.max(np.abs(uv_array_wls[:, 1])) + uv_resolution_wl,
        uv_resolution_wl,
    )
    v_bin_edges = np.append(v_bin_edges[::-1], v_bin_edges[1:])  # Add negative values
    u_bin_inds = np.digitize(uv_array_wls[:, 0], u_bin_edges)
    v_bin_inds = np.digitize(uv_array_wls[:, 1], u_bin_edges)
    visibilities_binned = np.zeros(
        (len(u_bin_edges) - 1, len(v_bin_edges) - 1, uv.Nfreqs, uv.Npols),
        dtype=complex,
    )
    weights = np.zeros(
        (len(u_bin_edges) - 1, len(v_bin_edges) - 1, uv.Nfreqs, uv.Npols), dtype=int
    )
    for u_ind in range(len(u_bin_edges) - 1):
        for v_ind in range(len(v_bin_edges) - 1):
            use_inds = np.where((u_bin_inds == u_ind) & (v_bin_inds == v_ind))[0]
            if len(use_inds) > 0:
                weights[u_ind, v_ind, :, :] = np.sum(
                    ~uv.flag_array[use_inds, :, :], axis=0
                )
                visibilities_binned[u_ind, v_ind, :, :] = (
                    np.nansum(
                        uv.data_array[use_inds, :, :] * ~uv.flag_array[use_inds, :, :],
                        axis=0,
                    )
                    / weights[u_ind, v_ind, :, :]
                )

    return visibilities_binned, weights, u_bin_edges, v_bin_edges


def calculate_ps(
    filepath,
    delay_ps=True,
    uv_resolution_wl=0.5,
    kperp_resolution_wl=1.0,
    use_w_terms=False,
    use_flags=True,
    use_freq_flags=False,
):

    if not delay_ps:
        print("ERROR: Imaging PS is not yet supported.")
        sys.exit()

    if use_w_terms:
        print("ERROR: w-terms are not yet supported.")
        sys.exit()

    if use_freq_flags:
        print("ERROR: Frequency-dependent flagging is not yet supported.")
        sys.exit()

    uv = pyuvdata.UVData()
    uv.read(filepath)
    uv.select(polarizations=[-5, -6])
    uv.phase_to_time(np.mean(uv.time_array))
    uv.conjugate_bls(convention="u>0")
    uv.flag_array[np.where(~np.isfinite(uv.data_array))] = True  # Flag all nan-ed data
    if not use_freq_flags:  # Extend flags in frequency
        uv.flag_array = np.repeat(
            np.max(uv.flag_array, axis=1)[:, np.newaxis, :], uv.Nfreqs, axis=1
        )

    visibilities_binned, weights, u_bin_edges, v_bin_edges = bin_visibilities(
        uv, uv_resolution_wl=uv_resolution_wl
    )

    visibilities_ft = np.fft.ifft(
        visibilities_binned, axis=2
    )  # Fourier transform across frequency
    delay_axis = np.fft.fftshift(np.fft.fftfreq(uv.Nfreqs, d=np.mean(uv.channel_width)))

    u_bin_centers = (u_bin_edges[:-1] + u_bin_edges[1:]) / 2
    v_bin_centers = (v_bin_edges[:-1] + v_bin_edges[1:]) / 2
    kperp_vals = np.hypot(u_bin_centers[:, None], v_bin_centers[None, :])
    kperp_bin_edges = np.arange(
        0, np.max(kperp_vals) + kperp_resolution_wl, kperp_resolution_wl
    )
    ps_2d = np.zeros((len(kperp_bin_edges) - 1, uv.Nfreqs, uv.Npols), dtype=float)
    bin_indices = np.digitize(kperp_vals, kperp_bin_edges)
    for bin_ind in range(np.max(bin_indices)):
        use_inds = np.where(bin_indices == bin_ind)
        use_vis = visibilities_ft[use_inds[0], use_inds[1], :, :]
        use_weights = weights[use_inds[0], use_inds[1], 0, :][
            :, np.newaxis, :
        ]  # Assume the weights are identical for all delays
        ps_2d[bin_ind, :, :] = np.nansum(
            np.abs(use_vis) ** 2 * use_weights, axis=0
        ) / np.sum(use_weights, axis=0)
    ps_2d = (ps_2d + ps_2d[:, ::-1, :])[
        :, np.where(delay_axis == 0)[0][0] :, :
    ] / 2  # Fold in delay

    return ps_2d, kperp_bin_edges, delay_axis, uv.freq_array


def dft_visibilities(
    visibilities,
    uv_array_wls,
):

    n_pixels = int(
        np.ceil(2 * np.max(uv_array_wls) / 0.5)
    )  # Corresponds to half wavelength spacing at the largest uv extent
    l_vals = np.linspace(
        -1,
        1,
        num=n_pixels,
    )
    m_vals = np.linspace(
        -1,
        1,
        num=n_pixels,
    )
    l_array, m_array = np.meshgrid(l_vals, m_vals, indexing="ij")
    apparent_sky = np.zeros_like(l_array, dtype=float)
    for vis_ind, vis in enumerate(visibilities):
        apparent_sky += np.real(
            vis
            * np.exp(
                -2
                * np.pi
                * 1j
                * (
                    l_array * uv_array_wls[vis_ind, 0]
                    + m_array * uv_array_wls[vis_ind, 1]
                )
            )
        )
    apparent_sky /= len(visibilities)  # Normalize
    return apparent_sky

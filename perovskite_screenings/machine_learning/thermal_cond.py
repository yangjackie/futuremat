"""
Module contains various functions to run molecular dynamic simulations with machine learning forcefield
to calculate the lattice thermal conductivity of materials

Some of the functions are adapted from https://gitlab.com/vibes-developers/vibes/
"""

import os
import numpy as np
import scipy.signal as sl
from scipy import integrate as si
import random
import multiprocessing

from core.dao.vasp import VaspReader


def _hann(nsamples: int):
    """Return one-side Hann function

    Args:
        nsamples (int): number of samples
    """
    return sl.windows.hann(2 * nsamples)[nsamples:]


def _correlate(f1, f2, normalize=1, hann=True):
    """Compute correlation function for signal f1 and signal f2

    Reference:
        https://gitlab.com/flokno/python_recipes/-/blob/master/mathematics/
        correlation_function/autocorrelation.ipynb

    Args:
        f1: signal 1
        f2: signal 2
        normalize: no (0), by length (1), by lag (2)
        hann: apply Hann window
    Returns:
        the correlation function
    """
    a1, a2 = (np.asarray(f) for f in (f1, f2))
    Nt = min(len(a1), len(a2))

    if Nt != max(len(a1), len(a2)):
        msg = "The two signals are not equally long: "
        msg += f"len(a1), len(a2) = {len(a1)}, {len(a2)}"
        warn(msg)

    corr = sl.correlate(a1[:Nt], a2[:Nt], 'full')[Nt - 1:]
    if normalize is True or normalize == 1:
        corr /= corr[0]
    elif normalize == 2:
        corr /= np.arange(Nt, 0, -1)

    if hann:
        corr *= _hann(Nt)

    return corr


correlate = _correlate


def cumtrapz(series, index=None, axis=0, initial=0):
    """wrap `scipy.integrate.cumtrapz`"""
    array = np.asarray(series)
    if index is not None and len(index) > 1:
        x = np.asarray(index)
    else:
        # print(f"index = {index}, use `x=None`")
        x = None
    ct = si.cumtrapz(array, x=x, axis=axis, initial=initial)

    return ct


def get_cumtrapz(series, **kwargs):
    """Compute cumulative trapezoid integral of ndarray, Series/DataArray

    Return:
        ndarray/Series/DataArray: cumulative trapezoid rule applied
    """
    if isinstance(series, np.ndarray):
        ctrapz = cumtrapz(series, **kwargs)
        return ctrapz

    raise TypeError("`series` not of type ndarray, Series, or DataArray?")


def moving_average(data, window_size=150):
    i = 0
    moving_averages = []

    while i < len(data) - window_size + 1:
        window_average = round(np.sum(data[i:i + window_size]) / window_size, 2)
        moving_averages.append(window_average)
        i += 1

    return moving_averages


def get_k_trajectory(input):
    q = input[0]
    start_index = input[1]
    traj_length = input[2]
    pre_factor = input[3]
    _q = q[start_index:start_index + traj_length]
    _corr = _correlate(_q, _q)
    _kappa = pre_factor * get_cumtrapz(_corr)
    return _kappa


def cumulative_kappa(qx, qy, qz, time_step=1, volume=None, temp=500, number_of_sampled_traj=None, number_of_thread=28,
                     write_output=True):
    CONVERSION_FACTOR = 1.85392e10
    pre_factor = CONVERSION_FACTOR / (3 * volume * temp * temp)
    _times = []

    if (number_of_sampled_traj is None) and (number_of_thread is not None):
        number_of_sampled_traj = number_of_thread * 2

    pool = multiprocessing.Pool(number_of_thread)

    for t in np.arange(1000, len(qx) - 1000, 1000):  # np.arange(250,len(qx),250):
        traj_length = len(qx) - t
        _times.append(traj_length / 1000)
        start_interval = len(qx) - traj_length
        start_points = [i for i in range(start_interval)]

        kappa_ave = [0, 0, 0]

        for ii, _q in enumerate([qx, qy, qz]):
            start_index = random.choices(start_points, k=number_of_sampled_traj)
            map_data = [[_q, s, traj_length, pre_factor] for s in start_index]

            p = pool.map_async(get_k_trajectory, map_data, chunksize=1)
            kappas = p.get(999999)

            kappa_ave[ii] = np.mean(kappas)

        kappa_mean = np.mean(kappa_ave)

        if write_output is True:
            out_file = open('kappa.dat', 'a')
            out_file.write(str(traj_length / 1000 * time_step) + '\t' + str(round(kappa_ave[0], 5)) + '\t' + str(
                round(kappa_ave[1], 5)) + '\t' + str(round(kappa_ave[2], 5)) + '\t' + str(round(kappa_mean, 5)) + '\n')
            out_file.close()

    pool.terminate()

if __name__ == '__main__':
    get_cumulant_k = True

    if get_cumulant_k:
        folder = os.getcwd() + '/'
        qx_1, qy_1, qz_1 = VaspReader(input_location=folder + 'ML_HEAT_1').read_ml_heat()
        qx_2, qy_2, qz_2 = VaspReader(input_location=folder + 'ML_HEAT').read_ml_heat()
        qx = np.concatenate((qx_1, qx_2))
        qy = np.concatenate((qy_1, qy_2))
        qz = np.concatenate((qz_1, qz_2))
        #qx, qy, qz = VaspReader(input_location=folder + 'ML_HEAT').read_ml_heat()
        cumulative_kappa(qx, qy, qz, time_step=1, volume=7.8258 ** 3, temp=500, number_of_sampled_traj=None,
                         number_of_thread=28, write_output=True)

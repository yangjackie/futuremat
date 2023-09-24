"""
Module contains various functions to run molecular dynamic simulations with machine learning forcefield
to calculate the lattice thermal conductivity of materials

Some of the functions are adapted from https://gitlab.com/vibes-developers/vibes/
"""

import os
from warnings import warn

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

    for t in np.arange(250,len(qx),250): #np.arange(100, len(qx) - 100, 100):
        #traj_length = len(qx) - t
        traj_length = t
        #_times.append(traj_length / 1000)
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
            out_file.write(str(traj_length / 1000 * time_step) + '\t' + "{:.4f}".format(kappa_ave[0]) + '\t' + "{:.4f}".format(kappa_ave[1])
                           + '\t' + "{:.4f}".format(kappa_ave[2]) + '\t' + "{:.4f}".format(kappa_mean) + '\n')
            out_file.close()

    pool.terminate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='cmd utils for calculating thermal conductivity from vasp MLFF',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-cum", "--cum", action='store_true', help='switch to determing whether to run cummulative sum for kappa')
    parser.add_argument("-t", "--temp", type=float, default=300, help='temperature of the simulation')
    parser.add_argument("-v", "--volume", type=float, default=None, help='volume of the simulation supercell')
    parser.add_argument("-n", "--nthread", type=int, default=32, help='number of threads for parallelisation')
    parser.add_argument("-potim","--potim",type=int,default=1,help='Time step for MD simulation')
    args = parser.parse_args()


    if args.cum:
        folder = os.getcwd() + '/'

        if args.volume is None:
            from core.dao.vasp import VaspReader
            crystal = VaspReader(input_location=folder+'/CONTCAR').read_POSCAR()
            volume = crystal.lattice.volume
        else:
            volume=args.volume

        print("Cell volume is ",volume)

        import glob
        all_data=glob.glob(folder+'ML_HEAT*')

        #if len(all_data)<120:
        #    raise Exception('Not a completed trajectory!')

        qx = []
        qy = []
        qz = []
        #for i in range(len(all_data)):
        _qx, _qy, _qz = VaspReader(input_location=folder + 'ML_HEAT').read_ml_heat()
        qx=qx+_qx
        qy=qy+_qy
        qz=qz+_qz

        #print('Reading file No. '+str(i)+' Cumulative length of data: '+str(len(qx)))

        cumulative_kappa(qx, qy, qz, time_step=args.potim, volume=volume, temp=args.temp, number_of_sampled_traj=None,
                         number_of_thread=args.nthread, write_output=True)

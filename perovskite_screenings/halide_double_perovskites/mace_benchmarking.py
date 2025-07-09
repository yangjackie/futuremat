import numpy as np
from mace.calculators import mace_mp
from ase.io import read
from ase import Atoms
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import os
import pickle

from artificial_intelligence.phonon_plotter import prepare_and_plot

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rc
from ase.neighborlist import *
rc('text', usetex=True)

import matplotlib.pylab as pylab
params = {'legend.fontsize': '14',
          'figure.figsize': (7.5, 6),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

class Benchmarking:
    def __init__(self,
                 mace_model_path: str = "/Users/z3079335/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
                 mace_model_name: str = "mace-mp-0b3-medium.model",
                 directory: str = "/Users/z3079335/OneDrive - UNSW/Documents/Projects/perovskite_anharmonic_screening/halide_double_perovskites/raw_data/",
                 system: str = "fluorides"):
        self.mace_model_path = mace_model_path
        self.mace_model_name = mace_model_name
        self.directory = directory
        self.system = system
        self.path = f"{self.directory}{self.system}/"

    @property
    def calculator(self):
        if not hasattr(self, '_calculator') or self._calculator is None:
            self._calculator = mace_mp(model=self.mace_model_path + self.mace_model_name, device='cpu')
        return self._calculator

    def benchmark_frequencies(self):
        _cwd = os.getcwd()
        os.chdir(self.path)
        _system_cwd = os.getcwd()
        all_compounds = list(sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith('dpv_')]))

        for comp in all_compounds:
            print(f"Processing {comp}...")
            filename = _system_cwd + "/" + comp + '/' + comp + '_dft_mlff_phonon_' + self.mace_model_name.replace(
                '.model', '') + '.pdf'

            try:
                return_dict = prepare_and_plot(dft_path=_system_cwd + "/" + comp + '/',
                                               calculator=self.calculator,
                                               savefig=True,
                                               filename=filename,
                                               plot=False)

                with open(_system_cwd + "/" + comp + '/phonon_data_' + str(self.mace_model_name.replace(
                        '.model', '')) + '.pkl', 'wb') as f:
                    pickle.dump(return_dict, f)
            except:
                pass
        os.chdir(_cwd)

    def plot_frequency_data(self):
        _cwd = os.getcwd()
        os.chdir(self.path)
        _system_cwd = os.getcwd()
        all_compounds = list(sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith('dpv_')]))

        stdev_frequencies = []
        stdev_velocities = []

        for comp in all_compounds:
            os.chdir(_system_cwd + "/" + comp + '/')
            pickle_name='phonon_data_' + str(self.mace_model_name.replace('.model', '')) + '.pkl'
            if os.path.isfile(pickle_name):
                print(f"Loading phonon data for {comp}...")

                with open(pickle_name, 'rb') as f:
                    phonon_data = pickle.load(f)

                dft_frequencies = phonon_data['dft_frequencies']
                mace_frequencies = phonon_data['mace_frequencies']
                dft_group_velocities = phonon_data['dft_group_velocities']
                mace_group_velocities = phonon_data['mace_group_velocities']

                _dft_frequencies = np.ndarray.flatten(np.array(dft_frequencies))
                _dft_frequencies = _dft_frequencies.astype(complex)
                _dft_frequencies[_dft_frequencies < 0] = 1j * np.abs(_dft_frequencies[_dft_frequencies < 0])

                _mace_frequencies = np.ndarray.flatten(np.array(mace_frequencies))
                _mace_frequencies = _mace_frequencies.astype(complex)
                _mace_frequencies[_mace_frequencies < 0] = 1j * np.abs(_mace_frequencies[_mace_frequencies < 0])

                stdev_freq = np.real(
                    np.sqrt(np.average(np.square(np.square(_mace_frequencies) - np.square(_dft_frequencies)))))

                vel_stdev = np.sqrt(np.average(np.square(dft_group_velocities - mace_group_velocities)))
                stdev_frequencies.append(stdev_freq)
                stdev_velocities.append(vel_stdev)

            os.chdir(_system_cwd)

        fig, ax = plt.subplots()

        ax.scatter(stdev_frequencies, stdev_velocities,marker='o',s=40, alpha=0.8, edgecolors='k', facecolors='#66A5AD')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.05,100])
        ax.set_ylim([0.7,100])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

        ax.xaxis.set_minor_formatter(ticker.FuncFormatter(format_minor_tick_label))
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(format_minor_tick_label))

        ax.set_xlabel('RMSE($\\omega^2$ (THz$^2$))')
        ax.set_ylabel('RMSE($v_g$ (THz$\\cdot$ $\\mbox{\\AA}$))')
        ax.set_title(self.system)
        plt.tight_layout()
        plt.show()
        os.chdir(_cwd)

def format_minor_tick_label(x, pos):
    if (abs(x) < 1) :
        x = float(f'{x:.1f}')
        if (abs(x) > 0.1) and ((10*x) % 2 == 0):
            return f'{x:.1f}'
        #elif (abs(x) <= 0.1) and (int(100 * x) % 2 == 0):
        #    print(x)
        #    return f'{x:.2f}'
        return None
    if (abs(x) > 1) and (abs(x) < 11):
        x = int(x)
        if (x % 2 == 0):
            return f'{int(x)}'
        return None
    if (abs(x) > 11):
        x = int(x)
        if (x%10==0) and (x%20==0):
            return f'{int(x)}'
    else:
        return f'{int(x)}'


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmarking MACE model for halide double perovskites.")
    parser.add_argument("--system", type=str, default="fluorides",help="System to benchmark (e.g., 'fluorides', 'chlorides', 'bromides', 'iodides').")
    parser.add_argument("--run_benchmark", action='store_true', help="Run the benchmarking of frequencies.")
    parser.add_argument("--plot_freq", action='store_true', help="Plot the frequency data.")
    parser.add_argument("--model_name", type=str, default="mace-mp-0b3-medium.model")
    args = parser.parse_args()

    benchmark = Benchmarking(system=args.system,mace_model_name=args.model_name)
    if args.run_benchmark:
        benchmark.benchmark_frequencies()
    elif args.plot_freq:
        benchmark.plot_frequency_data()
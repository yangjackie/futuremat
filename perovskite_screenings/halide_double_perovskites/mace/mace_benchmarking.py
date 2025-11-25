import time
import numpy as np
from mace.calculators import mace_mp
from ase.io import read, Trajectory
from ase import Atoms, units
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import os
import pickle

from artificial_intelligence.phonon_plotter import prepare_and_plot

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
from ase.neighborlist import *
from ase.io import read

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
                 mace_model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
                 mace_model_name: str = "mace-mp-0b3-medium.model",
                 directory: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/perovskite_anharmonic_screening/halide_double_perovskites/raw_data/",
                 system: str = "fluorides"):
        self.mace_model_path = mace_model_path
        self.mace_model_name = mace_model_name
        self.directory = directory
        self.system = system
        self.path = f"{self.directory}{self.system}/"

        # arranged in the order of improving performances
        self.available_mace_models = ['MACE-matpes-pbe-omat-ft.model', 'mace-mpa-0-medium.model',
                                      'mace-mp-0b3-medium.model', 'mace-omat-0-medium.model']

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
                                               plot=True)

                with open(_system_cwd + "/" + comp + '/phonon_data_' + str(self.mace_model_name.replace(
                        '.model', '')) + '.pkl', 'wb') as f:
                    pickle.dump(return_dict, f)
            except:
                pass
        os.chdir(_cwd)

    def plot_cumulative_mace_statistics(self):
        _cwd = os.getcwd()
        systems = ['fluorides', 'chlorides', 'bromides', 'iodides']
        all_data = {k: {'fluorides': {'stdev_freq': [], 'stdev_vel': []},
                        'chlorides': {'stdev_freq': [], 'stdev_vel': []},
                        'bromides': {'stdev_freq': [], 'stdev_vel': []},
                        'iodides': {'stdev_freq': [], 'stdev_vel': []}} for k in self.available_mace_models}

        # data collections
        for model in self.available_mace_models:
            for system in systems:
                path = f"{self.directory}{system}/"
                # get all the compounds in this system
                os.chdir(path)
                all_compounds = list(sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith('dpv_')]))
                for comp in all_compounds:
                    os.chdir(comp + '/')
                    print(f"Processing {comp}: {model} .....")
                    pickle_name = 'phonon_data_' + str(model.replace('.model', '')) + '.pkl'
                    if os.path.isfile(pickle_name):
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

                        stdev_vel = np.sqrt(np.average(np.square(dft_group_velocities - mace_group_velocities)))
                        all_data[model][system]['stdev_freq'].append(stdev_freq)
                        all_data[model][system]['stdev_vel'].append(stdev_vel)
                    os.chdir("../..")
                os.chdir(_cwd)

        # making the box plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        flierprops = dict(marker='+', markeredgecolor='#6FB98F', markersize=5, alpha=0.5)
        positions = [1, 2, 3, 4, 6.5, 7.5, 8.5, 9.5, 12, 13, 14, 15, 17.5, 18.5, 19.5, 20.5]
        tick_positions = [2.5, 8, 13.5, 19]
        tick_labels = ['matpes-pbe-omat-ft', 'mpa-0-medium', 'mp-0b3-medium', 'omat-0-medium']

        colors = ['#003B46', '#07575B', '#66A5AD', '#C4DFE6']
        for i in range(3):
            colors += colors

        legend_patches = [
            mpatches.Patch(color='#003B46', label='Fluorides'),
            mpatches.Patch(color='#07575B', label='Chlorides'),
            mpatches.Patch(color='#66A5AD', label='Bromides'),
            mpatches.Patch(color='#C4DFE6', label='Iodides'),
        ]

        # first one
        data = []
        for model in self.available_mace_models:
            for system in systems:
                print(f"Processing {system}: {model} .....")
                data.append(all_data[model][system]['stdev_freq'])
                print(all_data[model][system]['stdev_freq'][:10])
        bx1 = ax1.boxplot(data, showfliers=True, widths=0.5, flierprops=flierprops, positions=positions,
                          patch_artist=True)
        for box, color in zip(bx1['boxes'], colors):
            box.set_facecolor(color)

        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, rotation=20, fontsize=10)
        ax1.set_ylabel('RMSE($\\omega^2$ (THz$^2$))', fontsize=15)
        ax1.set_yscale('log')

        ax1.legend(handles=legend_patches, loc='upper right', title="Material Systems", fontsize=10)

        data = []
        for model in self.available_mace_models:
            for system in systems:
                print(f"Processing {system}: {model} .....")
                data.append(all_data[model][system]['stdev_vel'])
        bx2 = ax2.boxplot(data, showfliers=True, widths=0.5, flierprops=flierprops, positions=positions,
                          patch_artist=True)
        for box, color in zip(bx2['boxes'], colors):
            box.set_facecolor(color)

        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=20, fontsize=10)
        ax2.set_ylabel('RMSE($v_g$ (THz$\\cdot$ $\\mbox{\\AA}$))', fontsize=15)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.show()

    def plot_frequency_data(self):
        # loading the anharmonic score data for plotting
        from ase.db import connect
        db_name = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/perovskite_anharmonic_screening/halide_double_perovskites/HDP_database/double_halide_pv.db"
        db = connect(db_name)

        _cwd = os.getcwd()
        os.chdir(self.path)
        _system_cwd = os.getcwd()
        all_compounds = list(sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith('dpv_')]))

        stdev_frequencies = []
        stdev_velocities = []
        sigmas = []

        target_frequencies = []
        target_velocities = []

        for comp in all_compounds:

            row = None
            try:
                row = db.get(selection=[('uid', '=', comp)])
            except:
                pass
            if row is not None:
                try:
                    sigma = row.key_value_pairs['sigma_300K_single']

                except KeyError:
                        continue
            else:

                print(f"Skipping {comp}...no precomputed sigma value")
                continue

            os.chdir(_system_cwd + "/" + comp + '/')
            pickle_name = 'phonon_data_' + str(self.mace_model_name.replace('.model', '')) + '.pkl'
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

                if 'K2RbSbF6' in comp:
                    target_frequencies.append(stdev_freq)
                    target_velocities.append(vel_stdev)
                sigmas.append(sigma)

            os.chdir(_system_cwd)

        fig, ax = plt.subplots()

        # ax.scatter(stdev_frequencies, stdev_velocities,marker='o',s=40, alpha=0.8, edgecolors='k', facecolors='#66A5AD')

        sc = ax.scatter(stdev_frequencies, stdev_velocities, marker='o', s=60, alpha=0.8, edgecolor=None, c=sigmas,
                        cmap=plt.get_cmap('coolwarm'), vmax=1.2)

        ax.scatter(target_frequencies,target_velocities,marker='s', s=60, alpha=0.8, edgecolor='k', facecolor=None)
        plt.colorbar(sc, ax=ax, label='$\\sigma^{(2)}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_xlim([0.05, 100])
        #ax.set_ylim([0.7, 100])
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
    if (abs(x) < 1):
        x = float(f'{x:.1f}')
        if x in [0.2, 0.4, 0.7]:
            return f'{x:.1f}'
        # elif (abs(x) <= 0.1) and (int(100 * x) % 2 == 0):
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
        if x in [20, 40, 60, 100]:
            return f'{int(x)}'
    else:
        return f'{int(x)}'


class MACEMolecularDynamicsRunner(object):

    def __init__(self,
                 mace_model_path: str = "/Users/jackyang-macmini/OneDrive - UNSW/Documents/Projects/artificial_intelligence/mace_models/",
                 mace_model_name: str = "mace-mp-0b3-medium.model",
                 target_temperature: float = 300.0,
                 equilibration_steps: int = 5000,
                 production_steps: int = 10000,
                 andersen_prob: float = 0.5,
                 ):
        self.mace_model_path = mace_model_path
        self.mace_model_name = mace_model_name
        self.target_temperature = target_temperature
        self.equilibration_steps = equilibration_steps
        self.production_steps = production_steps
        self.andersen_prob = andersen_prob

    @property
    def calculator(self):
        if not hasattr(self, '_calculator') or self._calculator is None:
            self._calculator = mace_mp(model=self.mace_model_path + self.mace_model_name, device='cpu')
        return self._calculator

    def initialise_structure(self, poscar_path: str = None):
        assert (poscar_path is not None)
        self.structure = read(poscar_path)
        self.structure.calc = self.calculator
        self.structure.set_pbc(True)

        self.trajectory_file = poscar_path.replace("SPOSCAR",
                                                   "trajectory-" + self.mace_model_name.replace(".model", "") + ".traj")

    def initialise_structure_from_xml(self, xml_path: str = None):
        import random
        assert (xml_path is not None)
        self.structure = read(xml_path,index=random.randint(0,700))
        self.structure.calc = self.calculator
        self.structure.set_pbc(True)
        _xml_path = xml_path.replace("/MD","")
        for i in range(10):
            if 'vasprun_prod_'+str(i)+'.xml' in _xml_path:
                target='vasprun_prod_'+str(i)+'.xml'
        self.trajectory_file = _xml_path.replace(target,
                                                   "trajectory-" + self.mace_model_name.replace(".model", "") + ".traj")

    def equilibrate_structure(self):
        # doesnt work
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
        from ase.md.verlet import VelocityVerlet
        from ase import units
        MaxwellBoltzmannDistribution(self.structure, temperature_K=10)
        Stationary(self.structure)
        dyn = VelocityVerlet(self.structure, timestep=1.0 * units.fs)

        def rescale_velocity():
            ekin = self.structure.get_kinetic_energy()
            dof = 3 * len(self.structure) - 3  # degrees of freedom (remove COM motion)
            T_inst = ekin / (0.5 * dof * units.kB)  # instantaneous T
            scale = (self.target_temperature / T_inst) ** 0.5
            self.structure.set_velocities(self.structure.get_velocities() * scale)

        dyn.attach(rescale_velocity, interval=10)

        def print_status():
            step = dyn.nsteps
            epot = self.structure.get_potential_energy() / float(len(self.structure))
            ekin = self.structure.get_kinetic_energy() / float(len(self.structure))
            temp = ekin / (1.5 * units.kB)
            print(f"Equilibration Step {step:6d} | E_pot = {epot:.8f} eV | E_kin = {ekin:.8f} eV | T = {temp:.4f} K", end=" ", flush=True)

        dyn.attach(print_status, interval=5)
        dyn.run(self.equilibration_steps)

    def andersen_thermostat_nvt(self,trajectory_file=None):
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.andersen import Andersen
        import sys
        # Step 1 Initialise the velocities according to the target temperature
        start_time = time.time()
        MaxwellBoltzmannDistribution(self.structure, temperature_K=self.target_temperature)
        # Step 2 Define the Andersen thermostat
        dyn = Andersen(self.structure,
                       timestep=1.0 * units.fs,
                       temperature_K=self.target_temperature,
                       andersen_prob=self.andersen_prob)

        # logging function
        def print_status():
            step = dyn.nsteps
            epot = self.structure.get_potential_energy() / len(self.structure)
            ekin = self.structure.get_kinetic_energy() / len(self.structure)
            temp = ekin / (1.5 * units.kB)
            elapsed_time = time.time() - start_time
            sys.stdout.write(
                f"\rStep {step:6d} | Time = {step * 1.0:.1f} fs| E_pot = {epot:.3f} eV | E_kin = {ekin:.3f} eV | T = {temp:.1f} K | ...... | Elapsed: {elapsed_time:7.2f} s | Time per cycle: {elapsed_time/(step+1):7.2f} s")
            sys.stdout.flush()

        dyn.attach(print_status, interval=1)
        if trajectory_file is None:
            traj_file = self.trajectory_file
        dyn.attach(Trajectory(trajectory_file, "w", self.structure))

        dyn.run(self.production_steps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmarking MACE model for halide double perovskites.")
    parser.add_argument("--system", type=str, default="fluorides",
                        help="System to benchmark (e.g., 'fluorides', 'chlorides', 'bromides', 'iodides').")
    parser.add_argument("--run_benchmark", action='store_true', help="Run the benchmarking of frequencies.")
    parser.add_argument("--plot_freq", action='store_true', help="Plot the frequency data.")
    parser.add_argument("--summary_plot", action='store_true', help="Plot the summary of frequency data.")
    parser.add_argument("--model_name", type=str, default="mace-mp-0b3-medium.model")
    parser.add_argument("--md", action='store_true', help="Run MD simulations with MACE model.")
    parser.add_argument("--md_run_this", action='store_true', help="Run MD simulations with MACE model for this compound.")
    parser.add_argument("--md_run_system", type=str, default=None, help="Run MD simulations with MACE model for this system.")

    parser.add_argument("--nsteps", type=int, default=10000, help="Number of steps to run.")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    if not args.md:
        benchmark = Benchmarking(system=args.system, mace_model_name=args.model_name)
        if args.run_benchmark:
            benchmark.benchmark_frequencies()
        elif args.plot_freq:
            benchmark.plot_frequency_data()
        elif args.summary_plot:
            benchmark.plot_cumulative_mace_statistics()
    else:
        if args.md_run_this:
            md_runner = MACEMolecularDynamicsRunner(mace_model_name=args.model_name, production_steps=args.nsteps)
            #md_runner.initialise_structure(poscar_path=os.getcwd()+'/SPOSCAR')
            md_runner.initialise_structure_from_xml(xml_path=os.getcwd()+'/MD/vasprun_prod_2.xml')
            md_runner.andersen_thermostat_nvt()
        else:
            import glob
            assert args.md_run_system is not None
            assert args.md_run_system in ['fluorides', 'chlorides', 'bromides', 'iodides']
            cwd = os.getcwd()
            os.chdir(args.md_run_system)
            sys_cwd = os.getcwd()
            all_directories = glob.glob(os.getcwd() + '/dpv_*/')
            for id,directory in enumerate(list(sorted(all_directories))):
                os.chdir(directory)
                print("\n")
                print(">>>>>> Working directory: " + os.getcwd().split('/')[-1] + " <<<<<<")
                print("......................" + str(id + 1) + '/' + str(len(all_directories)) + "......................")

                md_runner = MACEMolecularDynamicsRunner(mace_model_name=args.model_name,
                                                        production_steps=args.nsteps)

                for run_id in range(2): # run 2 different MD trajectories
                    #check if this has been run before.
                    print("\n")
                    print("...... MD run: " + str(run_id + 1) + '/' + str(2) + "..system.." + str(id + 1) + '/' + str(len(all_directories)) + "..."+">> Working directory: " + os.getcwd().split('/')[-1] + " <<")
                    run_this = True
                    trajectory_file = "andersen_md_"+args.model_name.replace(".model", "") + "_run_"+str(run_id+1)+".traj"
                    if os.path.exists(os.getcwd()+'/'+trajectory_file):
                        from ase.io import read as ase_read
                        traj=ase_read(os.getcwd()+'/'+trajectory_file,':')
                        if len(traj)==args.nsteps+1:
                            run_this = False
                    if run_this:
                        if os.path.exists(os.getcwd() + '/MD/vasprun_prod_2.xml'):
                            xml_path = os.getcwd() + '/MD/vasprun_prod_2.xml'
                        elif os.path.exists(os.getcwd() + '/MD/vasprun_prod_1.xml'):
                            xml_path = os.getcwd() + '/MD/vasprun_prod_1.xml'
                        else:
                            run_this = False
                            print("No pre-existing DFT MD calculations, skip this")
                    if run_this:
                        #md_runner = MACEMolecularDynamicsRunner(mace_model_name=args.model_name,
                        #                                       production_steps=args.nsteps)

                        try:
                            md_runner.initialise_structure_from_xml(xml_path=xml_path)
                        except:
                            try:
                                md_runner.initialise_structure_from_xml(xml_path=os.getcwd() + '/MD/vasprun_prod_2.xml')
                            except:
                                run_this = False
                        if run_this:
                            md_runner.andersen_thermostat_nvt(trajectory_file=trajectory_file)

                os.chdir(sys_cwd)
            os.chdir(cwd)

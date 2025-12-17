# Electron-Phonon Coupling (EPH) Package

The EPH package provides tools for computing material properties that are affected by electron-phonon couplings in materials.

## Installation

The EPH package requires:
- `phonopy`: Lattice dynamics calculations
- `pymatgen`: Structure manipulation and VASP I/O
- `numpy`, `scipy`: Scientific computing

## Overview

This package contains modules for:
- **Bandgap Renormalization**: Providing orchestrations of the workflows to compute temperature-dependent and mode-dependent bandgap shifts due to electron-phonon interactions, following the work on **J. Am. Chem. Soc., 147, 37506 (2025).**


## Handling I/O and VASP calculations

The implmentations heavily relies on the `pymatgen` API so interested user can use this code without digging into how the rest of our (old) codes were implmented. The aim is to automate and standardised the computations as much as possible, so each step of the workflow can be called with minimal input and setup efforts. In particular, the `core.calculators.pymatgen.vasp` module provide our own extension of the the **pymatgen** `Vasp` calculator, which execute the VASP calculation with a given setting, followed by extracting informations for post-processing. For using this calculator:

1. A set of standard VASP incar settings can be loaded from `core.data.vasp_settings` as dictionary, which is to be passed into the vasp calculator as the keyword arguments.
2. For the VASP binary and the MPI command that are used to execute the VASP calculations, at the moment this is hard-coded in the `command` and `executable` functions in the `core.calculators.abstract_calculator.VaspBase` module, so the code knows how to run VASP calculation once the calculation is setup. **NOTE** Many things are hard-coded here and will be reimplemented to become more flexible later.
3. The code automatically pick up the number of cores to be used when the job is submitted to the PBS queue. You should check if the correct environment variable also works for your system.




# Usage

## Temperature-dependent band gap variations

Here we briefly describe the steps to carry out the workflow to compute temperature-dependent band gap change due to electron-phonon couplings. 

### Structure Optimisation

Given a starting structure in POSCAR file, the following command line option can be used to optimise the starting structure with our default calculation setting.
```bash
python3 electron_phonon_coupling.py --optimise_structure
```
which will pickup the POSCAR file in the current directoryand perform structural optimisation task in a autocreated subfolder `structure_optimisation` in the current directory. Auto-skip is the default option if a previous calculation under this subfolder is already existing, so the same calculation will not be repeated.

With API:
```python
from core.phonon.eph.electron_phonon_coupling import execute_vasp_calculation
from core.data.vasp_settings import DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS

execute_vasp_calculation(
    structure=Structure.from_file('POSCAR'),
    params=DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS,
    job_type="structure_optimisation",
)
```

By default, the VASP calculation will be performed with PBEsol functional with 520 eV energy cutoff. A densed MP KPOINT grids with `kppa=2000` is selected to use the `pymatgen` API to autogenerate the KPOUNTS file for the calculation.

### Compute the Born charges for the optimised structure.

This is for polar crystals. Starting from the optimised structure, it can be carried out from the command line:
```bash
python3 electron_phonon_coupling.py --calculate_born_charges --structure_file CONTCAR
```
which will be carried out in the `born_charges` subdirectory. Currently, the `phonopy` binary `phonopy-vasp-born` will be used to automate the generation of the `BORN` file from `vasprun.xml` file. The generated `BORN` file will be placed under the root-directory. **Shall the subsequent phonon analysis be carried out with NAC, the BORN file should be copied into the** `finite_displacement_phonons` directory.  

This, as well as the following steps, can also be called from Python API. Examples are given in the `electron_phonon_coupling.py` module which will not be repeated here. 

### Phonon computation with finite-displacement approach

Starting from the optimised structure, it can be carried out from the command line:
```bash
python3 electron_phonon_coupling.py --calculate_phonons --structure_file CONTCAR
```
This will set up a subdirectory called `finite_displacement_phonons` for the task. At the moment, the VASP single point calculations **sequentially** to determine the atomic forces for each finite displaced structure. If your structure has low symmetry and large supercell size, for which you would like all these calculations to be performed parallely, you will need to implement your own workflow to handle this!

KPOINTS are auto-generated same to the structural optimisation workflow.

After all VASP single point calculations, the `force_constants.hdf5` file that stores the force constants are generated in the same sub-diretory by using the `phonopy` Python API.

### Visualise the phonon band structure

To plot the phonon band structure from finite-displacement calculations, one can use the following command line option executed in the root calculation folder:

```bash
python3 electron_phonon_coupling.py -ppbs
```

Provided the `BORN` file is also in the `finite_displacement_phonons` directory, one can also apply the non-analytical correction for the LO-TO phonon splitting by including the `-nac` tag:

```bash
python3 electron_phonon_coupling.py -ppbs -nac
```

This will then generate the `phonon_band_structure.pdf` file under the `finite_displacement_phonons` directory.


### Electronic structure calculations

Starting from an input structure, the following command:
```bash
python3 electron_phonon_coupling.py --calculate_band_structure --structure_file CONTCAR --num_kpoints 20 
```
set up, and carry out a band structure calculation for the input structure specified in the input file under the `band_structure` sub-directory. We use the `pymatgen` to help automatically determine the k-point path and write out the KPOINTS file for the input structure. This is achieved with the following code snipplet in the `Vasp` class in the `core.calculatiors.pymatgen.vasp` module:

```python
from pymatgen.symmetry.bandstructure import HighSymmKpath

kpath = HighSymmKpath(self.structure)  # auto-detects space group and path
kpoints_bs = Kpoints.automatic_linemode(
    divisions=self.kppa_band,  # number of points between labels
    ibz=kpath,  # can pass HighSymmKpath directly
)
kpoints_bs.write_file(filename)
```

For visualising the computed band structure, we recommend the use of the sumo-plot package (https://smtg-bham.github.io/sumo/index.html). e.g.
```bash
sumo-bandplot --band-edges
```


## Module Structure

- `electron_phonon_coupling.py`: Bandgap renormalization calculations


## References

- Phonopy documentation: https://phonopy.github.io/
- PyMatGen: https://pymatgen.org/


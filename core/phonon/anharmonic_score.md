# Anharmonic score — module documentation

This document describes the `anharmonic_score.py` module and the `AnharmonicScore`
class. The implementation analyses vibrational anharmonicity from ab-initio
molecular dynamics (AIMD) trajectories (VASP `vasprun.xml` or ASE `.traj`) and
is based on the algorithm from F. Knoop et al., "Anharmonic Measures for
Materials" (arXiv:2006.14672). Example of how to include higher order anharmonic contributions can be found from J. Yang and S. Li, "An atlas of room-temperature stability and vibrational anharmonicity of cubic perovskites" (DOI: 10.1039/D2MH00272H)

Contents
- Overview
- Installation / requirements
- Quick usage (script & API)
- `AnharmonicScore` class: constructor, important properties and methods
- Notes and tips

Overview
--------
The module provides tools to:

- Load reference crystal structure (POSCAR/CONTCAR) and MD trajectories
  (`vasprun.xml` or ASE `.traj`).
- Load phonon force-constants (HDF5 or FORCE_SETS/SPOSCAR) for harmonic
  forces via phonopy.
- Compute harmonic, as well as (optionally) third- and fourth-order contributions to atomic forces
  from displacements along an MD trajectory.
- Compute anharmonic forces (difference between DFT MD forces and harmonic
  prediction) and several measures/plots used to quantify anharmonicity.

Requirements
------------
- Python 3.8+
- numpy, matplotlib, phonopy, seekpath
- ase (optional for reading/writing trajectories), pymatgen
- h5py (for reading fc3 files)
- sklearn (optional — KernelDensity used if available)

Quick usage
-----------
From the repository root you can run the module as a script. Example:

```bash
python core/phonon/anharmonic_score.py \
  --md_xml vasprun.xml --ref_frame POSCAR --unit_cell_frame POSCAR \
  --md_time_step 1 --fc force_constants.hdf5 --third --fc3 ./phono3py/fc3.hdf5
```

This will create plots and print summary statistics (structural sigma,
frequency/group-velocity deviations) depending on flags passed.

API: `AnharmonicScore` class
--------------------------------
Constructor signature (important args):

```py
AnharmonicScore(
    ref_frame=None,
    unit_cell_frame=None,
    md_frames=None,
    potim=1,
    force_constants=None,
    supercell=[1,1,1],
    primitive_matrix=[[1,0,0],[0,1,0],[0,0,1]],
    atoms=None,
    include_third_order=False,
    third_order_fc='./phono3py/fc3.hdf5',
    include_fourth_order=False,
    fourth_order_fc=None,
    force_sets_filename='FORCE_SETS',
    mode_resolved=False
)
```

Constructor behavior and key arguments
- `ref_frame`: path to POSCAR/CONTCAR or a `Crystal` instance describing
  the reference (0 K) atomic positions. If a path is provided, `VaspReader`
  is used to read the file.
- `unit_cell_frame`: unit cell filename (used when loading phonopy hdf5). In this implementation, we need the `ref_frame` (which is supposed to be the supercell) to be the identical file as the `unit_cell_frame` as we didnt figure out how to do the supercell expansion.
- `md_frames`: list of MD trajectory files. Can be one or more `vasprun.xml`
  files (VASP MD) or ASE `.traj` files. The `.traj` file is typically
  generated when running MD with some MLFF foundation model.
- `potim`: MD timestep in fs (used for time series generation).
- `force_constants`: second-order force constants input. If a path to an
  HDF5 file is given, phonopy is used to load the phonon object; alternatively
  FORCE_SETS may be used (but may raise error!). If you follow the origian publication of the anharmonicity score, this is supposed to be the second order force constant from phonopy. However, you can also use the 2nd-order force constant obtained from TDEP, that will allow you to see how the vibrational anharmonicity is renormalised by temperature.
- `supercell` / `primitive_matrix`: used when constructing the phonopy
  object from HDF5.
- `atoms`: optional list of atomic labels to restrict structural sigma
  calculations to subsets of atoms.
- `include_third_order` / `third_order_fc`: if True, third-order forces
  are loaded from `third_order_fc` (HDF5 fc3) and included in the anharmonic
  force calculation.


Examples (programmatic):

```py
from core.phonon.anharmonic_score import AnharmonicScore

scorer = AnharmonicScore(
    md_frames=['vasprun.xml'],
    ref_frame='POSCAR',
    unit_cell_frame='POSCAR',
    potim=1.0,
    force_constants='force_constants.hdf5',
    mode_resolved=True
)

# produce joint PDF plot of DFT vs anharmonic forces
scorer.plot_total_joint_distribution(x='DFT', y='anh')

# compute structural sigma averaged across the whole trajectory
sigma, times = scorer.structural_sigma(return_trajectory=False)

# compute the time-dependent sigma across the trajectory such that its evolution may be plotted
sigma, times = scorer.structural_sigma(return_trajectory=True)

# This allows you to compute the sigma projected onto each phonon
# eigenvector and plot sigma as a function of phonon eigenfrequencies # next to a phonon dispersion relationship.
# see Chemistry of Materials 34 (20), 9072 (2022) for an example.
scorer.mode_resolved_sigma_band()
```



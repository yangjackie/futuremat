# Phonon (phonopy) utilities

Lightweight helpers and examples for generating finite-displacement supercells
and producing force-constants using the phonopy API, as well as performing anharmoinicity
quantification based on the paper "Anharmonicity measure for materials" published on Phys.
Rev. Maters. 4, 083809 (2020). This package contains
wrappers used by the repository (for example `phonopy_worker.py`) and a
reference README showing common workflows.


## Requirements

- Python 3.8+
- phonopy
- ase (optional, used for running calculators on displaced structures)
- A calculator (VASP, MACE, etc.) accessible from Python to evaluate forces
- mace-torch

Install basics with pip:

```bash
pip install phonopy ase pymatgen mace-torch
```

## Quick example — use a foundation model to compute the harmonic force constants

```python
import phonopy_worker
from mace.calculators import mace_mp

#define the mace mp calculator by specifying the path and model name to which the MACE foundation model is stored
calculator = mace_mp(model=args.model_path + "/" + args.model_name, device='cpu')

#read in structure from CONTCAR file, replace with your own structure file as needed
phonopy_worker = PhonopyWorker(structure=read("CONTCAR"),calculator=calculator)

#generate and save the force constants into a file for further analysis
phonopy_worker.generate_force_constants(save_fc=True,fc_file_name="force_constants-mace-mp-0b3-medium.hdf5")
```

This can also be called from command line as 
```bash
python3 phonopy_worker.py --model_path xxx --model_name mace-mp-0b3-medium.model --input_structure ./CONTCAR
```

## Comparing the phonon dispersion relationships computed with two different methods

The `prepare_and_plot` routine in the `phonopy_worker.py` provides a wrapper function which allows you to:

1. Quickly run a phonon force constant calculation with a foundation model, and
2. Plot the dispersion relationships from DFT and the foundation model, providing that the DFT force constants is also given.

See the documentations for the corresponding method on how to use it.


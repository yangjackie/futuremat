DEFAULT_STRUCTURE_OPTIMISATION_SET_FOR_PHONONS = {
    "SYSTEM": "Structure optimisation for phonons",
    "ISTART": 0,
    "ICHARG": 2,
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-8,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "FALSE",
    "GGA": "PS",
    "IBRION": 2,
    "ISIF": 3,
    "NSW": 200,
    "LASPH": True,
    "ADDGRID": True,
    "LWAVE": False,
    "LCHARG": False,
    "use_gw": True,
    "kpoint_mode": "grid",
    "kppa": 2000,
    "clean_after_success": True,
}

DEFAULT_STATIC_SET_FOR_PHONONS = {
    "SYSTEM": "Static calculation for phonons",
    "ISTART": 0,
    "ICHARG": 2,
    "ENCUT": 520,
    "PREC": "Accurate",
    "EDIFF": 1e-8,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LREAL": "FALSE",
    "GGA": "PS",
    "IBRION": -1,  # No ionic relaxation
    "NSW": 0,
    "LASPH": True,
    "ADDGRID": True,
    "LWAVE": False,
    "LCHARG": False,
    "use_gw": True,
    "kpoint_mode": "grid",
    "kppa": 2000,
    "clean_after_success": True,
}

DEFAULT_BORN_EFFECTIVE_CHARGES_SET_FOR_PHONONS = {
    "SYSTEM": "Born effective charges calculation",
    "IBRION": 8,  # DFPT for dielectric and Born effective charges
    "LEPSILON": True,  # Calculate dielectric tensor & Born effective charges
    "ISYM": 0,  # Disable symmetry (recommended for DFPT)
    "ENCUT": 520,  # Plane-wave cutoff (example; use appropriate value)
    "PREC": "Accurate",  # Precision level
    "EDIFF": 1e-8,  # Tight SCF convergence
    "LREAL": False,  # No real-space projection (recommended)
    "ISMEAR": 0,  # Gaussian smearing for insulators
    "SIGMA": 0.05,
    "GGA": "PS",
    "LASPH": True,
    "ADDGRID": True,
    "LWAVE": False,
    "LCHARG": False,
    "use_gw": True,
    "kpoint_mode": "grid",
    "kppa": 2000,
    "clean_after_success": True,
}

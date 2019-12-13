"""
The 2D library module, which builds slab models of two-dimensional perovskite structures with
different (a) orientations, (b) thicknesses and (c) surface terminations. Structures that
are built will be written to a directory with the corresponding set ups for VASP calculations.
"""

from ase.io import read, write
from ase.build import cut, stack, mx2, add_vacuum
from ase.io.vasp import write_vasp

from core.dao.vasp import *
from core.internal.builders.crystal import *
import os

from twodPV.elements import A_site_list, B_site_list, C_site_list

thicknesses = [3, 5, 7, 9]

orientation_dict = {  '100': {'a': (1, 0, 0), 'b': (0, 1, 0),
            'origio': {'AO': (0, 0, 0), 'BO2': (0, 0, 0.25)}},
    '111': {'a': (1, 1, 0), 'b': (-1, 0, 1),
            'origio': {'AO3': (0, 0, 0), 'B': (0, 0, 0.25)}}}

cwd = os.getcwd()

for i in range(len(A_site_list)):
    for a in A_site_list[i]:
        for b in B_site_list[i]:
            for c in C_site_list[i]:
                wd = cwd + '/relax_Pm3m/' + a + b + c + '_Pm3m' + '/'
                crystal = read(wd + 'CONTCAR', format='vasp')

                for orient in orientation_dict.keys():
                    for terminations in orientation_dict[orient]['origio'].keys():
                        for thick in thicknesses:
                            print(a, b, c, orientation_dict[orient]['a'], orientation_dict[orient]['b'], thick)
                            slab = cut(crystal,
                                       a=orientation_dict[orient]['a'],
                                       b=orientation_dict[orient]['b'],
                                       nlayers=thick,
                                       origo=orientation_dict[orient]['origio'][terminations])
                            add_vacuum(slab, 40)

                            slab_wd = cwd + '/slab_' + str(orient) + '_' + str(
                                terminations) + '_small/' + a + b + c + "_" + str(thick) + '/'

                            if not os.path.exists(slab_wd):
                                os.makedirs(slab_wd)

                            os.chdir(slab_wd)
                            write_vasp('POSCAR', slab, vasp5=True, sort=True)
                            os.chdir(wd)

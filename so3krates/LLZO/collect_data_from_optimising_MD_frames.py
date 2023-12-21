import os
import zipfile
import glob
import ase
import ase.io
import numpy as np

pwd = os.getcwd()
all_zips = glob.glob("frame_*.zip")

total_number_of_data = 0

R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask = [[] for _ in range(9)]

for zip in list(sorted(all_zips)):
    with zipfile.ZipFile(zip) as zf:
        with open("./vasprun_temp.xml", 'wb') as f:
            f.write(zf.read(zip.replace('.zip','')+"/vasprun.xml"))
    f.close()

    for atoms in ase.io.read("./vasprun_temp.xml", format='vasp-xml', index='1:'):
        R.append(atoms.get_positions())
        F.append(atoms.get_forces())
        E.append(atoms.get_potential_energy())
        z.append(atoms.get_atomic_numbers())
        pbc.append(atoms.get_pbc())
        unit_cell.append(atoms.get_cell())
        node_mask.append([True for _ in range(len(R[-1]))])

    print("Total number of data points ", len(R))
    #total_number_of_data += len(R)

    os.remove("./vasprun_temp.xml")

np.savez('opt_data_400K.npz', R=R, F=F, E=E, z=z, pbc=pbc, unit_cell=unit_cell, node_mask=node_mask)
~                                                                                                     

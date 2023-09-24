import os
import subprocess
import os
import tarfile
import ase.io
import ase
import numpy as np
import shutil
# Define the directory containing the folders with disp_* names
root_dir = "/scratch/dy3/jy8620/LLZO/pure/"


# Iterate through the folders
for folder_name in list(sorted(os.listdir(root_dir))):
    if folder_name.startswith("disp_") and os.path.isdir(os.path.join(root_dir, folder_name)):
        # Construct the full path to the folder
        folder_path = os.path.join(root_dir, folder_name)

        # Find all .tar.gz files in the folder
        tar_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".tar.gz")]
        R, F, E, z, pbc, unit_cell, idx_i, idx_j, node_mask= [[] for _ in range(9)]
        for tar_count,tar_file in enumerate(tar_files):
            tar_file_path = os.path.join(folder_path, tar_file)
            print(f"{tar_count}/{len(tar_files)}, Processing {tar_file_path}...")

            _path=os.path.join(folder_path, tar_file[:-7])

            # Extract the gz file
            with tarfile.open(tar_file_path, 'r') as tar:
                tar.extractall(path=folder_path)

            # Read the OUTCAR file if it exists
            try:
               outcar_file = os.path.join(_path, 'OUTCAR')
               if os.path.exists(outcar_file):
                  atoms = ase.io.read(os.path.join(_path, 'OUTCAR'), format='vasp-out')
                  # Process the contents of OUTCAR as needed
                  R.append(atoms.get_positions())
                  F.append(atoms.get_forces())
                  E.append(atoms.get_potential_energy())
                  z.append(atoms.get_atomic_numbers())
                  pbc.append(atoms.get_pbc())
                  unit_cell.append(atoms.get_cell())
                  node_mask.append([False for _ in range(len(R[-1]))])

                  print(E[-1])
            except:
                print('Error parsing OUTCAR, skipped')

            # Clean up the extracted files if needed
            try:
               shutil.rmtree(os.path.join(_path))
            except:
                pass

            print(f"Finished processing {tar_file_path}")
        R, F, E, z, pbc, unit_cell, node_mask = np.array(R),np.array(F),np.array(E),np.array(z),np.array(pbc),np.array(unit_cell),np.array(node_mask)
        np.savez(folder_name,R=R,F=F,E=E,z=z,pbc=pbc,unit_cell=unit_cell,node_mask=node_mask)

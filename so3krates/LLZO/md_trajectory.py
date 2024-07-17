import h5py
import matplotlib.pyplot as plt

time_step=0.5
traj = h5py.File('trajectory.h5')
potential_energies=traj['potential_energy']
plt.plot([time_step*i/1000.0 for i in range(len(potential_energies))],potential_energies,'b-')
plt.xlabel('Time (ps)')
plt.ylabel('Energy (eV)')
plt.tight_layout()
plt.savefig('energy_trajectory.pdf')
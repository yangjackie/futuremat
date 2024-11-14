import h5py
import matplotlib.pyplot as plt

time_step=1.0
traj = h5py.File('trajectory.h5')
print(traj.keys())
potential_energies=traj['potential_energy']
kinetic_energies=traj['kinetic_energy']
total_energies=[potential_energies[i]+kinetic_energies[i] for i in range(len(potential_energies))]
plt.plot([time_step*i/1000.0 for i in range(len(total_energies))],total_energies,'b-')
plt.xlabel('Time (ps)')
plt.ylabel('Energy (eV)')
plt.ylim([-1493,-1489])
plt.tight_layout()
plt.savefig('energy_trajectory.pdf')
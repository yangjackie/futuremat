import os,math
from core.dao.vasp import VaspReader

from matplotlib import rc
from scipy.interpolate import interp1d
rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

params = {'legend.fontsize': '16',
          'figure.figsize': (6, 5),
          'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}

pylab.rcParams.update(params)

def get_geometry_corrected_electronic_polarizability(directory=None):
    if directory is None:
        directory = os.getcwd()

    #get the raw data for dielectric tensor
    outcar_reader = VaspReader(input_location=directory+'/OUTCAR.RPA.DIAG')
    dielectric_tensor = outcar_reader.get_rpa_frequency_dependent_dielectric_tensor_from_outcar()

    #find out the length of the vacuum layer in the supercell
    #This is done in a very crude way by using the length of the supercell along the c-direction minus
    # the thickness of the slab
    structure = VaspReader(input_location=directory+'/POSCAR').read_POSCAR()
    c = structure.lattice.c
    z_coords = [a.position.z for a in structure.all_atoms()]

    for i in range(len(z_coords)):
        if abs(z_coords[i]-c)<=1:
            z_coords[i] = z_coords[i]-c # crude implementation of PBC

    slab_thickness = max(z_coords)-min(z_coords)
    vacuum_thickness = c - slab_thickness

    alpha_2d = {}
    for e in dielectric_tensor.keys():
        epsilon_inplane = 0.5*(dielectric_tensor[e]['xx']+dielectric_tensor[e]['yy'])
        epsilon_outplane = dielectric_tensor[e]['zz']

        alpha_2d[e] = {"inplane":(vacuum_thickness/(4.0*math.pi))*(epsilon_inplane-1.0),
                       "outplane":(vacuum_thickness/(4.0*math.pi))*(1.0-1.0/epsilon_outplane)}
    return alpha_2d

def plot_frequency_dependent_electronic_polarizability(directory=None,output='dielectric.pdf'):
    fig, ax1 = plt.subplots(figsize=(6,5))
    if directory is None:
        directory = os.getcwd()
    alpha_2d = get_geometry_corrected_electronic_polarizability(directory=directory)
    energies = list(sorted(alpha_2d.keys()))
    if max(energies)>100:
        max_en = 20

    else:
        max_en = max(energies)
    xnew = np.linspace(0, max_en, num=600, endpoint=True)
    x = energies
    y = [alpha_2d[e]['inplane'].imag for e in energies]
    f = interp1d(x, y, kind='cubic')
    ax1.plot(xnew, f(xnew), 'b-')
    ax1.set_ylabel('$\\alpha_{2D}^{\\parallel}$',color='b')
    ax1.set_xlabel('Energy (eV)')
    ax1.tick_params(axis='y', labelcolor='b')
    #plt.plot(energies,y,'bo')
    y = [alpha_2d[e]['outplane'].imag for e in energies]
    f = interp1d(x, y, kind='cubic')
    ax2=ax1.twinx()
    ax2.plot(xnew, f(xnew), 'r-')#, label='$\\alpha_{2D}^{\\perp}$')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylabel('$\\alpha_{2D}^{\\perp}$',color='r')

    #if max(energies)>100:
    ax1.set_xlim([-1,10])
    plt.tight_layout()
    plt.savefig(output)

if __name__=="__main__":
    plot_frequency_dependent_electronic_polarizability()
import os,argparse
from ase.db import connect
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pylab as pylab
params = {'legend.fontsize': '14',
          'figure.figsize': (7.5,6),
          'axes.labelsize': 24,
          'axes.titlesize': 24,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)



from analysis import halide_A,halide_B,halide_C,chalco_A,chalco_B,chalco_C

def formation_energy_sigma_landscape(systems='halides'):
    color_dict = {0: '#A3586D', 1: '#5C4A72', 2: '#F3B05A', 3: '#F4874B'}
    color_dict = {0: '#061283', 1: '#FD3C3C', 2: '#FFB74C', 3: '#138D90'}

    if systems not in ['halides', 'chalcogenides']:
        raise Exception("Wrong system specification, must be either halides or chalcogenides.")

    if systems == 'halides':
        A = halide_A
        B = halide_B
        C = halide_C
    elif systems == 'chalcogenides':  # including oxides
        A = chalco_A
        B = chalco_B
        C = chalco_C

    for c_counter, c in enumerate(C):
        db = connect('perovskites_updated_' + c + '.db')
        sigmas=[]
        pm3m_formation_es=[]
        for a in A:
            for b in B:
                row = None
                system_name = a + b + c
                uid = system_name + '_Pm3m'
                try:
                    row = db.get(selection=[('uid', '=', uid)])
                except:
                    continue

                sigma = None
                pm3m_formation_e = None
                if row is not None:
                    try:
                        sigma = row.key_value_pairs['sigma_300K_single']
                    except:
                        continue
                    try:
                        pm3m_formation_e = row.key_value_pairs['formation_energy']
                    except:
                        continue
                    if (sigma is not None) and (pm3m_formation_e is not None):
                        sigmas.append(sigma)
                        pm3m_formation_es.append(pm3m_formation_e)

        plt.scatter(pm3m_formation_es,sigmas,marker='o', c=color_dict[c_counter],edgecolor=None, alpha=0.45, s=25)

    plt.axhline(y=1, color='k', linestyle='--')

    from matplotlib.patches import Patch
    legend_elements=[]
    if systems == 'halides':
        legend_elements = [Patch(facecolor=color_dict[0], edgecolor='k', label='X=' + str(C[0])),
                           Patch(facecolor=color_dict[1], edgecolor='k', label='X=' + str(C[1])),
                           Patch(facecolor=color_dict[2], edgecolor='k', label='X=' + str(C[2])),
                           Patch(facecolor=color_dict[3], edgecolor='k', label='X=' + str(C[3]))]
    if systems == 'chalcogenides':
        legend_elements = [Patch(facecolor=color_dict[0], edgecolor='k', label='X=' + str(C[0])),
                           Patch(facecolor=color_dict[1], edgecolor='k', label='X=' + str(C[1])),
                           Patch(facecolor=color_dict[2], edgecolor='k', label='X=' + str(C[2]))]

    plt.legend(handles=legend_elements, loc=1, fontsize=12, ncol=1)

    if systems == 'halides':
        plt.ylim([0,2])
    elif systems == 'chalcogenides':
        plt.ylim([0,6])

    plt.xlabel("$\\Delta E_{f}$ (eV/atom)")
    plt.ylabel("$\\sigma^{(2)}$(300 K)")
    plt.tight_layout()
    plt.savefig("formation_energy_sigma_"+systems+'.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Switches for analyzing the screening results of bulk cubic perovskites',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default=os.getcwd() + '/perovskites.db',
                        help="Name of the database that contains the results of the screenings.")
    parser.add_argument("--C", type=str,
                        help="Anion in ABCs.")
    args = parser.parse_args()

    #if os.path.exists(args.db):
    #    args.db = connect(args.db)
    #else:
    #    raise Exception("Database " + args.db + " does not exists, cannot proceed!")

    formation_energy_sigma_landscape(systems='halides')
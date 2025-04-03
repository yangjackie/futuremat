import argparse
from mp_api.client import MPRester
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

rc('text', usetex=True)
import matplotlib.pylab as pylab

params = {'legend.fontsize': '15',
          'figure.figsize': (7, 6),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
pylab.rcParams.update(params)

# Replace with your Materials Project API key
API_KEY = "l3tSgHcRPO5Sf8pQQHGg3o6Q2ZJJDywb"


def get_compounds(anion='O', number_of_elements=2, theoretical=False):
    """Retrieve all structures from the Materials Project database that satisfy the chemistry criteria."""
    elements = [anion]
    mpr = MPRester(API_KEY)
    # Retrieve matching entries
    compounds = mpr.materials.summary.search(num_elements=number_of_elements, elements=elements,
                                             fields=['symmetry.number','structure','database_IDs','formula_pretty','material_id'], theoretical=theoretical)
    print("Total number of compounds: {}".format(len(compounds)))
    return compounds


if __name__ == "__main__":
    print("Main program for analysing statistics of inorganic crystal structures stored in the Materials Project")

    parser = argparse.ArgumentParser(description="Control for setting up the analysis for crystal framework analysis.")

    # Adding arguments
    parser.add_argument("--anion", type=str, default="O",
                        help="anion of the compound space that you would like to analyse.")
    parser.add_argument("--num_elements", type=int, default=2, help="total number of elements in the compound.")

    parser.add_argument("--total_stat", action="store_true",
                        help="plot statistics of all compounds as per total number of elements in the formula unit.")
    parser.add_argument("--sg_stat", action="store_true", help="plot statistics broken down into individual space group.")
    # Parse arguments
    args = parser.parse_args()

    # compounds = get_compounds(anion=args.anion, number_of_elements=args.num_elements)
    # print(compounds[0].formula_pretty)
    # print(compounds[0].symmetry.number)

    if args.total_stat:
        total_counts = []
        num_elements = [2, 3, 4, 5]
        for num_elem in num_elements:
            compounds = get_compounds(anion=args.anion, number_of_elements=num_elem)
            total_counts.append(len(compounds))
            print("Number of elements: {}".format(num_elem) + " giving total number of compounds: {}".format(
                len(compounds)))

        colors = plt.cm.viridis(np.linspace(0, 1, len(num_elements)))  # Using the Viridis colormap
        plt.figure(figsize=(8, 5))
        bars = plt.bar(num_elements, total_counts, color=colors, width=0.6, edgecolor="black")
        # Add labels on top of each bar
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                     bar.get_height(),  # Y position (top of bar)
                     f'{int(bar.get_height())}',  # Label (y value)
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.xticks([2, 3, 4, 5], [2, 3, 4, 5])
        plt.xlabel("Number of elements in the formula unit")
        plt.ylabel("Number of structures in MP")
        plt.tight_layout()
        plt.show()
    elif args.sg_stat:
        spacegroups=range(1,231,1)
        num_elements = [2, 3, 4, 5]
        all_counts = []
        for num_elem in num_elements:
            this_sg_counts = [0 for _ in range(len(spacegroups))]
            compounds = get_compounds(anion=args.anion, number_of_elements=num_elem)
            for compound in compounds:
                this_sg_counts[compound.symmetry.number-1] += 1
            all_counts.append(this_sg_counts)

        bottom = np.zeros(len(spacegroups))
        colors = plt.cm.viridis(np.linspace(0, 1, len(num_elements)))
        plt.figure(figsize=(10, 6))
        for i, y in enumerate(all_counts):
            plt.bar(spacegroups, y, bottom=bottom, color=colors[i], width=0.9, edgecolor=None,
                    label="$N=$"+str(num_elements[i]),alpha=0.8)
            bottom += np.array(y)
        plt.legend(loc=1)
        plt.xlabel("Space Group Number")
        plt.ylabel("Number of structures in MP")
        plt.tight_layout()
        plt.show()
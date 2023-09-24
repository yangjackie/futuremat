from PredictG import PredictGibbsEnergy
import json
import os
from pandas import DataFrame

"""
Total formation energy dataset for H2O and NH3 at different temperatures calculated by FactSage/Calculate/Reaction.
Unit:eV.
"""
H2O_formation_energies = {"300": "-2.38046354",
                          "600": "-2.22966875",
                          "900": "-2.06395938",
                          "1200": "-1.890613542",
                          "1500": "-1.713179167",
                          "1800": "-1.533504167"
                          }
NH3_formation_energies = {"300": "-0.168062",
                          "600": "0.163906",
                          "900": "0.519782",
                          "1200": "0.883492",
                          "1500": "1.249247",
                          "1800": "1.614672"
                          }


class CalculateGibbsEnergy:
    def __init__(self, path_to_pair_file):
        self.path_to_pair_file = path_to_pair_file

    @staticmethod
    def calculate_gibbs_formation_energies(initial_formula, formation_energy_0k, temperature, volume):
        path_to_structure = False
        path_to_masses = 'masses.json'
        path_to_chempots = 'Gels.json'
        obj = PredictGibbsEnergy(initial_formula,
                                 formation_energy_0k,
                                 path_to_structure,
                                 path_to_masses,
                                 path_to_chempots)
        return obj.dG(temperature, vol_per_atom=volume)

    @staticmethod
    def binary_reaction_coefficients(reaction, list_of_indices):
        coefficients = None
        [a, b, c, d] = [i for i in list_of_indices]
        if reaction == "hydrolysis":
            coefficients = [1 / b, a * d / b / c, a / b / c, 1, a * d / b / c]
        elif reaction == "reduction":
            coefficients = [a / b / c, a * d / b / c, a / d, a * d / b / c]
        elif reaction == "nitridation":
            coefficients = [a / b, 1 / 2, 1 / b]
        return coefficients

    @staticmethod
    def ternary_reaction_coefficients(reaction, list_of_indices):
        coefficients = None
        [a, b, c, d, n] = [i for i in list_of_indices]
        if reaction == "hydrolysis":
            coefficients = [1 / c, d / c / n, 1 / c / n, 1, d / c / n - 3 / 2]
        elif reaction == "reduction":
            coefficients = [1 / c / n, d / c / n, a / c, b / c, d / c / n]
        elif reaction == "nitridation":
            coefficients = [a / c, b / c, 1 / 2, 1 / c]
        return coefficients

    def calculate_pair_formation_energies(self, temperature):
        """
        Calculate the Gibbs energies for each pair at certain temperature
        and use pandas to save and export data to .csv file
        """
        pair_gibbs_formation_energy_dict = {}
        nitride_formation_energy_per_atom_list = []
        oxide_formation_energy_per_atom_list = []
        with open(self.path_to_pair_file) as f:
            data = json.load(f)
        pair_gibbs_formation_energy_dict["nitride_id"] = [e["nitride_id"] for e in data["pairs"]]
        pair_gibbs_formation_energy_dict["nitride"] = [e["nitride_formula"] for e in data["pairs"]]
        pair_gibbs_formation_energy_dict["oxide_id"] = [e["oxide_id"] for e in data["pairs"]]
        pair_gibbs_formation_energy_dict["oxide"] = [e["oxide_formula"] for e in data["pairs"]]
        if temperature == 0:
            pair_gibbs_formation_energy_dict["nitride_formation_energy"] = [e["nitride_formation_energy_0K"]
                                                                            for e in data["pairs"]]
            pair_gibbs_formation_energy_dict["oxide_formation_energy"] = [e["oxide_formation_energy_0K"]
                                                                          for e in data["pairs"]]
        else:
            for i in data["pairs"]:
                nitride_energy = self.calculate_gibbs_formation_energies(i["nitride_formula"],
                                                                         i["nitride_formation_energy_0K"],
                                                                         int(temperature),
                                                                         i["nitride_volume_per_atom"])
                nitride_formation_energy_per_atom_list.append(nitride_energy)
                oxide_energy = self.calculate_gibbs_formation_energies(i["oxide_formula"],
                                                                       i["oxide_formation_energy_0K"],
                                                                       int(temperature),
                                                                       i["oxide_volume_per_atom"])
                oxide_formation_energy_per_atom_list.append(oxide_energy)
            pair_gibbs_formation_energy_dict["nitride_formation_energy"] = nitride_formation_energy_per_atom_list
            pair_gibbs_formation_energy_dict["oxide_formation_energy"] = oxide_formation_energy_per_atom_list
        return pair_gibbs_formation_energy_dict

    def save_formation_energy(self, system, temperature):
        pair_gibbs_formation_energy_dict = self.calculate_pair_formation_energies(temperature)
        # change the name of keys using dict.pop()
        pair_gibbs_formation_energy_dict["nitride_formation_energy@" + str(temperature) + "K"] = \
            pair_gibbs_formation_energy_dict.pop("nitride_formation_energy")
        pair_gibbs_formation_energy_dict["oxide_formation_energy@" + str(temperature) + "K"] = \
            pair_gibbs_formation_energy_dict.pop("oxide_formation_energy")
        df = DataFrame(pair_gibbs_formation_energy_dict,
                       columns=['nitride_id', 'nitride', 'oxide_id', 'oxide',
                                "nitride_formation_energy@" + str(temperature) + "K",
                                "oxide_formation_energy@" + str(temperature) + "K"])
        cwd = os.getcwd()
        wd = cwd + '/data/' + system + '/formation_energies/'
        if not os.path.exists(wd):  # Test if directory exists.
            os.makedirs(wd)
        df.to_csv(wd + 'pair_formation_energies_' + str(temperature) + 'k.csv',
                  index=None, header=True)

    def calculate_reaction_energy(self, system, temperature, reaction_type):
        """
        This function is to calculate the hydrolysis reaction energy. In this case we consider G_f_H2
        """
        df = None
        coefficients = None
        dG_red = None
        looping_index = 0
        reaction_energy_list = []
        with open(self.path_to_pair_file) as f:
            data = json.load(f)
        pair_gibbs_formation_energy_dict = self.calculate_pair_formation_energies(temperature)
        oxide_energy_list = pair_gibbs_formation_energy_dict["oxide_formation_energy"]
        nitride_energy_list = pair_gibbs_formation_energy_dict["nitride_formation_energy"]
        if reaction_type == "hydrolysis":
            for i in data["pairs"]:
                if system == 'binary':
                    coefficients = self.binary_reaction_coefficients("hydrolysis", i["balance_equation"])
                elif system == 'ternary':
                    coefficients = self.ternary_reaction_coefficients("hydrolysis", i["balance_equation"])
                dG_hyd = coefficients[2] * oxide_energy_list[looping_index] * i["oxide_atom_num_per_formula"] + \
                         coefficients[3] * \
                         float(NH3_formation_energies[str(temperature)]) - \
                         (coefficients[0] * nitride_energy_list[looping_index] * i["nitride_atom_num_per_formula"] +
                          coefficients[1] *
                          float(H2O_formation_energies[str(temperature)]))
                reaction_energy_list.append(dG_hyd)
                looping_index += 1
            pair_gibbs_formation_energy_dict["hydrolysis_energy"] = reaction_energy_list
            df = DataFrame(pair_gibbs_formation_energy_dict,
                           columns=['nitride_id', 'nitride', 'oxide_id', 'oxide', "nitride_formation_energy",
                                    "oxide_formation_energy", "hydrolysis_energy"])
        elif reaction_type == "reduction":
            for i in data["pairs"]:
                if system == 'binary':
                    coefficients = self.binary_reaction_coefficients("reduction", i["balance_equation"])
                    dG_red = coefficients[3] * float(H2O_formation_energies[str(temperature)]) - \
                             coefficients[0] * oxide_energy_list[looping_index] * i["oxide_atom_num_per_formula"]
                elif system == 'ternary':
                    coefficients = self.ternary_reaction_coefficients("reduction", i["balance_equation"])
                    dG_red = coefficients[4] * float(H2O_formation_energies[str(temperature)]) - \
                             coefficients[0] * oxide_energy_list[looping_index] * i["oxide_atom_num_per_formula"]
                reaction_energy_list.append(dG_red)
                looping_index += 1
            pair_gibbs_formation_energy_dict["reduction_energy"] = reaction_energy_list
            df = DataFrame(pair_gibbs_formation_energy_dict,
                           columns=['nitride_id', 'nitride', 'oxide_id', 'oxide', "nitride_formation_energy",
                                    "oxide_formation_energy", "reduction_energy"])
        return df

    def save_reaction_energy(self, system, temperature, reaction_type):
        cwd = os.getcwd()
        wd = cwd + '/data/' + system + '/' + reaction_type + '_energies/'
        if not os.path.exists(wd):  # Test if directory exists.
            os.makedirs(wd)
        df = self.calculate_reaction_energy(system, temperature, reaction_type)
        df.to_csv(wd + 'pair_' + reaction_type + '_energies_' + str(temperature) + 'k.csv',
                  index=None, header=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='pair the compounds for solar thermal ammonia synthesis calculations')
    parser.add_argument('system', action='store', choices=['binary', 'ternary'],
                        type=str, help='choose the system to match the pairs.')
    parser.add_argument('reaction', action='store', choices=['hydrolysis', 'reduction', 'nitridation'],
                        type=str, help='choose the reaction to calculate the gibbs reaction energy.')
    parser.add_argument('--htem', action='store', type=int, default=600, help='change the hydrolysis temperature')
    parser.add_argument('--rtem', action='store', type=int, default=1800, help='change the reduction temperature')
    parser.add_argument('--ftem', action='store', type=int, default=600,
                        help='change the nitride formation/nitridation temperature')
    args = parser.parse_args()
    cge = None
    if args.system == 'binary':
        cge = CalculateGibbsEnergy(path_to_pair_file='data/binary/binary_nitride_oxide_pair.json')
    elif args.system == 'ternary':
        cge = CalculateGibbsEnergy(path_to_pair_file='data/ternary/ternary_nitride_oxide_pair.json')
    if args.reaction == 'nitridation':
        cge.save_formation_energy(args.system, args.ftem)
    elif args.reaction == 'hydrolysis':
        cge.save_reaction_energy(args.system, args.htem, args.reaction)
    elif args.reaction == 'reduction':
        cge.save_reaction_energy(args.system, args.rtem, args.reaction)

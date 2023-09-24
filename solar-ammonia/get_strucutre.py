"""
Retrieve the formation energies from Materials Project
"""

from pymatgen.ext.matproj import MPRester
import json
import os


def prepare_entries(key, anion, cation_number):
    """
    Get the ternary nitrides that consist of two metal components
    """
    mpr = MPRester(key)
    entries = None
    if cation_number == 1:
        criteria = {"elements": {"$in": ["Li", "Na", "K", "Rb", "Cs",
                                         "Be", "Mg", "Ca", "Sr", "Ba",
                                         "B", "Al", "Ga", "In", "Tl",
                                         "Si", "Ge", "Sn", "Pb",
                                         "P", "As", "Sb", "Bi",
                                         "S", "Se", "Te",
                                         "Sc", "Ti", "V", "Cr", "Mn",
                                         "Fe", "Co", "Ni", "Cu", "Zn",
                                         "Y", "Zr", "Nb", "Mo", "Tc",
                                         "Ru", "Rh", "Pd", "Ag", "Cd",
                                         "La", "Ce", "Pr", "Nd", "Pm",
                                         "Sm", "Eu", "Gd", "Tb", "Dy",
                                         "Ho", "Er", "Tm", "Yb", "Lu",
                                         "Hf", "Ta", "W", "Re", "Os",
                                         "Ir", "Pt", "Au", "Hg"]
                                 },
                    "nelements": 2}
        criteria['elements']['$all'] = list(anion)
        entries = mpr.query(criteria=criteria,
                            properties=["material_id", "pretty_formula", "formation_energy_per_atom"])
    elif cation_number == 2:
        if anion == "N":
            entries = mpr.query(criteria={"elements": {"$all": ["N"],
                                                       "$nin": ["H", "C", "O", "F", "Cl", "Br", "I",
                                                                "Th", "U", "Np", "Pu"]},
                                          "nelements": 3},
                                properties=["material_id", "pretty_formula", "formation_energy_per_atom"])
            nitride_mp_ids = [e['material_id'] for e in entries]
            print("The total number of nitrides: {}".format(len(nitride_mp_ids)))
        elif anion == "O":
            entries = mpr.query(criteria={"elements": {"$all": ["O"],
                                                       "$nin": ["H", "C", "N", "F", "Cl", "Br", "I",
                                                                "Th", "U", "Np", "Pu"]},
                                          "nelements": 3},
                                properties=["material_id", "pretty_formula", "formation_energy_per_atom"])
            oxide_mp_ids = [e['material_id'] for e in entries]
            print("The total number of oxides: {}".format(len(oxide_mp_ids)))
    return entries


def filter_stability(key, entries):
    # Select the materials with lowest formation energies
    mpr = MPRester(key)
    formation_energy_dict = {}
    min_id_list = []
    for i in entries:
        for j in entries:
            if i["pretty_formula"] == j["pretty_formula"]:
                formation_energy_dict[j["material_id"]] = j["formation_energy_per_atom"]
        min_id = min(formation_energy_dict, key=formation_energy_dict.get)
        min_id_list.append(str(min_id))
        formation_energy_dict.clear()
    min_id_list = list(dict.fromkeys(min_id_list))
    # prepare the new material ids with lowest formation energies
    new_entries = mpr.query(criteria={"task_id": {"$in": min_id_list}},
                            properties=["material_id", "pretty_formula",
                                        "formation_energy_per_atom", "volume",
                                        "unit_cell_formula", "e_above_hull"])
    return new_entries


def filter_atom_number(key, entries, atom_number):
    mpr = MPRester(key)
    compound_id_list = []
    for i in entries:
        if sum(i["unit_cell_formula"].values()) <= atom_number:
            compound_id_list.append(str(i["material_id"]))
    new_entries = mpr.query(criteria={"task_id": {"$in": compound_id_list}},
                            properties=["material_id", "pretty_formula",
                                        "formation_energy_per_atom", "volume",
                                        "unit_cell_formula", "e_above_hull"])
    return new_entries


def get_dataset(key, anion, cation_number, stable=None, number=None):
    compound_name_dict = {"N": "nitrides", "O": "oxides"}
    system_dict = {1: 'binary', 2: 'ternary'}
    properties_dict = {}
    dataset = {}
    compound_list = []
    entries = prepare_entries(key, anion, cation_number)
    if stable is None:
        pass
    else:
        entries = filter_stability(key, entries)
    if number is None:
        pass
    else:
        entries = filter_atom_number(key, entries, number)
    for i in entries:
        properties_dict['material_id'] = i["material_id"]
        properties_dict['formula'] = i["pretty_formula"]
        properties_dict['formation_energy'] = i["formation_energy_per_atom"]
        properties_dict['volume_per_atom'] = i["volume"] / sum(i['unit_cell_formula'].values())
        properties_dict['e_above_hull'] = i["e_above_hull"]
        compound_list.append(properties_dict.copy())
    dataset[compound_name_dict[anion]] = compound_list
    compounds = list(dataset.values())
    number_of_saved_materials = len(compounds[0])
    print("Save {} materials after the filtering.".format(number_of_saved_materials))
    cwd = os.getcwd()
    wd = cwd + '/data/' + system_dict[cation_number] + '/'
    if not os.path.exists(wd):
        os.makedirs(wd)
    os.chdir(wd)
    with open(compound_name_dict[anion] + '_' + system_dict[cation_number] + '_0k.json', 'w') as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='get the structures for solar thermal ammonia synthesis calculations')
    parser.add_argument('key', action='store', type=str, help='API key')
    parser.add_argument('anion', action='store', choices=['N', 'O'], type=str, help='define the anion in the compound')
    parser.add_argument('cation', action='store', choices=[1, 2], type=int, help='define the number of cations')
    parser.add_argument('--number', action='store', type=int,
                        help='the total number of atoms in the chemical composition')
    parser.add_argument('--stable', action='store_true', help='get the most structure with the lowest E above hull')
    args = parser.parse_args()
    get_dataset(args.key, args.anion, args.cation, stable=args.stable, number=args.number)

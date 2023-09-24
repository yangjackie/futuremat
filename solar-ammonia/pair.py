"""
Match the nitrides/oxides pairs based on the following reactions.
binary:
(1/b) M(a)N(b) + (ad)/(bc) H(2)O ---> a/(bc) M(c)O(d) + NH(3) + ((ad)/bc - 3/2) H(2)
ternary:
(1/c) M(a)M'(b)N(c) + (d/cn) H(2)O ---> (1/cn) M(na)M'(nb)O(d) + NH(3) + (d/cn - 3/2) H(2)
"""

import json
import os
from ase.formula import Formula


def match_binary(path_to_nitride_file, path_to_oxide_file):
    with open(path_to_nitride_file) as f:
        nitride_data = json.load(f)
    with open(path_to_oxide_file) as f:
        oxide_data = json.load(f)
    nitride_oxide_pair_properties = {}
    nitride_oxide_pair_list = []
    dataset = {}
    for i in nitride_data['nitrides']:
        nitride_dict = Formula(i['formula']).count()
        nitride_element_list = list(nitride_dict.keys())
        nitride_element_number_list = list(nitride_dict.values())
        nitride_atom_num_of_formula = sum(nitride_element_number_list)
        m_nitride = nitride_element_list[0]
        a = nitride_element_number_list[0]
        b = nitride_element_number_list[1]
        for j in oxide_data['oxides']:
            oxide_dict = Formula(j['formula']).count()
            oxide_element_list = list(oxide_dict.keys())
            nitride_element_number_list = list(oxide_dict.values())
            oxide_atom_num_per_formula = sum(nitride_element_number_list)
            m_oxide = oxide_element_list[0]
            c = nitride_element_number_list[0]
            d = nitride_element_number_list[1]
            hydrogen_index = a * d / b / c
            if m_nitride == m_oxide and hydrogen_index >= 3 / 2:
                nitride_oxide_pair_properties['nitride_id'] = i['material_id']
                nitride_oxide_pair_properties['nitride_formula'] = i['formula']
                nitride_oxide_pair_properties['nitride_formation_energy_0K'] = i['formation_energy']
                nitride_oxide_pair_properties['nitride_volume_per_atom'] = i['volume_per_atom']
                nitride_oxide_pair_properties['nitride_atom_num_per_formula'] = nitride_atom_num_of_formula
                nitride_oxide_pair_properties['oxide_id'] = j["material_id"]
                nitride_oxide_pair_properties['oxide_formula'] = j["formula"]
                nitride_oxide_pair_properties['oxide_formation_energy_0K'] = j["formation_energy"]
                nitride_oxide_pair_properties['oxide_volume_per_atom'] = j["volume_per_atom"]
                nitride_oxide_pair_properties['oxide_atom_num_per_formula'] = oxide_atom_num_per_formula
                nitride_oxide_pair_properties['balance_equation'] = [a, b, c, d]
                nitride_oxide_pair_list.append(nitride_oxide_pair_properties.copy())
                print("nitride:{}          oxide:{}".format(i['formula'], j['formula']))
    dataset["pairs"] = nitride_oxide_pair_list
    print("{} pairs have been matched.".format(len(nitride_oxide_pair_list)))
    cwd = os.getcwd()
    save_dir = cwd + '/data/binary/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'binary_nitride_oxide_pair.json', 'w') as f:
        json.dump(dataset, f, indent=2)


def match_ternary(path_to_nitride_file, path_to_oxide_file):
    with open(path_to_nitride_file) as f:
        nitride_data = json.load(f)
    with open(path_to_oxide_file) as f:
        oxide_data = json.load(f)
    nitride_oxide_pair_properties = {}
    nitride_oxide_pair_list = []
    dataset = {}
    for i in nitride_data['nitrides']:
        nitride_dict = Formula(i['formula']).count()
        nitride_element_list = list(nitride_dict.keys())
        nitride_element_number_list = list(nitride_dict.values())
        nitride_atom_num_of_formula = sum(nitride_element_number_list)
        a = nitride_element_number_list[0]
        b = nitride_element_number_list[1]
        nitride_element_list.remove("N")
        nitride_element_list.sort()
        nitride_metal_1 = nitride_element_list[0]
        nitride_metal_2 = nitride_element_list[1]
        nitride_cation_ratio = nitride_dict[str(nitride_metal_1)] / nitride_dict[str(nitride_metal_2)]
        for j in oxide_data['oxides']:
            oxide_dict = Formula(j['formula']).count()
            oxide_atom_num_per_formula = sum(oxide_dict.values())
            oxide_element_list = list(oxide_dict.keys())
            oxide_element_list.remove("O")
            oxide_element_list.sort()
            oxide_metal_1 = oxide_element_list[0]
            oxide_metal_2 = oxide_element_list[1]
            oxide_cation_ratio = oxide_dict[str(oxide_metal_1)] / oxide_dict[str(oxide_metal_2)]
            if nitride_element_list == oxide_element_list and nitride_cation_ratio == oxide_cation_ratio:
                c = nitride_dict["N"]
                d = oxide_dict["O"]
                n = oxide_dict[str(oxide_metal_1)] / nitride_dict[str(nitride_metal_1)]
                hydrogen_index = d / c / n - 3 / 2
                if hydrogen_index >= 0:
                    nitride_oxide_pair_properties['nitride_id'] = i['material_id']
                    nitride_oxide_pair_properties['nitride_formula'] = i['formula']
                    nitride_oxide_pair_properties['nitride_formation_energy_0K'] = i['formation_energy']
                    nitride_oxide_pair_properties['nitride_volume_per_atom'] = i['volume_per_atom']
                    nitride_oxide_pair_properties['nitride_atom_num_per_formula'] = nitride_atom_num_of_formula
                    nitride_oxide_pair_properties['nitride_e_above_hull'] = i['e_above_hull']
                    nitride_oxide_pair_properties['oxide_id'] = j["material_id"]
                    nitride_oxide_pair_properties['oxide_formula'] = j["formula"]
                    nitride_oxide_pair_properties['oxide_formation_energy_0K'] = j["formation_energy"]
                    nitride_oxide_pair_properties['oxide_volume_per_atom'] = j["volume_per_atom"]
                    nitride_oxide_pair_properties['oxide_atom_num_per_formula'] = oxide_atom_num_per_formula
                    nitride_oxide_pair_properties['oxide_e_above_hull'] = j['e_above_hull']
                    nitride_oxide_pair_properties['balance_equation'] = [a, b, c, d, n]
                    nitride_oxide_pair_list.append(nitride_oxide_pair_properties.copy())
                    print("nitride:{}          oxide:{}".format(i['formula'], j['formula']))
    dataset["pairs"] = nitride_oxide_pair_list
    print("{} pairs have been matched.".format(len(nitride_oxide_pair_list)))
    cwd = os.getcwd()
    save_dir = cwd + '/data/ternary/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'ternary_nitride_oxide_pair.json', 'w') as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='pair the compounds for solar thermal ammonia synthesis calculations')
    parser.add_argument('system', action='store', choices=['binary', 'ternary'],
                        type=str, help='choose the system to match the pairs.')
    args = parser.parse_args()
    if args.system == 'binary':
        nitride_file_path = Path('data/binary/nitrides_binary_0k.json')
        oxide_file_path = Path('data/binary/oxides_binary_0k.json')
        if nitride_file_path.is_file() and oxide_file_path.is_file():
            match_binary(nitride_file_path, oxide_file_path)
    elif args.system == 'ternary':
        nitride_file_path = Path('data/ternary/nitrides_ternary_0k.json')
        oxide_file_path = Path('data/ternary/oxides_ternary_0k.json')
        if nitride_file_path.is_file() and oxide_file_path.is_file():
            match_ternary(nitride_file_path, oxide_file_path)

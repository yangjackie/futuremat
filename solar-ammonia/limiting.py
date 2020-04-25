"""
Determine the limiting energy for the three-step solar thermochemical synthesis process
"""
import pandas as pd
from pandas import DataFrame
import os


def find_files(system, t_hyd, t_red, t_nit):
    cwd = os.getcwd()
    path_to_hyd_file = cwd + "/data/" + system + "/hydrolysis_energies/pair_hydrolysis_energies_" + str(t_hyd) + "k.csv"
    path_to_red_file = cwd + "/data/" + system + "/reduction_energies/pair_reduction_energies_" + str(t_red) + "k.csv"
    path_to_nit_file = cwd + "/data/" + system + "/formation_energies/pair_formation_energies_" + str(t_nit) + "k.csv"
    path_to_for_file = cwd + "/data/" + system + "/formation_energies/pair_formation_energies_0k.csv"
    return path_to_hyd_file, path_to_red_file, path_to_nit_file, path_to_for_file


def find_limiting_reaction(system, t_hyd, t_red, t_nit):
    pair_gibbs_limiting_energy_dict = {}
    index_list = []
    limiting_energy_list = []
    path_to_hyd_file, path_to_red_file, path_to_nit_file, path_to_for_file = find_files(system, t_hyd, t_red, t_nit)
    df_nit = pd.read_csv(path_to_nit_file)
    df_hyd = pd.read_csv(path_to_hyd_file)
    df_red = pd.read_csv(path_to_red_file)
    df_for = pd.read_csv(path_to_for_file)
    pair_gibbs_limiting_energy_dict["nitride_id"] = [df_hyd.iloc[e, 0] for e in range(df_hyd['nitride'].size)]
    pair_gibbs_limiting_energy_dict["nitride"] = [df_hyd.iloc[e, 1] for e in range(df_hyd['nitride'].size)]
    pair_gibbs_limiting_energy_dict["oxide_id"] = [df_hyd.iloc[e, 2] for e in range(df_hyd['nitride'].size)]
    pair_gibbs_limiting_energy_dict["oxide"] = [df_hyd.iloc[e, 3] for e in range(df_hyd['nitride'].size)]
    pair_gibbs_limiting_energy_dict["nitride_formation_energy@0K"] = [df_for.iloc[e, 4] for e in
                                                                      range(df_for['nitride_formation_energy@0K'].size)]
    pair_gibbs_limiting_energy_dict["oxide_formation_energy@0K"] = [df_for.iloc[e, 5] for e in
                                                                    range(df_for['oxide_formation_energy@0K'].size)]
    for i in range(df_for['nitride'].size):
        hydrolysis_energy = df_hyd.iloc[i, 6]
        reduction_energy = df_red.iloc[i, 6]
        nitridation_energy = df_nit.iloc[i, 4]
        reaction_energies_list = [hydrolysis_energy, reduction_energy, nitridation_energy]
        limiting_energy = max(reaction_energies_list)
        index = reaction_energies_list.index(max(reaction_energies_list))
        # Index for the step with limiting energy: {0: hydrolysis, 1: reduction, 2: nitridation}
        limiting_energy_list.append(limiting_energy)
        index_list.append(index)
        if limiting_energy < 0:
            print("limiting reaction: {}  pair:{}   energy: {}".format(index,
                                                                       [df_hyd.iloc[i, 1], df_hyd.iloc[i, 3]],
                                                                       limiting_energy))
    pair_gibbs_limiting_energy_dict["limiting_energy"] = limiting_energy_list
    pair_gibbs_limiting_energy_dict["limiting_reaction_index"] = index_list
    df = DataFrame(pair_gibbs_limiting_energy_dict,
                   columns=['nitride_id', 'nitride', 'oxide_id', 'oxide', "nitride_formation_energy@0K",
                            "oxide_formation_energy@0K", "limiting_energy", "limiting_reaction_index"])
    cwd = os.getcwd()
    wd = cwd + '/data/' + system + '/limiting_energies/'
    if not os.path.exists(wd):  # Test if directory exists.
        os.makedirs(wd)
    df.to_csv(wd + 'limiting_energies_' + str(t_hyd) + "_" + str(t_red) + "_"
              + str(t_nit) + '.csv', index=None, header=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='determine the limiting energies.')
    parser.add_argument('system', action='store', type=str, choices=['binary', 'ternary'])
    parser.add_argument('htem', action='store', type=int, default=600, help='choose the temperature for hydrolysis.')
    parser.add_argument('rtem', action='store', type=int, default=1800, help='choose the temperature for reduction.')
    parser.add_argument('ntem', action='store', type=int, default=600, help='choose the temperature for nitridation.')
    args = parser.parse_args()
    find_limiting_reaction(args.system, args.htem, args.rtem, args.ntem)

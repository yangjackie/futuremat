import os
import argparse
import tarfile
import shutil
from core.dao.vasp import *

from core.utils.loggings import setup_logger
parser = argparse.ArgumentParser(
    description='Switches for submitting MD calculation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--id", type=int)
args = parser.parse_args()

halide_C = ['F', 'Cl', 'Br', 'I']
halide_A = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Cu', 'Ag', 'Au', 'Hg', 'Ga', 'In', 'Tl']
halide_B = ['Mg', 'Ca', 'Sr', 'Ba', 'Se', 'Te', 'As', 'Si', 'Ge', 'Sn', 'Pb', 'Ga', 'In', 'Sc', 'Y', 'Ti', 'Zr', 'Hf',
            'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt',
            'Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg']

chalco_C = ['O', 'S', 'Se']
chalco_A = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag', 'Zn',
            'Cd', 'Hg', 'Ge', 'Sn', 'Pb']
chalco_B = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
            'Ni', 'Pd', 'Pt', 'Sn', 'Ge', 'Pb', 'Si', 'Te', 'Po']

A_site_list = [list(sorted(halide_A)), list(sorted(chalco_A))]
B_site_list = [list(sorted(halide_B)), list(sorted(chalco_B))]
C_site_list = [list(sorted(halide_C)), list(sorted(chalco_C))]

equilibrium_set = {"ALGO": "Fast",
                   "PREC": "Accurate",
                   "LREAL": "AUTO",
                   "ISMEAR": "0",
                   "ISYM": "0",
                   "IBRION": "0",
                   "MAXMIX": "40",
                   "NCORE": "14",
                   "NELMIN": "4",
                   "NSW": "100",
                   "SMASS": "-1",
                   "ISIF": "1",
                   "TEBEG": "10",
                   "TEEND": "300",
                   "POTIM": "1",
                   "NBLOCK": "10",
                   "NWRITE": "0",
                   "LCHARG": 'False',
                   "LWAVE": 'False',
                   "IWAVPR": "11",
                   "ENCUT": "300"}

production_set = {"ALGO": "Fast",
                  "PREC ": " Accurate",
                  "ENCUT": "300",
                  "LREAL": "AUTO",
                  "ISMEAR": "0",
                  "ISYM": "0",
                  "IBRION": "0",
                  "MAXMIX ": " 40",
                  "NCORE": "14",
                  "NELMIN": "4",
                  "NSW": "800",
                  "ISIF": "1",
                  "TEBEG": "300",
                  "TEEND": "300",
                  "POTIM": " 2",
                  "NBLOCK": "1",
                  "MDALGO": "1",
                  "ANDERSEN_PROB": "0.5",
                  "NWRITE": "0",
                  "LCHARG": "False",
                  "LWAVE": "False",
                  "IWAVPR": "11"}

# Figure out which system to work on according to the id value
sns = []
for i in range(len(A_site_list)):
    for a in A_site_list[i]:
        for b in B_site_list[i]:
            for c in C_site_list[i]:
                sns.append(a + b + c)

system_name = sns[args.id] + "_Pm3m"

# untar the tar ball
cwd = os.getcwd()
try:
    tf = tarfile.open(system_name + '_Pm3m.tar.gz')
    tf.extractall()
    os.chdir(system_name + '_Pm3m')
    logger = setup_logger(output_filename='md_run.log')
    structure_to_run = VaspReader(input_location='./POSCAR-md')
except:
    raise Exception(system_name + '_Pm3m' + ' tar ball not working')

# check that if a previous run has completed
md_output = 'vasprun_md_accurate.xml'
md_name = 'MD_accurate'
if os.path.isfile('./' + md_output):  # and os.path.isfile('./' + md_name + '.tar.gz'):
    logger.info("Previous run successful, skipping...")
    pass
else:
    os.mkdir('./' + md_name)
    os.chdir('./' + md_name)

    writer = VaspWriter()
    logger.info("write POSCAR")
    writer.write_structure(structure_to_run, filename='POSCAR', magnetic=False)
    logger.info("write POTCAR")
    writer.write_potcar(structure_to_run, sort=False, unique=True, magnetic=False, use_GW=False)
    logger.info("write Gamma point only KPOINTS")
    writer.write_KPOINTS(structure_to_run, K_points=[1, 1, 1], gamma_centered=True)

    #equilibration run

    try:
        os.remove('./INCAR')
    except:
        pass

    logger.info("EQUILIBRIUM RUN")
    logger.info("start a trial run see if there is any convergence issue")

    writer.write_INCAR('INCAR', default_options=production_set.update({'NSW':"10","TEBEG": "10","TEEND": "10"}))
    cmd = 'mpirun vasp_gam'
    os.system('%s > %s' % (cmd, 'vasp.log'))

    #check if SCF converges within 60 cycles
    no_covergence_issues = False
    oszicar = open('OSZICAR','r')
    start_count = False
    for l in oszicar.readlines():
        if '10 T=':
            start_count = True
        if start_count:
            if ('DAV' in l) or ('RMM in l'):
                scf_cycles = int(l.split()[1])
    if scf_cycles<60:
        no_covergence_issues = True

    if no_covergence_issues:
        logger.info("does not seem to have SCF issues, let's do a proper run")
        os.remove('./INCAR')
        writer.write_INCAR('INCAR', default_options=equilibrium_set)
        cmd = 'mpirun vasp_gam'
        os.system('%s > %s' % (cmd, 'vasp.log'))

        os.rename("./CONTCAR","./POSCAR")
        os.remove('./INCAR')
        writer.write_INCAR('INCAR', default_options=production_set.update({'NSW': "800", "TEBEG": "300", "TEEND": "300"}))
        cmd = 'mpirun vasp_gam'
        os.system('%s > %s' % (cmd, 'vasp.log'))

    #production run

    os.chdir("../")
# finish all the calculations, tar it back up and clear the directory.
os.chdir("..")  # step out this Pm3m folder
with tarfile.open(system_name + '_Pm3m.tar.gz', mode='w:gz') as archive:
    archive.add('./' + system_name + '_Pm3m', recursive=True)

try:
    shutil.rmtree(system_name + '_Pm3m')
except:
    pass
try:
    os.rmtree(system_name + '_Pm3m')
except:
    pass

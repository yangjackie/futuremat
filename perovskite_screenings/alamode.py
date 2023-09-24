# =======================================================================================================================
# ALAMODE Runner: Script that manage the workflow of running ALAMODE to extract effective force constants
#                 using self-consistent phonon (SCPH) approach and calculate lattice thermal conductivities
#                 for highly anharmonic materials
# =======================================================================================================================

import argparse, os, tarfile, shutil
from subprocess import STDOUT, check_output
from core.dao.vasp import VaspReader
from core.dao.alamode import AlamodeWriter
from pymatgen.io.vasp.outputs import Outcar
from perovskite_screenings.popen_timeout import PopenTimeout

parser = argparse.ArgumentParser(description='ALAMODE Runner Utilities',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-sz', '--systemzip',
                    help='zip that contains the system on which the calculation will be performed')
parser.add_argument('-ck', '--calculate_kappa', action='store_true',
                    help='is this a potential fitting or thermal cond. cal.; only the latter can run in parallel')
args = parser.parse_args()

current_dir = os.getcwd()

# =======================================================================================================================
# Untar the tarball that contains the system we want to work on
# =======================================================================================================================
if '.tar.gz' not in args.systemzip:
    args.systemzip = args.systemzip + '.tar.gz'
    system = args.systemzip
else:
    system = args.systemzip.replace('.tar.gz', '')
try:
    tf = tarfile.open(args.systemzip)
    tf.extractall()
except:
    raise Exception("Cannot work on system " + system + ' QUITTING')
os.chdir(system)

tf = tarfile.open('phonon_2_2_2.tar.gz')
tf.extractall()

unitcell_crystal = VaspReader(input_location='./CONTCAR').read_POSCAR()
supercell_crystal = VaspReader(input_location='./phonon_2_2_2/SPOSCAR').read_POSCAR()

# set up the alamode directory

if not os.path.isfile('./alamode.tar.gz'):
    if not os.path.exists('./alamode'):
        os.mkdir('alamode')
else:
    tf = tarfile.open('./alamode.tar.gz')
    tf.extractall()

os.chdir('alamode')

if not args.calculate_kappa:

    no_fit = False
    try:
        f = open('stage_2.out', 'r')
        for l in f.readlines():
            if '  RESIDUAL (%):' in l:
                print(system + ' ' + l)
                no_fit = True
        f.close()
    except:
        pass

    no_fit = False

    if not no_fit:
        # =======================================================================================================================
        # Get the harmonic forces
        # =======================================================================================================================
        os.system(
            "python3 /scratch/dy3/jy8620/alamode/tools/extract.py ../phonon_2_2_2/ph-POSCAR*/vasprun.xml --VASP ../phonon_2_2_2/SPOSCAR > DFSET_harmonic")

        # =======================================================================================================================
        # Get the MD forces
        # =======================================================================================================================
        os.system(
            "python3 /scratch/dy3/jy8620/alamode/tools/extract.py ../vasprun_md.xml --VASP ../phonon_2_2_2/SPOSCAR > DFSET")

        try:
            os.system("cp ../DFSET .")
        except:
            pass

        if os.path.isfile('DFSET') and os.path.isfile('DFSET_harmonic'):
            # =======================================================================================================================
            # Use ALM to fit the second order force constants from the DFSET_harmonic
            # =======================================================================================================================
            w = AlamodeWriter(supercell_crystal)
            alm_input_name = 'stage_1.in'
            alm_output_name = 'stage_1.out'
            second_fit_prefix = 'super_harm'
            w.write_alm_in_for_fitting_second_order(prefix=second_fit_prefix, input_name=alm_input_name, cutoff=[12])
            # os.system('source activate my-conda-env')
            # os.system('export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH')
            os.system("/scratch/dy3/jy8620/alamode/_build/alm/alm " + alm_input_name + " > " + alm_output_name)
            # os.system('conda deactivate')

            # =======================================================================================================================
            # Use ALM to fit the higher order force constants from the MD forces
            # =======================================================================================================================
            w = AlamodeWriter(supercell_crystal)
            alm_input_name = 'stage_2.in'
            alm_output_name = 'stage_2.out'
            high_order_fix_prefix = 'perovskite_300'
            w.write_alm_in_for_fitting_higher_order_FCs(prefix=high_order_fix_prefix,
                                                        second_fit_prefix=second_fit_prefix,
                                                        input_name=alm_input_name)
            # os.system("/scratch/dy3/jy8620/alamode/_build/alm/alm " + alm_input_name + " > " + alm_output_name)
            timeout = 3000
            # os.system('source activate my-conda-env')
            # os.system('export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH')
            exit_status = PopenTimeout(["/scratch/dy3/jy8620/alamode/_build/alm/alm", alm_input_name],
                                       output_file=open(alm_output_name, 'w')).run(timeout)
            # os.system('conda deactivate')
            try:
                f = open('stage_2.out', 'r')
                for l in f.readlines():
                    if '  RESIDUAL (%):' in l:
                        print(system + ' ' + l)
            except:
                pass
            # else:
            #    print("Problem with alm terminating on time in "+str(timeout)+" sec!")

else:

    # =======================================================================================================================
    # Get the Born effective charge
    # =======================================================================================================================
    born_effective_charges = None
    try:
        outcar = Outcar('../OUTCAR_born')
        outcar.read_lepsilon()
        born_effective_charges = outcar.born
    except:
        pass

    if born_effective_charges is not None:
        # write out the born file for alamode
        f = open('born_charges', 'w')
        for a in born_effective_charges:
            for b in a:
                for item in b:
                    f.write('{:.6f}'.format(item) + ' ')
                f.write("\n")
        f.close()

    # =======================================================================================================================
    # Use anphonon to perform self-consistent phonon calculations to renormalize the FC
    # =======================================================================================================================
    w = AlamodeWriter(unitcell_crystal)  # this one needs the primitive cell!!!
    alm_input_name = 'stage_3.in'
    alm_output_name = 'stage_3.out'
    scph_prefix = 'scph-run'
    high_order_fix_prefix = 'perovskite_300'
    if born_effective_charges is not None:
        born_info = 'born_charges'
    else:
        born_info = None
    w.write_alm_in_for_scph(prefix=scph_prefix, input_name=alm_input_name, high_order_fix_prefix=high_order_fix_prefix,
                            mode='mesh', mesh_grid=[10, 10, 10], born_info=born_info)
    os.system("/scratch/dy3/jy8620/alamode/_build/alm/alm " + alm_input_name + " > " + alm_output_name)

os.chdir('..')

with tarfile.open('alamode_quartic.tar.gz', mode='w:gz') as archive:
    archive.add('./alamode_quartic', recursive=True)

try:
    shutil.rmtree('phonon_2_2_2')
except:
    pass
try:
    os.rmtree('phonon_2_2_2')
except:
    pass

try:
    shutil.rmtree('alamode_quartic')
except:
    pass
try:
    os.rmtree('alamode_quartic')
except:
    pass

# =======================================================================================================================
# Calculation complete, re-tar the directory, clean up and exit
# =======================================================================================================================
os.chdir(current_dir)  # step out this Pm3m folder
with tarfile.open(args.systemzip, mode='w:gz') as archive:
    archive.add('./' + system, recursive=True)

try:
    shutil.rmtree(system)
except:
    pass
try:
    os.rmtree(system)
except:
    pass

exit()

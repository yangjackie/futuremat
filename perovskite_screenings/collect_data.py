import os,glob,tarfile,math,time
from core.calculators.vasp import Vasp
from core.dao.vasp import VaspReader, VaspWriter
from twodPV.collect_data import populate_db,_atom_dict
from pymatgen.io.vasp.outputs import Vasprun,BSVasprun
from pymatgen.electronic_structure.core import Spin
from sumo.electronic_structure.bandstructure import get_reconstructed_band_structure
from ase.db import connect
import shutil
from ase.io.vasp import *
import itertools
from core.external.vasp.anharmonic_score import *
from core.utils.loggings import setup_logger
logger = setup_logger(output_filename='data_collector.log')
#logger = setup_logger(output_filename='formation_energies_data.log')
#logger = setup_logger(output_filename='phonon_data.log')

halide_C=['F', 'Cl', 'Br', 'I']
halide_A=['Li', 'Na', 'K', 'Rb', 'Cs','Cu','Ag','Au','Hg','Ga','In','Tl']
halide_B=['Mg', 'Ca', 'Sr', 'Ba','Se','Te','As','Si','Ge','Sn','Pb','Ga','In','Sc','Y','Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Mn','Tc','Re','Fe','Ru','Os','Co','Rh','Ir','Ni','Pd','Pt','Cu','Ag','Au','Zn','Cd','Hg']

chalco_C=['O','S','Se']
chalco_A=['Be','Mg','Ca','Sr','Ba','Ra','Ti','V','Cr','Mn','Fe','Co','Ni','Pd','Pt','Cu','Ag','Zn','Cd','Hg','Ge','Sn','Pb']
chalco_B=['Ti','Zr','Hf','V','Nb','Ta','Cr','Mo','W','Mn','Tc','Re','Fe','Ru','Os','Co','Rh','Ir','Ni','Pd','Pt','Sn','Ge','Pb','Si','Te','Po']

#chalco_B=['Po']

A_site_list=[chalco_A, halide_A]
B_site_list=[chalco_B, halide_B]
C_site_list=[chalco_C, halide_C]


all_elements_list = list(itertools.chain(*[A_site_list, B_site_list, C_site_list]))
all_elements_list = list(itertools.chain(*all_elements_list))
all_elements_list = list(set(all_elements_list))

reference_atomic_energies={}

def element_energy(db):
    logger.info("========== Collecting reference energies for constituting elements ===========")
    cwd = os.getcwd()
    os.chdir('./elements')

    for element in all_elements_list:
        kvp = {}
        data = {}
        uid = 'element_' + str(element)
        logger.info(uid)
        tf = tarfile.open(element+'.tar.gz')
        tf.extractall()
        os.chdir(element)
      
        calculator = Vasp()
        calculator.check_convergence()
        if not calculator.completed:
           logger.info(uid,'failed')
        atoms = [i for i in read_vasp_xml(index=-1)][-1]
        e = list(_atom_dict(atoms).keys())[-1]
        reference_atomic_energies[element] = atoms.get_calculator().get_potential_energy() / _atom_dict(atoms)[e]
        kvp['uid'] = uid
        kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
        logger.info(uid+' '+str(reference_atomic_energies[element])+' eV/atom')
        populate_db(db, atoms, kvp, data)
        os.chdir("..")
        shutil.rmtree(element)
        try:
            os.rmtree(element)
        except:
            pass
    os.chdir('..')

def formation_energy(atoms):
    fe = atoms.get_calculator().get_potential_energy()
    #print(fe,_atom_dict(atoms),reference_atomic_energies)
    for k in _atom_dict(atoms).keys():
        fe = fe - _atom_dict(atoms)[k] * reference_atomic_energies[k]
     #   print(k,reference_atomic_energies[k],fe)
    return fe / atoms.get_number_of_atoms()

def full_relax_data(db):
    cwd = os.getcwd()
    system_counter = 0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue

                    try:
                        tf = tarfile.open('full_relax.tar.gz')
                        tf.extractall()

                    except:
                        logger.info(system_name + '_Pm3m' + ' full_relax tar ball not working')

                    if os.path.isdir('./full_relax'):
                        kvp={}
                        data={}
                        os.chdir('./full_relax')
                        try:
                            calculator = Vasp()
                            calculator.check_convergence()
                            atoms = None
                            if calculator.completed:
                                atoms = [a for a in read_vasp_xml(index=-1)][-1]
                                kvp['uid'] = uid + '_fullrelax'
                                kvp['total_energy'] = atoms.get_calculator().get_potential_energy()

                                kvp['formation_energy'] = formation_energy(atoms)
                                populate_db(db, atoms, kvp, data)
                                logger.info(system_name + '_Pm3m' + ' formation energy (fully relaxed): ' + str(kvp['formation_energy']) + ' eV/atom')
                        except:
                            logger.info(system_name + '_Pm3m' + ' formation energy (fully relaxed): NaN ')

                        os.chdir('..')

                    os.chdir("..")
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass


def all_data(db):
    cwd = os.getcwd()
    system_counter=0
    for i in range(len(A_site_list)):
        for a in A_site_list[i]:
            for b in B_site_list[i]:
                for c in C_site_list[i]:

                    system_counter += 1
                    logger.info("Working on system number: " + str(system_counter))
                    system_name = a + b + c
                    uid = system_name + '_Pm3m'

                    # open up the tar ball
                    cwd = os.getcwd()
                    try:
                        tf = tarfile.open(system_name + '_Pm3m.tar.gz')
                        tf.extractall()
                        os.chdir(system_name + '_Pm3m')
                    except:
                        logger.info(system_name + '_Pm3m' + ' tar ball not working')
                        continue

                    # get the formation energies for the randomised structures
                    try:
                        tf = tarfile.open('randomised.tar.gz')
                        tf.extractall()
                    except:
                        pass
                    # print(os.getcwd())
                    if os.path.isdir('./randomised'):
                        os.chdir('randomised')
                        # print(os.getcwd()+'\n')

                        for counter in range(10):
                            rkvp = {}
                            if os.path.isdir('./str_' + str(counter)):
                                os.chdir('./str_' + str(counter))
                                try:
                                    calculator = Vasp()
                                    calculator.check_convergence()
                                    atoms = None
                                    if calculator.completed:
                                        atoms = [a for a in read_vasp_xml(index=-1)][-1]
                                        rkvp['uid'] = uid + '_rand_str_' + str(counter)
                                        rkvp['total_energy'] = atoms.get_calculator().get_potential_energy()

                                        rkvp['formation_energy'] = formation_energy(atoms)
                                        populate_db(db, atoms, rkvp, data)
                                        logger.info(system_name + '_Pm3m' + ' formation energy (rand ' + str(
                                            counter) + '): ' + str(rkvp['formation_energy']) + ' eV/atom')
                                    # else:
                                    #    continue
                                except:
                                    logger.info(
                                        system_name + '_Pm3m' + ' formation energy (rand ' + str(counter) + '): ' + str(
                                            'NaN'))
                                os.chdir('..')
                        os.chdir('..')
                        try:
                            shutil.rmtree('randomised')
                        except:
                            pass
                        try:
                            os.rmtree('randomised')
                        except:
                            pass

                    kvp = {}
                    data = {}

                    #get the formation energies for the cubic Pm3m phase
                    get_properties=True
                    try:
                        calculator = Vasp()
                        calculator.check_convergence()
                        if calculator.completed:
                            atoms = [a for a in read_vasp_xml(index=-1)][-1]
                            kvp['uid'] = uid
                            kvp['total_energy'] = atoms.get_calculator().get_potential_energy()
                            kvp['formation_energy'] = formation_energy(atoms)
                            populate_db(db, atoms, kvp, data)
                            logger.info(system_name + '_Pm3m' + ' formation energy: '+str(kvp['formation_energy'])+' eV/atom')
                        else:
                            logger.info(system_name + '_Pm3m' + ' not converged in structure optimisation, not continuing ')
                            os.chdir("..")
                            try:
                                shutil.rmtree(system_name + '_Pm3m')
                            except:
                                pass
                            try:
                                os.rmtree(system_name + '_Pm3m')
                            except:
                                pass
                            get_properties=False
                    except:
                        logger.info(system_name + '_Pm3m' + ' formation energy: '+str('NaN'))

#                    os.chdir("..")
#                    try:
#                        shutil.rmtree(system_name + '_Pm3m')
#                    except:
#                        pass
#                    try:
#                        os.rmtree(system_name + '_Pm3m')
#                    except:
#                        pass

                    print(system_name + '_Pm3m'+' get_properties? '+str(get_properties))
                    if not get_properties: continue

                    #collect the electronic structure data (band gap)
                    try:
                        dos_tf = tarfile.open('dos.tar.gz')
                        dos_tf.extractall()
                    except:
                        pass

                    if os.path.isdir('./dos'): 
                       logger.info(system_name + '_Pm3m' + ' collect band gap')
                       os.chdir('./dos')
                       if os.path.exists('vasprun_BAND.xml'):
                           vr = BSVasprun('./vasprun_BAND.xml')
                           bs = vr.get_band_structure(line_mode=True, kpoints_filename='KPOINTS')
                           bs = get_reconstructed_band_structure([bs])
                           
                           # =====================================
                           # get band gap data
                           # =====================================
                           bg_data = bs.get_band_gap()
                           kvp['direct_band_gap']=bg_data['direct']
                           kvp['band_gap']=bg_data['energy']
                           logger.info(system_name + '_Pm3m' + ' direct band gap '+str(bg_data['energy'])+'  band gap energy:'+str(kvp['band_gap'])+' eV')
                           populate_db(db, atoms, kvp, data)

                       os.chdir('..')

                    #Check the phonon calculations are converged
                    force_constant_exists=os.path.isfile('force_constants.hdf5')
                    md_calculations_exists=os.path.isfile('vasprun_md.xml')
                    if force_constant_exists:
                       try:
                          phonon_tf = tarfile.open('phonon.tar.gz')
                          phonon_tf.extractall()
                       except:
                          pass

                    if os.path.isdir('./phonon'):
                       #Check that individual finite displacement calculation is well converged
                       phonon_converged=True
                       os.chdir('phonon')  
                       for sub_f in ['ph-POSCAR-001','ph-POSCAR-002','ph-POSCAR-003']:
                           os.chdir(sub_f)
                           f=open('./OUTCAR','r')
                           for l in f.readlines():
                               if 'NELM' in l:
                                  nelm=l.split()[2].replace(';','')
                                  nelm=int(nelm)
                           f.close()

                           f=open('./vasp.log','r')
                           lines=[]
                           for l in f.readlines():
                               if ('DAV:' in l) or ("RMM:" in l):
                                   lines.append(l)
                           if len(lines)>=nelm:
                              phonon_converged=False
                           os.chdir('..') # step out from str_# directory
                       os.chdir('..') # step out from phonon directory

                    md_done=False
                    if md_calculations_exists:
                       try:
                          md_tf = tarfile.open('MD.tar.gz')
                          md_tf.extractall()
                       except:
                          pass
                    if os.path.isdir('./MD'):
                       os.chdir('MD')
                       t=0
                       try:
                           f=open('./OSZICAR','r')
                           for l in f.readlines():
                               if 'T=' in l:
                                   t+=1
                           f.close()
                       except:
                           pass
                       if t==800:
                           md_done=True
                       os.chdir('..') #step out from MD directory

                    if force_constant_exists and phonon_converged and md_done:
                       logger.info(system_name + '_Pm3m' + ' Valid Phonon and MD Results')
                       uid = system_name + '_Pm3m'
                       kvp['uid'] = uid 
                       #calculate the anharmonic score
                       os.chdir('./phonon')

                       try:
                          #os.chdir('./phonon')
                          #temporatily rename force_constants.hdf5 file
                          os.rename('force_constants.hdf5','f.hdf5')
                          scorer = AnharmonicScore(md_frames='../vasprun_md.xml',ref_frame='./SPOSCAR',force_constants=None)
                          #__sigmas, _ = scorer.structural_sigma(return_trajectory=True)
                          __sigmas, _ = scorer.structural_sigma(return_trajectory=False)
                          print("sigma is "+str(__sigmas))
                          os.chdir('..')
                          kvp['sigma_300K']=True
                          kvp['sigma_300K_single']=__sigmas
                          #data['sigma_300K']=__sigmas
                          import numpy as np
                          logger.info(system_name + '_Pm3m' + ' anharmonic score done '+str(__sigmas))
                          populate_db(db, atoms, kvp, data)
                       except Exception as e:
                           print(e)
                           logger.info(system_name + '_Pm3m' + ' anharmonic score failed')
                           os.chdir('..')
                           pass
                           
                    else:
                       logger.info(system_name + '_Pm3m' + ' Invalid')   

                    os.chdir("..")
                    try:
                        shutil.rmtree(system_name + '_Pm3m')
                    except:
                        pass
                    try:
                        os.rmtree(system_name + '_Pm3m')
                    except:
                        pass

def collect(db):
    errors = []
    steps = [element_energy,full_relax_data]#all_data]
    steps = [all_data]
    for step in steps:
        try:
            step(db)
        except Exception as x:
            print(x)
            error = '{}: {}'.format(x.__class__.__name__, x)
            errors.append(error)
    return errors

if __name__ == "__main__":
   # We use absolute path because of chdir below!
   dbname = os.path.join(os.getcwd(), 'perovskites.db')
   db = connect(dbname)

   logger.info('Established a sqlite3 database object ' + str(db))
   collect(db)

